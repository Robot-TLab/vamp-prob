#pragma once

// 3-D visibility / observation-reward primitives for visibility-aware
// motion planning.
//
// Reformulation of the 2-D planar primitive that previously lived here
// (see git history pre-3D rewrite).  The math now operates on full 3-D
// camera poses and arbitrary 3-D Gaussian kernels, so a robot with
// pitch + yaw + roll neck DoFs can score visibility against kernels
// above / below the camera's elevation plane.
//
// Closed-form factorisation from the paper (eq:obs_reward_approx,
// eq:angular_fraction, eq:radial_fraction), extended to a 3-D cone:
//
//   O(c, n_gaze)  ≈  Σ_k  α_k · f_k^a(c, n_gaze) · f_k^r(c)
//
// where each ``k`` is one 3-D Gaussian kernel ``Gaussian3 = (μ_k, Σ_k,
// α_k)``.  Under the small-angle approximation (δ_k ≪ 1, σ_k ≪ r_k),
// the kernel's positional covariance ``Σ_k`` induces 2-D isotropic
// angular noise of variance ``(σ_k/r_k)² I_2`` in the local angular
// coordinates around the gaze axis; the probability mass that lands
// inside the FoV cone is therefore a non-central χ²₂ tail (Marcum-Q):
//
//   f_k^a  =  P[ χ²₂(λ_k) ≤ z_k ],
//             λ_k = (δ_k · r_k / σ_k)²,
//             z_k = (ψ/2 · r_k / σ_k)²,
//             δ_k = arccos⟨u_k, n_gaze⟩,
//             u_k = (μ_k − c) / r_k,
//             σ_k = √(tr(Σ_k)/3).
//
// The radial fraction is unchanged in form, with σ now per-kernel:
//
//   f_k^r  =  Φ((d_max − r_k) / σ_k).
//
// The 2-D paper's CDF-difference form is a 1-D projection of this
// construction and is recovered exactly when ``Σ_k`` collapses to a
// planar slice and the cone degenerates to a planar wedge.
//
// Robot-agnostic: ``c``, ``n_gaze``, and the kernel list are the only
// per-call inputs.  The camera world pose comes from the caller (a
// higher-level planner) via full-body FK; vamp doesn't know about
// joints or trajectories.  The kernel list is just ``Gaussian3<float>``,
// the same primary type collision uses (``GaussianObstacle`` slices
// down to it).  λ_s / λ_d priors stay outside vamp — callers fold
// them into each kernel's ``alpha`` weight upstream.
//
// No SDF / raycasting in v1.  The paper's factorisation drops
// occlusion masking; visibility against ``Environment.pointclouds``
// (map-based occlusion) is a separate primitive that can be added
// later without changing this API.
//
// SIMD: this primitive is scalar per Gaussian, matching the existing
// scalar style of ``normal_cdf`` / ``exp`` in ``collision/math.hh``.
// The collision pipeline's rake-of-configs SIMD parallelism lives at
// the FK layer (sphere_fk over 8 candidate configs at a time); the
// planner wrapper that calls this primitive should likewise FK in
// rake, extract per-lane camera poses, and call ``observation_reward``
// once per lane.  See ``vamp::planning::validate_config_risk`` for the
// analogous pattern on the collision side.

#include <cmath>
#include <cstddef>
#include <tuple>

#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // Camera-cone observation reward for camera at world position ``c``
    // looking along unit vector ``n_gaze`` (world frame).
    //
    //   c              camera position (cx, cy, cz)
    //   n_gaze         camera optical axis, unit vector (nx, ny, nz)
    //   d_max          maximum sensing range
    //   psi            full FoV cone angle (radians); half-angle = ψ/2
    //   gaussians      per-kernel (μ, Σ, α) Gaussian3 records
    //   n_gaussians    length of the kernel buffer
    //
    // Returns Σ_k α_k · f_k^a · f_k^r.  All kernels are evaluated
    // (each term is a small constant cost; no early exit).
    inline auto observation_reward(
        float cx,
        float cy,
        float cz,
        float nx,
        float ny,
        float nz,
        float d_max,
        float psi,
        const Gaussian3<float> *gaussians,
        std::size_t n_gaussians) noexcept -> float
    {
        constexpr float SIGMA_EPS = 1.0e-9F;  // floor avoids division-by-zero
                                              // on caller-supplied delta kernels
        const float half_psi = 0.5F * psi;
        float reward = 0.0F;

        for (std::size_t k = 0; k < n_gaussians; ++k)
        {
            const auto &g = gaussians[k];

            const float dx = g.mx - cx;
            const float dy = g.my - cy;
            const float dz = g.mz - cz;
            const float r2 = dx * dx + dy * dy + dz * dz;
            const float r_k = vamp::collision::sqrt(r2);

            const float sigma_k_raw = g.iso_sigma();
            const float sigma_k = (sigma_k_raw > SIGMA_EPS) ? sigma_k_raw : SIGMA_EPS;

            // Degenerate kernel sitting on the camera — contributes
            // the full mass if it's also within range.  (r_k ≈ 0
            // implies σ_k/r_k → ∞ angular noise, i.e. uniform on the
            // sphere; the cone captures fraction (1 − cos(ψ/2))/2 of
            // a uniform distribution.  For the visibility use case
            // we treat this as "fully visible if in range" — the
            // alternative would propagate NaN through atan2.)
            if (r_k < 1.0e-6F)
            {
                const float f_r0 = vamp::collision::normal_cdf(d_max / sigma_k);
                reward += g.alpha * f_r0;
                continue;
            }

            const float inv_r = 1.0F / r_k;
            const float ux = dx * inv_r;
            const float uy = dy * inv_r;
            const float uz = dz * inv_r;

            // δ_k = ∠(u_k, n_gaze) via atan2(‖u × n‖, ⟨u, n⟩) for
            // better precision near 0 and π than acos(dot).
            float cxn;
            float cyn;
            float czn;
            vamp::collision::cross_3(ux, uy, uz, nx, ny, nz, cxn, cyn, czn);
            const float cross_norm = vamp::collision::sqrt(cxn * cxn + cyn * cyn + czn * czn);
            const float dot_un = ux * nx + uy * ny + uz * nz;
            const float delta_k = std::atan2(cross_norm, dot_un);

            const float scale = r_k / sigma_k;
            const float lam_k = (delta_k * scale) * (delta_k * scale);
            const float z_k = (half_psi * scale) * (half_psi * scale);

            const float f_a = vamp::collision::ncx2_2_cdf(z_k, lam_k);
            const float f_r = vamp::collision::normal_cdf((d_max - r_k) / sigma_k);

            reward += g.alpha * f_a * f_r;
        }

        return reward;
    }

    // Internal helper for ``optimal_gaze``: build ``R_gaze · n_ref``
    // where ``R_gaze = R_head · Rz(az) · Ry(el)``.  ``R_head`` is
    // row-major (r00..r22).  No allocation; just six muls + adds.
    inline auto compose_gaze(
        float r00,
        float r01,
        float r02,
        float r10,
        float r11,
        float r12,
        float r20,
        float r21,
        float r22,
        float nx_ref,
        float ny_ref,
        float nz_ref,
        float az,
        float el,
        float &nx_out,
        float &ny_out,
        float &nz_out) noexcept -> void
    {
        const float ca = std::cos(az);
        const float sa = std::sin(az);
        const float ce = std::cos(el);
        const float se = std::sin(el);

        // n1 = Ry(el) · n_ref
        const float n1x = ce * nx_ref + se * nz_ref;
        const float n1y = ny_ref;
        const float n1z = -se * nx_ref + ce * nz_ref;

        // n2 = Rz(az) · n1
        const float n2x = ca * n1x - sa * n1y;
        const float n2y = sa * n1x + ca * n1y;
        const float n2z = n1z;

        // n_out = R_head · n2
        nx_out = r00 * n2x + r01 * n2y + r02 * n2z;
        ny_out = r10 * n2x + r11 * n2y + r12 * n2z;
        nz_out = r20 * n2x + r21 * n2y + r22 * n2z;
    }

    // Bounded 2-D search for the optimal gaze offsets ``(α*, β*)``
    // (head-frame yaw, pitch) maximising ``observation_reward``.
    //
    //   c                  camera world position
    //   R_head (r00..r22)  current head world-frame rotation
    //                      (row-major); the search composes
    //                      ``R_head · Rz(α) · Ry(β)`` on top.
    //   n_ref              optical-axis direction in head frame
    //                      (caller-supplied; matches CameraConfig).
    //   d_max, psi         FoV parameters, as in observation_reward.
    //   psi_h_az/el        full head-sweep widths in each axis.
    //   *_min/*_max        joint-limit clamps (radians).  The search
    //                      window per axis is the intersection of
    //                      ±ψ_h_*/2 and the joint clamp.
    //   gaussians, n       Gaussian kernel buffer.
    //   n_grid             coarse-scan resolution per axis (default 9
    //                      → 81 evaluations); set higher for very
    //                      narrow FoV / sparse kernels.
    //   n_refine           golden-section refinement iterations per
    //                      axis around the best grid cell.
    //
    // Returns ``(α*, β*, O*)``.  Strategy: coarse uniform grid scan
    // followed by per-axis golden-section refinement around the best
    // cell.  This is robust to the non-convex / sparse landscapes that
    // arise when ψ ≪ ψ_h (narrow cone over a wide head sweep, where
    // most of the search space gives reward 0 and pure GSS gets stuck
    // on the flat region).  Total cost: n_grid² + 2·n_refine evaluations
    // (default 81 + 40 = 121, well under a millisecond on a single
    // thread for typical kernel counts).
    inline auto optimal_gaze(
        float cx,
        float cy,
        float cz,
        float r00,
        float r01,
        float r02,
        float r10,
        float r11,
        float r12,
        float r20,
        float r21,
        float r22,
        float nx_ref,
        float ny_ref,
        float nz_ref,
        float d_max,
        float psi,
        float psi_h_az,
        float psi_h_el,
        float az_min,
        float az_max,
        float el_min,
        float el_max,
        const Gaussian3<float> *gaussians,
        std::size_t n_gaussians,
        int n_grid = 9,
        int n_refine = 20) noexcept -> std::tuple<float, float, float>
    {
        constexpr float INV_PHI = 0.6180339887498949F;  // 1 / golden ratio

        // Effective search window per axis = ±ψ_h_*/2 ∩ [*_min, *_max].
        const float half_psi_h_az = 0.5F * psi_h_az;
        const float half_psi_h_el = 0.5F * psi_h_el;
        const float az_lo = std::max(-half_psi_h_az, az_min);
        const float az_hi = std::min(half_psi_h_az, az_max);
        const float el_lo = std::max(-half_psi_h_el, el_min);
        const float el_hi = std::min(half_psi_h_el, el_max);

        // Empty window → clamp to the closest feasible point.
        const float az_lo_eff = (az_lo <= az_hi) ? az_lo : 0.5F * (az_lo + az_hi);
        const float az_hi_eff = (az_lo <= az_hi) ? az_hi : az_lo_eff;
        const float el_lo_eff = (el_lo <= el_hi) ? el_lo : 0.5F * (el_lo + el_hi);
        const float el_hi_eff = (el_lo <= el_hi) ? el_hi : el_lo_eff;

        auto reward_at = [&](float az, float el) -> float
        {
            float nx_g;
            float ny_g;
            float nz_g;
            compose_gaze(
                r00,
                r01,
                r02,
                r10,
                r11,
                r12,
                r20,
                r21,
                r22,
                nx_ref,
                ny_ref,
                nz_ref,
                az,
                el,
                nx_g,
                ny_g,
                nz_g);
            return observation_reward(
                cx, cy, cz, nx_g, ny_g, nz_g, d_max, psi, gaussians, n_gaussians);
        };

        if (n_grid < 2)
        {
            n_grid = 2;
        }

        // Coarse uniform scan.  Tracks the best cell + its neighbours
        // (one cell on each side, clamped to the search window) so the
        // refinement step has a bracketing interval guaranteed to
        // contain a local maximum.
        const float az_step = (az_hi_eff - az_lo_eff) / static_cast<float>(n_grid - 1);
        const float el_step = (el_hi_eff - el_lo_eff) / static_cast<float>(n_grid - 1);

        int best_i = 0;
        int best_j = 0;
        float best_o = -1.0F;
        for (int j = 0; j < n_grid; ++j)
        {
            const float el = el_lo_eff + el_step * static_cast<float>(j);
            for (int i = 0; i < n_grid; ++i)
            {
                const float az = az_lo_eff + az_step * static_cast<float>(i);
                const float o = reward_at(az, el);
                if (o > best_o)
                {
                    best_o = o;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Refinement bracket: one cell on each side of the best cell
        // (clamped to the search window).  This is convex enough for
        // golden-section to converge if ``observation_reward`` is
        // smooth across the bracket (which it is — σ-controlled
        // angular noise — once the bracket is small enough that no
        // kernel jumps in or out of the cone within it).
        auto bracket_lo = [&](int idx, int n, float lo, float step) -> float
        {
            const int prev = (idx > 0) ? idx - 1 : idx;
            return lo + step * static_cast<float>(prev);
        };
        auto bracket_hi = [&](int idx, int n, float lo, float step) -> float
        {
            const int next = (idx < n - 1) ? idx + 1 : idx;
            return lo + step * static_cast<float>(next);
        };

        const float az_brack_lo = bracket_lo(best_i, n_grid, az_lo_eff, az_step);
        const float az_brack_hi = bracket_hi(best_i, n_grid, az_lo_eff, az_step);
        const float el_brack_lo = bracket_lo(best_j, n_grid, el_lo_eff, el_step);
        const float el_brack_hi = bracket_hi(best_j, n_grid, el_lo_eff, el_step);

        const float az_best_grid = az_lo_eff + az_step * static_cast<float>(best_i);
        const float el_best_grid = el_lo_eff + el_step * static_cast<float>(best_j);

        // 1-D golden-section refinement: az first, then el, alternating
        // n_refine iterations on each.  Cheap and converges quadratically
        // near a smooth peak.
        float az_star = az_best_grid;
        float el_star = el_best_grid;

        if (az_brack_hi - az_brack_lo > 1.0e-6F)
        {
            float a = az_brack_lo;
            float b = az_brack_hi;
            float c = b - (b - a) * INV_PHI;
            float d = a + (b - a) * INV_PHI;
            float fc = reward_at(c, el_star);
            float fd = reward_at(d, el_star);
            for (int i = 0; i < n_refine; ++i)
            {
                if (fc > fd)
                {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - (b - a) * INV_PHI;
                    fc = reward_at(c, el_star);
                }
                else
                {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + (b - a) * INV_PHI;
                    fd = reward_at(d, el_star);
                }
            }
            az_star = (fc > fd) ? c : d;
        }

        if (el_brack_hi - el_brack_lo > 1.0e-6F)
        {
            float a = el_brack_lo;
            float b = el_brack_hi;
            float c = b - (b - a) * INV_PHI;
            float d = a + (b - a) * INV_PHI;
            float fc = reward_at(az_star, c);
            float fd = reward_at(az_star, d);
            for (int i = 0; i < n_refine; ++i)
            {
                if (fc > fd)
                {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - (b - a) * INV_PHI;
                    fc = reward_at(az_star, c);
                }
                else
                {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + (b - a) * INV_PHI;
                    fd = reward_at(az_star, d);
                }
            }
            el_star = (fc > fd) ? c : d;
        }

        const float o_star = reward_at(az_star, el_star);
        return {az_star, el_star, o_star};
    }
}  // namespace vamp::collision
