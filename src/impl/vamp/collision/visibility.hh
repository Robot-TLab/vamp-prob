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
// the same primary type the collision-risk path uses.  λ_s / λ_d priors
// stay outside vamp — callers fold them into each kernel's ``alpha``
// weight upstream.
//
// No SDF / raycasting in v1.  The paper's factorisation drops
// occlusion masking; visibility against ``Environment.pointclouds``
// (map-based occlusion) is a separate primitive that can be added
// later without changing this API.
//
// SIMD: ``observation_reward`` packs the Gaussian kernels into SIMD lanes
// and evaluates the per-kernel reward a whole vector at a time, exactly
// mirroring the collision-risk path — ``GaussianTree::descend`` broadcasts
// one query and runs ``gaussian_gaussian`` against the *stored* Gaussians
// packed ``num_scalars`` per block, then horizontally sums.  Here the camera
// pose ``(c, n_gaze)`` is the broadcast query and the kernels are the packed
// set: ‖u × n‖, ⟨u, n⟩, the angle, the radial Φ and the angular χ²₂ tail all
// run in lane arithmetic (``cross_3`` / ``atan2`` / ``normal_cdf`` /
// ``ncx2_2_cdf`` from ``collision/math.hh``), and one ``hsum`` collapses the
// lanes to the scalar reward.  ``optimal_gaze`` inherits the speedup through
// its per-candidate calls; a planner is still free to rake over camera poses
// one level up (each lane a scalar ``observation_reward``).

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

namespace vamp::collision
{
    // Full-fp32 square root: one Newton step off the fast 12-bit ``v·rsqrt(v)``
    // seed, refined with the (Newton-refined) ``rcp``.  The radial Φ and angular
    // χ²₂ tail are near-step functions for tight kernels, so the bare 12-bit
    // sqrt's 3.7e-4 relative error in r / σ / ‖u×n‖ would smear the reward by
    // ~0.04 per kernel near a boundary (and drift from the exact scalar path).
    // Argument must be > 0 (callers floor the radicand).
    inline auto precise_sqrt(const FloatVector<> &x) noexcept -> FloatVector<>
    {
        const auto s = x.sqrt();
        return (s + x * vamp::collision::rcp(s)) * 0.5F;
    }

    // Per-kernel angular fraction f^a = P[χ²₂(λ) ≤ z] for a packed lane of
    // kernels, given their unit directions ``u`` to the camera, the per-kernel
    // ``scale = r/σ`` and ``z = (ψ/2·scale)²`` (both gaze-independent), and a
    // broadcast gaze direction ``n_gaze``.  λ = (δ·scale)² with
    // δ = ∠(u, n_gaze) = atan2(‖u×n‖, ⟨u,n⟩) (precise near 0 and π).  This is
    // the *only* part of the reward that changes as the gaze sweeps, so it is
    // factored out and shared by ``observation_reward`` and ``optimal_gaze``.
    inline auto angular_fraction(
        const FloatVector<> &ux,
        const FloatVector<> &uy,
        const FloatVector<> &uz,
        const FloatVector<> &scale,
        const FloatVector<> &z,
        const FloatVector<> &nxv,
        const FloatVector<> &nyv,
        const FloatVector<> &nzv) noexcept -> FloatVector<>
    {
        using FV = FloatVector<>;
        FV cxn;
        FV cyn;
        FV czn;
        vamp::collision::cross_3(ux, uy, uz, nxv, nyv, nzv, cxn, cyn, czn);
        // ‖u×n‖² floored before the sqrt: u ∥ n_gaze (on-axis / antipodal) makes
        // it 0, and the rsqrt-based sqrt would turn sqrt(0) into 0·∞ = NaN; the
        // floor (≈1e-10 in ‖u×n‖) lands atan2 at δ ≈ 0 or π, as intended.
        const FV cross_norm = precise_sqrt((cxn * cxn + cyn * cyn + czn * czn).max(FV::fill(1.0e-20F)));
        const FV dot_un = ux * nxv + uy * nyv + uz * nzv;
        const FV delta = vamp::collision::atan2(cross_norm, dot_un);
        const FV lam = (delta * scale) * (delta * scale);
        return vamp::collision::ncx2_2_cdf(z, lam);
    }

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
        using FV = FloatVector<>;
        constexpr std::size_t W = FV::num_scalars;
        constexpr float SIGMA_EPS = 1.0e-9F;  // σ floor: division-by-zero guard
                                              // on caller-supplied delta kernels
        constexpr float R_EPS = 1.0e-6F;      // degenerate kernel-on-camera radius

        // The camera pose is the broadcast "query"; the kernels are the packed
        // set (cf. ``GaussianTree::descend``: one query, stored Gaussians per lane).
        const FV nxv = FV::fill(nx);
        const FV nyv = FV::fill(ny);
        const FV nzv = FV::fill(nz);
        const FV half_psi = FV::fill(0.5F * psi);

        FV acc = FV::fill(0.0F);
        for (std::size_t base = 0; base < n_gaussians; base += W)
        {
            // Struct-of-arrays pack of up to W kernels.  Only the diagonal of Σ
            // feeds the isotropic σ = √(tr(Σ)/3); the visibility math needs no
            // off-diagonal terms.  Padding lanes carry a finite dummy pose with
            // α = 0 so they contribute exactly 0 — no +inf mean (which would
            // make u = d·(1/r) a 0·∞ NaN, unlike the Gaussian-product path).
            alignas(FV::S::Alignment) std::array<float, W> amx;
            alignas(FV::S::Alignment) std::array<float, W> amy;
            alignas(FV::S::Alignment) std::array<float, W> amz;
            alignas(FV::S::Alignment) std::array<float, W> atrace;
            alignas(FV::S::Alignment) std::array<float, W> aalpha;
            for (std::size_t l = 0; l < W; ++l)
            {
                const std::size_t k = base + l;
                if (k < n_gaussians)
                {
                    const auto &g = gaussians[k];
                    amx[l] = g.mx;
                    amy[l] = g.my;
                    amz[l] = g.mz;
                    atrace[l] = g.sigma_xx + g.sigma_yy + g.sigma_zz;
                    aalpha[l] = g.alpha;
                }
                else
                {
                    amx[l] = cx + 1.0F;
                    amy[l] = cy;
                    amz[l] = cz;
                    atrace[l] = 3.0F;
                    aalpha[l] = 0.0F;
                }
            }

            const FV dx = FV(amx.data(), true) - cx;
            const FV dy = FV(amy.data(), true) - cy;
            const FV dz = FV(amz.data(), true) - cz;
            const FV alpha = FV(aalpha.data(), true);

            // r² is exact (never NaN), so it gives both the degenerate test and,
            // floored before the sqrt, a finite r — the AVX sqrt is v·rsqrt(v),
            // so an unfloored sqrt(0) would be 0·∞ = NaN and break the blend.
            const FV r2 = dx * dx + dy * dy + dz * dz;
            const FV degen = r2.less_than(FV::fill(R_EPS * R_EPS));      // kernel on camera
            const FV r = precise_sqrt(r2.max(FV::fill(R_EPS * R_EPS)));  // finite, ≥ R_EPS
            const FV sigma =
                precise_sqrt((FV(atrace.data(), true) * (1.0F / 3.0F)).max(FV::fill(SIGMA_EPS * SIGMA_EPS)));
            const FV inv_sigma = vamp::collision::rcp(sigma);
            const FV inv_r = vamp::collision::rcp(r);  // r ≥ R_EPS

            const FV scale = r * inv_sigma;  // r / σ
            const FV z = (half_psi * scale) * (half_psi * scale);
            const FV f_a = angular_fraction(dx * inv_r, dy * inv_r, dz * inv_r, scale, z, nxv, nyv, nzv);
            const FV f_r = vamp::collision::normal_cdf((d_max - r) * inv_sigma);
            const FV contrib = alpha * f_a * f_r;

            // Degenerate kernel sitting on the camera (r ≈ 0): the angular noise
            // σ/r → ∞ makes it "fully visible if in range", so the angular
            // fraction collapses to 1 and only the radial Φ(d_max/σ) gates it.
            const FV contrib_degen = alpha * vamp::collision::normal_cdf(FV::fill(d_max) * inv_sigma);
            acc = acc + contrib.blend(contrib_degen, degen);
        }

        return acc.hsum();
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

        // Precompute the gaze-independent per-kernel terms ONCE.  As the gaze
        // sweeps the search window only δ = ∠(u, n_gaze) changes; the camera
        // position, kernel geometry (r, σ), radial fraction f^r and z all stay
        // fixed.  Caching them as packed SoA lanes means each of the n_grid² +
        // refinement evaluations below does only the angle-dependent work
        // (``angular_fraction``) instead of re-packing the Gaussian array and
        // recomputing every sqrt / Φ — the bulk of the per-evaluation cost.
        // Kernels on the camera (r ≈ 0) are fully visible if in range,
        // gaze-independently, so they fold into the constant ``base``.
        using FV = FloatVector<>;
        constexpr std::size_t W = FV::num_scalars;
        constexpr float SIGMA_EPS = 1.0e-9F;
        constexpr float R_EPS = 1.0e-6F;
        constexpr float PI = 3.141592653589793F;
        const FV half_psi = FV::fill(0.5F * psi);

        // Cull kernels that cannot contribute to ANY reachable gaze, once, up
        // front — the dominant cost is the per-candidate sweep below, so every
        // kernel dropped here is dropped from all n_grid² + refinement
        // evaluations.  A kernel is irrelevant if it is out of range for every
        // gaze (r > d_max + 6σ ⇒ radial Φ < 1e-9) or its direction lies outside
        // the union of all reachable cones: the central gaze n_center = R_head·
        // n_ref, widened by the FoV half-angle ψ/2, the head-sweep half-width Θ
        // (a generous sum-of-axes bound), and a 6σ-angular margin folded into a
        // fixed 0.5 rad slack.  Margins are deliberately loose so the cull never
        // drops a kernel that carries non-negligible reward.
        float ncx;
        float ncy;
        float ncz;
        compose_gaze(
            r00, r01, r02, r10, r11, r12, r20, r21, r22, nx_ref, ny_ref, nz_ref, 0.0F, 0.0F, ncx, ncy, ncz);
        const float theta = std::max(std::abs(az_lo_eff), std::abs(az_hi_eff)) +
                            std::max(std::abs(el_lo_eff), std::abs(el_hi_eff));
        const float ang_thresh = 0.5F * psi + theta + 0.5F;
        // cos_thresh = −2 ⇒ angular test always passes (whole sphere reachable).
        const float cos_thresh = (ang_thresh < PI) ? std::cos(ang_thresh) : -2.0F;

        std::vector<std::size_t> survivors;
        survivors.reserve(n_gaussians);
        for (std::size_t k = 0; k < n_gaussians; ++k)
        {
            const auto &g = gaussians[k];
            const float dx = g.mx - cx;
            const float dy = g.my - cy;
            const float dz = g.mz - cz;
            const float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < R_EPS * R_EPS)
            {
                survivors.push_back(k);  // degenerate kernel on camera — always relevant
                continue;
            }
            const float sigma =
                std::sqrt(std::max((g.sigma_xx + g.sigma_yy + g.sigma_zz) / 3.0F, SIGMA_EPS * SIGMA_EPS));
            const float r = std::sqrt(r2);
            if (r > d_max + 6.0F * sigma)
            {
                continue;  // radial cull
            }
            // u·n_center ≥ cos_thresh  ⟺  (d·n_center) ≥ r·cos_thresh.
            const float dot = dx * ncx + dy * ncy + dz * ncz;
            if (dot < r * cos_thresh)
            {
                continue;  // angular cull
            }
            survivors.push_back(k);
        }

        // The gaze sweep below must hit microsecond latency over many kernels,
        // which rules out the smooth ncx2/erf reward per candidate (≈3
        // transcendentals/kernel).  Each surviving kernel is instead reduced to
        // a piecewise-linear cone ramp: f^a ≈ 1 for δ < ψ/2 − 2σ/r, 0 for
        // δ > ψ/2 + 2σ/r, linear in cos δ between (the ncx2 tail's boundary and
        // ≈2σ/r transition width).  The thresholds (cos of the two angles) and
        // the radial weight α·f^r are precomputed ONCE here, so each gaze
        // evaluation is just a dot product, a clamp and a multiply-add — no
        // transcendentals.  Approximation matches the smooth model at the
        // boundary (both = ½) and in the in/out limits; it is the deliberate
        // accuracy-for-speed trade for the search (the standalone
        // ``observation_reward`` keeps the exact model).
        const std::size_t n_surv = survivors.size();
        const std::size_t n_blocks = (n_surv + W - 1) / W;
        std::vector<FV> k_ux(n_blocks);
        std::vector<FV> k_uy(n_blocks);
        std::vector<FV> k_uz(n_blocks);
        std::vector<FV> k_coshi(n_blocks);     // cos(ψ/2 + 2σ/r): outer ramp edge
        std::vector<FV> k_invrange(n_blocks);  // 1 / (cos(inner) − cos(outer))
        std::vector<FV> k_scale(n_blocks);     // r/σ — for the exact O* at the chosen gaze
        std::vector<FV> k_z(n_blocks);         // (ψ/2·scale)² — for the exact O*
        std::vector<FV> k_w(n_blocks);         // α·f^r; 0 on degenerate / padding lanes
        float base = 0.0F;
        for (std::size_t b = 0; b < n_blocks; ++b)
        {
            alignas(FV::S::Alignment) std::array<float, W> amx;
            alignas(FV::S::Alignment) std::array<float, W> amy;
            alignas(FV::S::Alignment) std::array<float, W> amz;
            alignas(FV::S::Alignment) std::array<float, W> atrace;
            alignas(FV::S::Alignment) std::array<float, W> aalpha;
            for (std::size_t l = 0; l < W; ++l)
            {
                const std::size_t si = b * W + l;
                if (si < n_surv)
                {
                    const auto &g = gaussians[survivors[si]];
                    amx[l] = g.mx;
                    amy[l] = g.my;
                    amz[l] = g.mz;
                    atrace[l] = g.sigma_xx + g.sigma_yy + g.sigma_zz;
                    aalpha[l] = g.alpha;
                }
                else
                {
                    amx[l] = cx + 1.0F;
                    amy[l] = cy;
                    amz[l] = cz;
                    atrace[l] = 3.0F;
                    aalpha[l] = 0.0F;
                }
            }

            const FV dx = FV(amx.data(), true) - cx;
            const FV dy = FV(amy.data(), true) - cy;
            const FV dz = FV(amz.data(), true) - cz;
            const FV alpha = FV(aalpha.data(), true);

            const FV r2 = dx * dx + dy * dy + dz * dz;
            const FV degen = r2.less_than(FV::fill(R_EPS * R_EPS));
            const FV r = precise_sqrt(r2.max(FV::fill(R_EPS * R_EPS)));
            const FV sigma =
                precise_sqrt((FV(atrace.data(), true) * (1.0F / 3.0F)).max(FV::fill(SIGMA_EPS * SIGMA_EPS)));
            const FV inv_sigma = vamp::collision::rcp(sigma);
            const FV inv_r = vamp::collision::rcp(r);
            const FV scale = r * inv_sigma;
            const FV f_r = vamp::collision::normal_cdf((d_max - r) * inv_sigma);

            // Cone-ramp edges in cos δ.  2σ/r is the angular transition width.
            const FV delta_w = (sigma * inv_r) * 2.0F;
            const FV cos_hi = (half_psi + delta_w).clamp(0.0F, PI).cos();  // outer (δ = ψ/2 + 2σ/r)
            const FV cos_lo = (half_psi - delta_w).clamp(0.0F, PI).cos();  // inner (δ = ψ/2 − 2σ/r)

            // Degenerate (r ≈ 0) lanes: α·Φ(d_max/σ) for any gaze → constant base.
            base += FV::fill(0.0F)
                        .blend(alpha * vamp::collision::normal_cdf(FV::fill(d_max) * inv_sigma), degen)
                        .hsum();

            k_ux[b] = dx * inv_r;
            k_uy[b] = dy * inv_r;
            k_uz[b] = dz * inv_r;
            k_coshi[b] = cos_hi;
            k_invrange[b] = vamp::collision::rcp((cos_lo - cos_hi).max(FV::fill(1.0e-6F)));
            k_scale[b] = scale;
            k_z[b] = (half_psi * scale) * (half_psi * scale);
            k_w[b] = (alpha * f_r).blend(FV::fill(0.0F), degen);  // 0 on degenerate / padding
        }

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
            const FV nxv = FV::fill(nx_g);
            const FV nyv = FV::fill(ny_g);
            const FV nzv = FV::fill(nz_g);
            // ε·(u·n) alignment tiebreak: the clamp zeroes the ramp outside the
            // cone, which would drop the smooth reward's monotone-in-cos δ
            // gradient — the bit that still aims the gaze at the nearest kernel
            // when it sits in the flat-top cap (any in-cone direction scores 1)
            // or entirely outside every reachable cone.  Re-add it at a weight
            // small enough never to outrank an actual in-cone kernel.
            constexpr float EPS_ALIGN = 1.0e-2F;
            FV acc = FV::fill(0.0F);
            for (std::size_t b = 0; b < n_blocks; ++b)
            {
                const FV dot = k_ux[b] * nxv + k_uy[b] * nyv + k_uz[b] * nzv;  // cos δ
                const FV s = ((dot - k_coshi[b]) * k_invrange[b]).clamp(0.0F, 1.0F);
                acc = acc + k_w[b] * (s + dot * EPS_ALIGN);
            }
            return base + acc.hsum();
        };

        // Exact reward at one gaze (smooth ncx2 angular fraction).  Used only to
        // report O* at the chosen gaze: the cheap ramp + ε·alignment above steers
        // the search, but the returned O* feeds a hard visibility constraint
        // (O* ≥ O_min), so it must match the smooth ``observation_reward`` model
        // rather than carry the ramp's approximation or the ε tiebreak offset.
        auto accurate_reward_at = [&](float az, float el) -> float
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
            const FV nxv = FV::fill(nx_g);
            const FV nyv = FV::fill(ny_g);
            const FV nzv = FV::fill(nz_g);
            FV acc = FV::fill(0.0F);
            for (std::size_t b = 0; b < n_blocks; ++b)
            {
                acc = acc +
                      k_w[b] * angular_fraction(k_ux[b], k_uy[b], k_uz[b], k_scale[b], k_z[b], nxv, nyv, nzv);
            }
            return base + acc.hsum();
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
        auto bracket_lo = [&](int idx, int /*n*/, float lo, float step) -> float
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

        const float o_star = accurate_reward_at(az_star, el_star);
        return {az_star, el_star, o_star};
    }
}  // namespace vamp::collision
