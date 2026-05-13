#pragma once

// Visibility / observation-reward primitives for visibility-aware
// motion planning.
//
// Implements the closed-form factorisation from the paper
// (eq:obs_reward_approx, eq:angular_fraction, eq:radial_fraction):
//
//   O(q, φ)  ≈  Σ_k  f_k^a(q, φ) * f_k^r(q)
//
// where each k indexes a discretised waypoint of the future trajectory
// ξ_q convolved with an isotropic kernel of width σ_ρ, ``f_k^a`` is
// the standard-normal CDF mass within the camera field of view, and
// ``f_k^r`` is the CDF mass within the sensing range ``d_max``.
//
// Robot-agnostic: ``q`` is just a 2D base position + heading (the
// sensor-bearing surface of the robot), and the kernel set is opaque
// to vamp — the caller (a higher-level planner) builds it from
// whatever risk field it tracks.
//
// No SDF / raycasting in v1.  The paper's factorisation drops
// occlusion masking; visibility against ``Environment.pointclouds``
// (map-based occlusion) is a separate primitive that can be added
// later without changing this API.

#include <cmath>
#include <cstddef>

#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // One discretised waypoint of the future trajectory ξ_q, weighted
    // by the local unknown-region risk and convolved with an isotropic
    // kernel of width σ_ρ.  The caller supplies these; vamp integrates
    // them.
    struct RiskKernel
    {
        float x;
        float y;
        float z;
        float weight;
    };

    // Camera-cone observation reward at base configuration (qx, qy, θ)
    // with gaze direction ``phi`` (radians, world frame).  See
    // eq:obs_reward_approx.
    //
    //   q              base position (qx, qy) in world frame
    //   phi            sensor heading in world frame
    //   sigma_rho      kernel width (paper's σ_ρ)
    //   d_max          maximum sensing range
    //   psi            full camera field-of-view angle (radians)
    //   kernels        risk-weighted waypoints of the future trajectory
    //   n_kernels      length of the ``kernels`` buffer
    //
    // Returns Σ_k f_k^a · f_k^r.  All kernels are evaluated; no early
    // exit because each term is a small constant cost.
    inline auto observation_reward(
        float qx,
        float qy,
        float phi,
        float sigma_rho,
        float d_max,
        float psi,
        const RiskKernel *kernels,
        std::size_t n_kernels) noexcept -> float
    {
        const float half_psi = 0.5F * psi;
        float reward = 0.F;
        for (std::size_t k = 0; k < n_kernels; ++k)
        {
            const auto dx = kernels[k].x - qx;
            const auto dy = kernels[k].y - qy;
            const auto r = vamp::collision::sqrt(dx * dx + dy * dy);
            if (r < 1e-6F)
            {
                // Kernel sitting on top of the sensor — degenerate;
                // contributes the full kernel mass if within range.
                if (sigma_rho > 0.F)
                {
                    const auto f_r = vamp::collision::normal_cdf((d_max - r) / sigma_rho);
                    reward += kernels[k].weight * f_r;
                }
                continue;
            }

            // Angular offset from the gaze direction, wrapped to (-π, π].
            const auto bearing = std::atan2(dy, dx);
            auto delta = bearing - phi;
            while (delta > 3.14159265358979323846F)
            {
                delta -= 2.F * 3.14159265358979323846F;
            }
            while (delta <= -3.14159265358979323846F)
            {
                delta += 2.F * 3.14159265358979323846F;
            }

            // f_k^a = Φ((ψ/2 - δ)·r/σ_ρ) - Φ((-ψ/2 - δ)·r/σ_ρ)
            const auto inv_sigma = 1.F / sigma_rho;
            const auto f_a = vamp::collision::normal_cdf((half_psi - delta) * r * inv_sigma) -
                             vamp::collision::normal_cdf((-half_psi - delta) * r * inv_sigma);

            // f_k^r = Φ((d_max - r) / σ_ρ)
            const auto f_r = vamp::collision::normal_cdf((d_max - r) * inv_sigma);

            reward += kernels[k].weight * f_a * f_r;
        }
        return reward;
    }

    // Bounded 1-D golden-section search for the optimal gaze direction
    // φ* over the head-sweep range ``[theta - psi_h/2, theta + psi_h/2]``
    // (eq:gaze_node).  Returns ``(phi_star, O_star)``.
    //
    // The reward is non-convex in general (multiple kernels in
    // different directions) so golden-section can miss the global
    // optimum; for the head-sweep range typical in practice
    // (<= π radians) and σ_ρ that's a meaningful fraction of d_max,
    // it's empirically good enough.  Callers that need stronger
    // guarantees should call ``observation_reward`` over a fixed grid
    // of candidate ``phi`` values instead.
    //
    //   theta          robot body heading (centre of head-sweep range)
    //   psi_h          full head-sweep range (radians)
    //   n_iter         golden-section iterations (15–25 typical)
    inline auto optimal_gaze(
        float qx,
        float qy,
        float theta,
        float sigma_rho,
        float d_max,
        float psi,
        float psi_h,
        const RiskKernel *kernels,
        std::size_t n_kernels,
        int n_iter = 20) noexcept -> std::pair<float, float>
    {
        constexpr float INV_PHI = 0.6180339887498949F;  // 1 / golden ratio

        float a = theta - 0.5F * psi_h;
        float b = theta + 0.5F * psi_h;
        float c = b - (b - a) * INV_PHI;
        float d = a + (b - a) * INV_PHI;

        auto eval = [&](float p)
        { return observation_reward(qx, qy, p, sigma_rho, d_max, psi, kernels, n_kernels); };

        float fc = eval(c);
        float fd = eval(d);

        for (int i = 0; i < n_iter; ++i)
        {
            if (fc > fd)
            {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * INV_PHI;
                fc = eval(c);
            }
            else
            {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * INV_PHI;
                fd = eval(d);
            }
        }

        const float phi_star = (fc > fd) ? c : d;
        const float o_star = (fc > fd) ? fc : fd;
        return {phi_star, o_star};
    }
}  // namespace vamp::collision
