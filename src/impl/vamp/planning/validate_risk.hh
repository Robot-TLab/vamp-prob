#pragma once

// Probabilistic counterpart to ``validate.hh::validate_motion``.
//
// Same waypoint-packing-into-rake-blocks pattern as the deterministic
// validator, but each lane evaluates a per-waypoint scalar risk
// instead of a boolean collision test.  The per-sphere covariance
// ``Σ_r^s = J_s · Σ_q · J_s^T + r_s²·I`` is produced inside the FK
// codegen by the cricket-emitted ``Robot::sphere_fk_with_cov<rake>``
// (cf. cricket's ``trace_sphere_fk_with_cov``).  vamp consumes
// per-sphere ``(centre, Σ_r^s)`` directly — no Jacobian materialised,
// no runtime matrix product.

#include <array>
#include <cmath>
#include <cstdint>

#include <vamp/collision/environment.hh>
#include <vamp/collision/gaussian.hh>
#include <vamp/collision/risk_validity.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    // Generate evenly-spaced percentages 1/rake, 2/rake, ..., 1.0 for
    // the rake waypoints along an edge.  Identical to the helper in
    // ``validate.hh`` (kept private there); duplicated here to avoid
    // pulling in the deterministic validator just for the constant.
    template <std::size_t n, std::size_t... I>
    inline constexpr auto risk_percents_impl(std::index_sequence<I...>) -> std::array<float, n>
    {
        return {(static_cast<void>(I), static_cast<float>(I + 1) / static_cast<float>(n))...};
    }

    template <std::size_t n>
    struct RiskPercents
    {
        inline static constexpr auto percents = risk_percents_impl<n>(std::make_index_sequence<n>());
    };

    // Result of a per-edge probabilistic validation.
    //
    //   feasible       — true iff every waypoint's risk is at or below
    //                    its allocated budget.
    //   per_waypoint_risk — one float per waypoint along the edge,
    //                    indexed in the order the rake produces them.
    //                    Always populated even when ``feasible == false``
    //                    so callers (e.g. the IRA loop) can re-allocate.
    struct EdgeRiskResult
    {
        bool feasible;
        std::vector<float> per_waypoint_risk;
    };

    // Broadcast a scalar ``Σ_q`` (6 floats, upper triangle) into the
    // rake-packed input that ``sphere_fk_with_cov`` expects.  Each
    // rake lane ends up carrying the same covariance, which is the
    // common case (the stopped-base check and uniform-Σ edges).  When
    // LQG-MP supplies a per-waypoint Σ_b(τ_k), the caller can build
    // its own per-lane block — this helper is just for the simple path.
    template <std::size_t rake, std::size_t n_sigma_q>
    inline constexpr auto broadcast_sigma_q(const collision::Sym3 &sigma_q) noexcept
        -> FloatVector<rake, n_sigma_q>
    {
        static_assert(n_sigma_q == 6, "broadcast_sigma_q currently supports n_unc = 3 only");
        FloatVector<rake, n_sigma_q> out;
        out[0] = FloatVector<rake, 1>::fill(sigma_q[0]);
        out[1] = FloatVector<rake, 1>::fill(sigma_q[1]);
        out[2] = FloatVector<rake, 1>::fill(sigma_q[2]);
        out[3] = FloatVector<rake, 1>::fill(sigma_q[3]);
        out[4] = FloatVector<rake, 1>::fill(sigma_q[4]);
        out[5] = FloatVector<rake, 1>::fill(sigma_q[5]);
        return out;
    }

    // Per-rake-block risk: evaluate the per-waypoint risk for every
    // lane in ``block`` and return a ``rake``-element vector.  Internal
    // helper used by both ``validate_motion_risk`` and
    // ``validate_config_risk``.
    template <typename Robot, std::size_t rake>
    inline auto evaluate_risk(
        const typename Robot::template ConfigurationBlock<rake> &block,
        const collision::Environment<float> &env,
        const FloatVector<rake, Robot::n_sigma_q> &sigma_q_block) noexcept -> std::array<float, rake>
    {
        typename Robot::template SpheresWithCov<rake> spheres;
        Robot::template sphere_fk_with_cov<rake>(block, sigma_q_block, spheres);

        // Keep the rake packed.  Each sphere row is a ``FloatVector<rake>``
        // (the ``rake`` configurations of that sphere), so the robot body
        // is a ``Gaussian3<FloatVector<rake>>`` and every obstacle is
        // evaluated against all rake lanes in one SIMD step — the same
        // robot-vs-environment SIMD the deterministic
        // ``sphere_environment_in_collision`` uses.  Σ_r^s already has
        // r_s²·I folded in by the FK codegen.
        using FV = FloatVector<rake>;
        FV risk_acc = FV::fill(0.0F);
        const FV one = FV::fill(1.0F);
        for (std::size_t s = 0; s < Robot::n_spheres; ++s)
        {
            const collision::Gaussian3<FV> robot{
                spheres.x[s],        spheres.y[s],        spheres.z[s],
                spheres.sigma_xx[s], spheres.sigma_xy[s], spheres.sigma_xz[s],
                spheres.sigma_yy[s], spheres.sigma_yz[s], spheres.sigma_zz[s],
                one};
            risk_acc = risk_acc + sphere_environment_risk<FV>(env, robot);
        }

        return risk_acc.to_array();
    }

    // Per-configuration risk: probabilistic counterpart to single-state
    // ``validate``.  Returns the scalar per-waypoint risk for ``q``.
    template <typename Robot, std::size_t rake>
    inline auto validate_config_risk(
        const typename Robot::Configuration &q,
        const collision::Environment<float> &env,
        const collision::Sym3 &sigma_q) noexcept -> float
    {
        // Build a rake block with every lane = q (cheap; only one lane
        // is actually distinct in the result, the others are wasted
        // work).  Future optimisation: a scalar `sphere_fk_with_cov<1>`
        // path.  For v1 we accept the rake-1-of-N cost.
        typename Robot::template ConfigurationBlock<rake> block;
        for (auto i = 0U; i < Robot::dimension; ++i)
        {
            block[i] = q.broadcast(i);
        }

        const auto sigma_q_block = broadcast_sigma_q<rake, Robot::n_sigma_q>(sigma_q);
        const auto risks = evaluate_risk<Robot, rake>(block, env, sigma_q_block);
        return risks[0];
    }

    // Per-edge risk along the linear interpolation from ``start`` to
    // ``goal``.  K = ``rake * n_substeps`` waypoints are evaluated; the
    // number of substeps is derived the same way ``validate_vector``
    // derives its FK substep count (``ceil(distance / rake * resolution)``).
    //
    // ``sigma_q``    scalar base covariance (broadcast to every lane).
    // ``eps_budget`` per-waypoint risk budget.  An empty span disables
    //                early-exit and forces a full evaluation.
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline auto validate_motion_risk(
        const typename Robot::Configuration &start,
        const typename Robot::Configuration &goal,
        const collision::Environment<float> &env,
        const collision::Sym3 &sigma_q,
        float eps_per_waypoint = std::numeric_limits<float>::infinity()) -> EdgeRiskResult
    {
        const auto vector = goal - start;
        const float distance = vector.l2_norm();

        const std::size_t n = std::max(
            std::ceil(distance / static_cast<float>(rake) * resolution), 1.F);

        // Contiguous waypoint layout: block ``i`` carries the ``rake`` *adjacent*
        // waypoints at fractions (i·rake + l + 1)/(rake·n).  Packing spatially
        // close waypoints into one rake lets the GaussianTree descend a whole
        // block in a single packet walk (``GaussianTree::sum_overlap``), keeping
        // the risk sum SIMD with no per-lane unpack.  The deterministic validator
        // spreads the rake across the edge so it can early-exit anywhere; the
        // risk sum never early-exits within an edge, so it packs tightly.  The
        // evaluated waypoint *set* {k/(rake·n) : k=1..rake·n} is identical to the
        // spread layout — only the lane grouping changes.
        std::array<float, rake> base_p{};
        for (std::size_t l = 0; l < rake; ++l)
        {
            base_p[l] = static_cast<float>(l + 1) / static_cast<float>(rake * n);
        }
        const auto base_percents = FloatVector<rake>(base_p);

        typename Robot::template ConfigurationBlock<rake> block;
        for (auto i = 0U; i < Robot::dimension; ++i)
        {
            block[i] = start.broadcast(i) + (vector.broadcast(i) * base_percents);
        }

        const auto sigma_q_block = broadcast_sigma_q<rake, Robot::n_sigma_q>(sigma_q);

        EdgeRiskResult result;
        result.feasible = true;
        result.per_waypoint_risk.reserve(rake * n);

        // First rake-block (the K end-points of the substep sweep).
        auto risks = evaluate_risk<Robot, rake>(block, env, sigma_q_block);
        for (auto lane = 0U; lane < rake; ++lane)
        {
            result.per_waypoint_risk.push_back(risks[lane]);
            if (risks[lane] > eps_per_waypoint)
            {
                result.feasible = false;
            }
        }
        if (!result.feasible or n == 1)
        {
            return result;
        }

        // Step forward one whole rake-block (rake waypoints) per iteration.
        const auto forward = vector / static_cast<float>(n);
        for (auto i = 1U; i < n; ++i)
        {
            for (auto j = 0U; j < Robot::dimension; ++j)
            {
                block[j] = block[j] + forward.broadcast(j);
            }
            risks = evaluate_risk<Robot, rake>(block, env, sigma_q_block);
            for (auto lane = 0U; lane < rake; ++lane)
            {
                result.per_waypoint_risk.push_back(risks[lane]);
                if (risks[lane] > eps_per_waypoint)
                {
                    result.feasible = false;
                    return result;
                }
            }
        }

        return result;
    }
}  // namespace vamp::planning
