#pragma once

// Per-sphere collision-risk evaluator over a probabilistic ``Environment``.
// The probabilistic counterpart to
// ``validity.hh::sphere_environment_in_collision``: instead of a boolean
// overlap test, each robot-body sphere (an uncertain sphere — Gaussian centre
// with covariance Σ_r^s propagated by the cricket FK codegen, plus a physical
// radius r_s) is tested against the environment's uncertain obstacle spheres,
// and the routine returns the probability the body sphere collides with NONE of
// them: the noisy-OR "no-collision" product ∏_o (1 − p_o).
//
// Written ``DataT``-generic like the collision primitive: the robot body is an
// uncertain sphere carried in ``DataT`` arithmetic, so the same body serves a
// scalar evaluation (``DataT = float``) or a whole rake of configurations
// (``DataT = FloatVector<rake>``); the rake batching is the driver's job
// (``planning/validate_risk.hh``), which multiplies these per-sphere factors
// into one no-collision product and reports P = 1 − that.
//
// The obstacle population is ``env.gaussian_trees`` — one or more
// ``GaussianTree`` indices, each returning its no-collision product by a pruned
// range descent (``GaussianTree::no_collision_product``).  A blob obstacle and
// a static-map cloud point are the same thing — an uncertain sphere — so they
// share this one structure.  The descent is per-config (each lane's sphere
// centre takes a distinct tree path), so for a rake the lanes are kept packed
// and queried together.

#include <array>
#include <type_traits>

#include <vamp/collision/environment.hh>
#include <vamp/collision/gaussian.hh>

namespace vamp
{
    template <typename DataT>
    inline auto sphere_environment_no_collision(
        const collision::Environment<float> &e,
        const collision::Gaussian3<DataT> &robot,
        float min_no_collision = 0.0F) noexcept -> DataT
    {
        DataT no_collision = DataT(1.0F);

        // ``DataT == float`` selects the scalar ``no_collision_product``; a
        // packed ``DataT == FloatVector<rake>`` selects the rake-packed overload,
        // which descends each tree once for *all* lanes and accumulates per-lane.
        // The whole risk evaluation thus stays SIMD — no unpack to scalar, no
        // per-lane tree descent.  Independent obstacle trees compose by the same
        // noisy-OR product (collide with none of tree A AND none of tree B).
        //
        // ``min_no_collision`` is the noisy-OR reject early-out floor, forwarded
        // to each tree's descent so a single saturating tree short-circuits (see
        // ``GaussianTree::no_collision_product``).  ``0`` disables it.  It is a
        // total-P floor, so a stop is always safe: the running product is already
        // ``≤ floor`` before the caller multiplies any further factor in.
        for (const auto &tree : e.gaussian_trees)
        {
            no_collision =
                no_collision * tree.no_collision_product(robot, min_no_collision);
        }
        return no_collision;
    }
}  // namespace vamp
