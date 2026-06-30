#pragma once

// Per-sphere collision-risk evaluator over a probabilistic ``Environment``.
// The probabilistic counterpart to
// ``validity.hh::sphere_environment_in_collision``: instead of a boolean
// overlap test, each robot-body sphere (a Gaussian ``N(c_s, Σ_r^s)``, Σ
// propagated by the cricket FK codegen) contributes the summed Gaussian-product
// overlap against the environment's Gaussian obstacle population.
//
// Written ``DataT``-generic like the collision primitive: the robot body is a
// Gaussian carried in ``DataT`` arithmetic, so the same body serves a scalar
// evaluation (``DataT = float``) or a whole rake of configurations
// (``DataT = FloatVector<rake>``); the rake batching is the driver's job
// (``planning/validate_risk.hh``).
//
// The obstacle population is ``env.gaussian_trees`` — one or more
// ``GaussianTree`` indices, each summed by a complete range descent
// (``GaussianTree::sum_overlap``).  A blob obstacle and a static-map cloud point
// are the same thing — a 3-D Gaussian — so they share this one structure.  The
// descent is per-config (each lane's sphere centre takes a distinct tree path),
// so for a rake the lanes are unpacked, queried one at a time, and repacked.

#include <array>
#include <type_traits>

#include <vamp/collision/environment.hh>
#include <vamp/collision/gaussian.hh>

namespace vamp
{
    template <typename DataT>
    inline auto sphere_environment_risk(
        const collision::Environment<float> &e,
        const collision::Gaussian3<DataT> &robot) noexcept -> DataT
    {
        DataT risk = DataT(0.0F);

        // ``DataT == float`` selects the scalar ``sum_overlap``; a packed
        // ``DataT == FloatVector<rake>`` selects the rake-packed overload, which
        // descends each tree once for *all* lanes and accumulates per-lane.  The
        // whole risk evaluation thus stays SIMD — no unpack to scalar, no
        // per-lane tree descent (cf. the old per-lane loop this replaces).
        for (const auto &tree : e.gaussian_trees)
        {
            risk = risk + tree.sum_overlap(robot);
        }
        return risk;
    }
}  // namespace vamp
