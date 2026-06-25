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

        if constexpr (std::is_same_v<DataT, float>)
        {
            for (const auto &tree : e.gaussian_trees)
            {
                risk = risk + tree.sum_overlap(robot);
            }
        }
        else if (not e.gaussian_trees.empty())
        {
            const auto mx = robot.mx.to_array();
            const auto my = robot.my.to_array();
            const auto mz = robot.mz.to_array();
            const auto sxx = robot.sigma_xx.to_array();
            const auto sxy = robot.sigma_xy.to_array();
            const auto sxz = robot.sigma_xz.to_array();
            const auto syy = robot.sigma_yy.to_array();
            const auto syz = robot.sigma_yz.to_array();
            const auto szz = robot.sigma_zz.to_array();
            const auto al = robot.alpha.to_array();
            for (const auto &tree : e.gaussian_trees)
            {
                std::array<float, DataT::num_scalars> lane_risk{};
                for (std::size_t l = 0; l < DataT::num_scalars; ++l)
                {
                    const collision::Gaussian3<float> robot_l{
                        mx[l],  my[l],  mz[l],  sxx[l], sxy[l],
                        sxz[l], syy[l], syz[l], szz[l], al[l]};
                    lane_risk[l] = tree.sum_overlap(robot_l);
                }
                risk = risk + DataT(lane_risk);
            }
        }
        return risk;
    }
}  // namespace vamp
