#pragma once

// Per-sphere collision-risk evaluator over a probabilistic
// ``Environment``.  This is the probabilistic counterpart to
// ``validity.hh::sphere_environment_in_collision``: same iteration
// pattern (sorted obstacles + early-exit cull on ``min_distance``)
// but the inner function evaluates a Gaussian kernel and sums into
// a scalar risk instead of testing a boolean overlap.
//
// Scope (v1):
//   - Iterates ``env.gaussian_obstacles`` only.
//   - Static-map Diracs from ``env.pointclouds`` are *not* iterated
//     yet: the CAPT stores points in a private SIMD-aligned layout
//     without a public iterator, and exposing one is a separate
//     concern from this PR.  Callers who need static-map risk should
//     either (a) encode static voxels as ``GaussianObstacle``s with
//     small body-only covariance, or (b) wait for the follow-up
//     that adds a const-iterator accessor to ``CAPT``.
//
// The edge-level driver ``validate_motion_risk<Robot, rake, resolution>``
// is **not** in this header.  It depends on the cricket-emitted
// ``sphere_fk_jacobian`` + ``SpheresJac<rake>`` struct, which lives
// in the robot codegen and is not yet shipped on the cricket_prob
// branch.  Once that lands, the driver wraps the per-sphere call
// below across K interpolated waypoints; see the prob extension plan
// for the staging.

#include <vamp/collision/environment.hh>
#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>
#include <vamp/collision/sphere_gaussian.hh>

namespace vamp
{
    // Per-sphere risk: sum of Gaussian-product contributions from every
    // tracked Gaussian obstacle whose 3σ extent overlaps a generous
    // cull window around the sphere centre.  Returns the value that
    // the union-of-spheres approximation (eq:waypoint_bound) sums
    // across spheres to obtain the per-waypoint risk.
    //
    //   c_s            sphere centre in world frame
    //   r_s            sphere radius (body kernel)
    //   sigma_b_part   J_s Σ_b J_s^T contribution from base uncertainty,
    //                  computed by the caller from the cricket-emitted
    //                  per-sphere base Jacobian and the planar-base
    //                  covariance.  Pass ``sym3_zero()`` for the
    //                  stopped-base degenerate case (deterministic).
    //
    // The final per-sphere covariance is assembled here as
    //   Σ_r^s = sigma_b_part + r_s² * I
    // matching eq:sphere_psv.
    template <typename DataT>
    inline auto sphere_environment_risk(
        const collision::Environment<DataT> &e,
        float cs_x,
        float cs_y,
        float cs_z,
        float r_s,
        const collision::Sym3 &sigma_b_part) noexcept -> float
    {
        const auto sigma_rs = collision::sym3_add(sigma_b_part, collision::sym3_iso(r_s * r_s));

        const auto centre_extent = collision::sqrt(cs_x * cs_x + cs_y * cs_y + cs_z * cs_z) + r_s;

        float risk = 0.F;
        for (const auto &g : e.gaussian_obstacles)
        {
            // Early-exit cull.  Obstacles are sorted by ``min_distance``
            // (the obstacle's mean-to-origin distance minus its 3σ
            // extent).  Once an obstacle's ``min_distance`` exceeds the
            // sphere's reach, every subsequent obstacle is even farther
            // and contributes negligibly.
            if (static_cast<float>(g.min_distance) > centre_extent + static_cast<float>(g.three_sigma_extent))
            {
                break;
            }
            risk += collision::sphere_risk_against_gaussian(cs_x, cs_y, cs_z, sigma_rs, g);
        }
        return risk;
    }
}  // namespace vamp
