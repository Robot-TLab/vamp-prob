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
// The caller (typically the edge driver in
// ``planning/validate_risk.hh``) supplies the per-sphere covariance
// ``Σ_r^s`` already assembled — this matches the cricket-emitted
// ``sphere_fk_with_cov`` output, where ``Σ_r^s = J_s Σ_q J_s^T +
// r_s²·I`` is computed inside the FK codegen and arrives in
// ``SpheresWithCov<rake>`` alongside the centre.

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
    //   r_s            sphere radius (used only for the cull radius)
    //   sigma_rs       fully-assembled per-sphere covariance Σ_r^s =
    //                  J_s · Σ_q · J_s^T + r_s²·I.  Comes directly
    //                  from the cricket-emitted SpheresWithCov<rake>.
    //                  Pass ``sym3_iso(r_s * r_s)`` for the stopped-
    //                  base degenerate case (no uncertainty).
    template <typename DataT>
    inline auto sphere_environment_risk(
        const collision::Environment<DataT> &e,
        float cs_x,
        float cs_y,
        float cs_z,
        float r_s,
        const collision::Sym3 &sigma_rs) noexcept -> float
    {
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
