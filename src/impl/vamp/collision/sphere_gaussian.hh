#pragma once

// Per-term Gaussian risk primitives — the atomic building blocks of
// the probabilistic collision-checking pipeline.
//
// Each call evaluates one term of the per-sphere risk sum from the
// paper (eq:risk_static / eq:risk_dynamic):
//
//     P_s^{type}(τ_k) = sum_j  weight_j *
//                       N( c_s(τ_k);  μ_j,  Σ_r^s(τ_k) + Σ_j + Σ_body )
//
// Inputs:
//   c_s ∈ ℝ³                          sphere centre in the world frame
//   Σ_r^s ∈ Sym3                       J_s Σ_b J_s^T + r_s² I, already
//                                      assembled by the caller from the
//                                      cricket-generated SpheresJac and
//                                      the planar-base covariance
//   GaussianObstacle / Point          obstacle data from Environment
//
// The functions return the (signed) per-term risk contribution as a
// plain ``float``.  They do **not** sum across multiple obstacles —
// that is the job of ``sphere_environment_risk`` in
// ``collision/risk_validity.hh``.

#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>
#include <vamp/collision/shapes.hh>

namespace vamp::collision
{
    // Risk of a per-sphere occupancy ``N(c_s, Σ_r^s)`` evaluated against
    // a single point obstacle ``x_j`` with body kernel σ²_body.  The
    // point is treated as a Dirac mass convolved with the body kernel:
    //
    //   contribution = N( x_j; c_s, Σ_r^s + σ²_body * I )
    //
    // ``weight`` lets the caller scale by a per-point occupancy density
    // (e.g. the static map's expected occupancy per voxel).  Returns
    // the un-summed kernel value.
    inline auto sphere_risk_against_point(
        float cs_x,
        float cs_y,
        float cs_z,
        const Sym3 &sigma_rs,
        float xj,
        float yj,
        float zj,
        float body_sigma_sq,
        float weight = 1.F) noexcept -> float
    {
        const auto sigma_total = sym3_add(sigma_rs, sym3_iso(body_sigma_sq));
        return weight * gaussian3_density(cs_x - xj, cs_y - yj, cs_z - zj, sigma_total);
    }

    // Risk of a per-sphere occupancy ``N(c_s, Σ_r^s)`` evaluated against
    // a tracked Gaussian obstacle (mean μ, covariance Σ_obs, weight α):
    //
    //   contribution = α * N( μ; c_s, Σ_r^s + Σ_obs )
    //
    // The Gaussian-product identity makes this symmetric in the two
    // distributions; we evaluate at the offset ``c_s - μ`` against the
    // summed covariance.  Returns the un-summed kernel value.
    template <typename DataT>
    inline auto sphere_risk_against_gaussian(
        float cs_x,
        float cs_y,
        float cs_z,
        const Sym3 &sigma_rs,
        const GaussianObstacle<DataT> &g) noexcept -> float
    {
        const auto sigma_obs = Sym3{
            static_cast<float>(g.sigma_xx),
            static_cast<float>(g.sigma_xy),
            static_cast<float>(g.sigma_xz),
            static_cast<float>(g.sigma_yy),
            static_cast<float>(g.sigma_yz),
            static_cast<float>(g.sigma_zz),
        };
        const auto sigma_total = sym3_add(sigma_rs, sigma_obs);
        return static_cast<float>(g.alpha) *
               gaussian3_density(
                   cs_x - static_cast<float>(g.mx),
                   cs_y - static_cast<float>(g.my),
                   cs_z - static_cast<float>(g.mz),
                   sigma_total);
    }
}  // namespace vamp::collision
