#pragma once

// ``gaussian_gaussian`` — the atomic probabilistic-collision primitive: the
// Gaussian-product overlap of two 3-D Gaussians.  It is the probability-mass
// analogue of ``sphere_sphere_sql2`` (``collision/sphere_sphere.hh``): a
// robot-body sphere is itself a Gaussian (mean c_s, covariance
// Σ_r^s = J Σ_q Jᵀ + r_s²·I — the FK-propagated position uncertainty with the
// radius folded in), an obstacle is a Gaussian, and the per-pair risk is their
// overlap.  The driver (``collision/risk_validity.hh``, fkcc-side) sums it over
// obstacles; the SIMD/rake batching lives there, exactly as for the sphere
// primitives.

#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

namespace vamp::collision
{
    // Gaussian-product overlap of two 3-D Gaussians:
    //
    //   ⟨a, b⟩ = α_a · α_b · N(μ_a − μ_b; 0, Σ_a + Σ_b)
    //
    // Symmetric in the two Gaussians.  Written in ``DataT`` arithmetic so the
    // *same* routine serves a scalar pair (``DataT = float``) or a whole rake of
    // lanes (``DataT = FloatVector``); the batching — FK, obstacle iteration —
    // lives in the caller, exactly as for the sphere primitives.  Σ_a + Σ_b ≻ 0
    // for any risk input (each carries an r²·I / body-kernel term), so no
    // determinant guard is needed.
    template <typename DataT>
    inline auto gaussian_gaussian(const Gaussian3<DataT> &a, const Gaussian3<DataT> &b) noexcept
        -> DataT
    {
        // Σ_total = Σ_a + Σ_b.
        const DataT txx = a.sigma_xx + b.sigma_xx;
        const DataT txy = a.sigma_xy + b.sigma_xy;
        const DataT txz = a.sigma_xz + b.sigma_xz;
        const DataT tyy = a.sigma_yy + b.sigma_yy;
        const DataT tyz = a.sigma_yz + b.sigma_yz;
        const DataT tzz = a.sigma_zz + b.sigma_zz;

        // μ_diff = μ_a − μ_b.
        const DataT dx = a.mx - b.mx;
        const DataT dy = a.my - b.my;
        const DataT dz = a.mz - b.mz;

        // Symmetric 3×3 cofactors (matches the scalar ``sym3_solve``).
        const DataT c00 = tyy * tzz - tyz * tyz;
        const DataT c01 = txz * tyz - txy * tzz;
        const DataT c02 = txy * tyz - txz * tyy;
        const DataT c11 = txx * tzz - txz * txz;
        const DataT c12 = txz * txy - txx * tyz;
        const DataT c22 = txx * tyy - txy * txy;

        const DataT det = txx * c00 + txy * c01 + txz * c02;
        const DataT inv_det = rcp(det);

        const DataT solx = (c00 * dx + c01 * dy + c02 * dz) * inv_det;
        const DataT soly = (c01 * dx + c11 * dy + c12 * dz) * inv_det;
        const DataT solz = (c02 * dx + c12 * dy + c22 * dz) * inv_det;
        const DataT quad = dx * solx + dy * soly + dz * solz;

        constexpr float TWO_PI_POW_3_2 = 15.749609945722419F;  // (2π)^{3/2}
        const DataT norm = rcp(sqrt(det)) * (1.0F / TWO_PI_POW_3_2);
        return a.alpha * b.alpha * (norm * exp(quad * -0.5F));
    }
}  // namespace vamp::collision
