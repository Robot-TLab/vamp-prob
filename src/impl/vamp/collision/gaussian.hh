#pragma once

// Closed-form 3-D Gaussian product evaluator used by the probabilistic
// collision-checking and visibility primitives.
//
// The paper this targets (mobile grasping under uncertainty) reduces
// every per-sphere risk integral to a sum of expressions of the form
//
//     N(mu_diff; 0, Sigma_total)
//   = (2*pi)^{-3/2} * det(Sigma_total)^{-1/2} *
//     exp(-1/2 * mu_diff^T * Sigma_total^{-1} * mu_diff)
//
// where ``mu_diff = mu1 - mu2`` and ``Sigma_total = Sigma1 + Sigma2``
// (the Gaussian-product identity).  Because all the inputs are
// 3-D positions and 3x3 symmetric covariance matrices, every linear
// algebra step has a closed form and never needs an iterative solver.
//
// Covariance layout: ``Sym3 = {xx, xy, xz, yy, yz, zz}`` (upper
// triangle, row-major).  Each ``Sym3`` is a 6-float ``std::array``.

#include <array>
#include <cmath>

#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // Symmetric 3x3 matrix, upper-triangle row-major:
    //   [xx xy xz]
    //   [xy yy yz]
    //   [xz yz zz]
    using Sym3 = std::array<float, 6>;

    inline constexpr auto sym3_zero() noexcept -> Sym3
    {
        return Sym3{0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
    }

    inline constexpr auto sym3_iso(float sigma_sq) noexcept -> Sym3
    {
        return Sym3{sigma_sq, 0.F, 0.F, sigma_sq, 0.F, sigma_sq};
    }

    inline constexpr auto sym3_add(const Sym3 &a, const Sym3 &b) noexcept -> Sym3
    {
        return Sym3{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5]};
    }

    // The 3-D Gaussian distribution: mean + symmetric covariance
    // (6 upper-triangle entries) + scalar weight ``alpha``.  This is the
    // single Gaussian type used by every vamp primitive that integrates a
    // 3-D Gaussian — sphere-vs-Gaussian collision risk
    // (``collision/gaussian_gaussian.hh``), the indexed risk population
    // (``collision/gaussian_tree.hh``), and visibility / observation reward
    // (``collision/visibility.hh``) all consume it directly.
    template <typename DataT>
    struct Gaussian3
    {
        DataT mx;
        DataT my;
        DataT mz;

        DataT sigma_xx;
        DataT sigma_xy;
        DataT sigma_xz;
        DataT sigma_yy;
        DataT sigma_yz;
        DataT sigma_zz;

        DataT alpha;

        Gaussian3()
          : mx()
          , my()
          , mz()
          , sigma_xx()
          , sigma_xy()
          , sigma_xz()
          , sigma_yy()
          , sigma_yz()
          , sigma_zz()
          , alpha(static_cast<DataT>(1))
        {
        }

        Gaussian3(
            DataT mx,
            DataT my,
            DataT mz,
            DataT sigma_xx,
            DataT sigma_xy,
            DataT sigma_xz,
            DataT sigma_yy,
            DataT sigma_yz,
            DataT sigma_zz,
            DataT alpha = static_cast<DataT>(1))
          : mx(mx)
          , my(my)
          , mz(mz)
          , sigma_xx(sigma_xx)
          , sigma_xy(sigma_xy)
          , sigma_xz(sigma_xz)
          , sigma_yy(sigma_yy)
          , sigma_yz(sigma_yz)
          , sigma_zz(sigma_zz)
          , alpha(alpha)
        {
        }

        template <typename OtherDataT>
        explicit Gaussian3(const Gaussian3<OtherDataT> &other)
          : mx(other.mx)
          , my(other.my)
          , mz(other.mz)
          , sigma_xx(other.sigma_xx)
          , sigma_xy(other.sigma_xy)
          , sigma_xz(other.sigma_xz)
          , sigma_yy(other.sigma_yy)
          , sigma_yz(other.sigma_yz)
          , sigma_zz(other.sigma_zz)
          , alpha(other.alpha)
        {
        }

        // Trace of Σ.  Used both as the operator-norm bound for 3σ
        // extent computation and as the input to ``iso_sigma``.
        inline constexpr auto trace_sigma() const noexcept -> DataT
        {
            return sigma_xx + sigma_yy + sigma_zz;
        }

        // Isotropic-projection scalar σ = √(tr(Σ)/3).  v1 visibility
        // uses this as the 1-D bandwidth in the ncx2 angular fraction;
        // the rigorous anisotropic projection onto the plane orthogonal
        // to the gaze direction is a future extension that will replace
        // this call site without changing the visibility surface.
        inline auto iso_sigma() const noexcept -> DataT
        {
            return vamp::collision::sqrt(trace_sigma() / static_cast<DataT>(3));
        }

        // Pack the symmetric covariance into a ``Sym3``.  Convenience
        // for callers that hand it on to ``sym3_*`` / ``gaussian3_density``.
        inline constexpr auto sigma_sym3() const noexcept -> Sym3
        {
            return Sym3{sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz};
        }
    };

    // Determinant of a 3x3 symmetric matrix in ``Sym3`` layout.
    inline constexpr auto sym3_det(const Sym3 &m) noexcept -> float
    {
        const auto xx = m[0], xy = m[1], xz = m[2];
        const auto yy = m[3], yz = m[4], zz = m[5];
        return xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz);
    }

    // Solve ``M * x = b`` for symmetric 3x3 ``M`` via the cofactor inverse.
    // Caller guarantees det != 0; on a near-zero det the result is
    // numerically unreliable but won't crash.  ``out_det`` returns the
    // determinant so the Gaussian normalizer can reuse it without
    // recomputing.
    inline constexpr auto sym3_solve(const Sym3 &m, float bx, float by, float bz, float &out_det) noexcept
        -> std::array<float, 3>
    {
        const auto xx = m[0], xy = m[1], xz = m[2];
        const auto yy = m[3], yz = m[4], zz = m[5];

        // Cofactors of the symmetric matrix.
        const auto c00 = yy * zz - yz * yz;
        const auto c01 = xz * yz - xy * zz;
        const auto c02 = xy * yz - xz * yy;
        const auto c11 = xx * zz - xz * xz;
        const auto c12 = xz * xy - xx * yz;
        const auto c22 = xx * yy - xy * xy;

        out_det = xx * c00 + xy * c01 + xz * c02;
        const auto inv_det = 1.F / out_det;

        return {
            (c00 * bx + c01 * by + c02 * bz) * inv_det,
            (c01 * bx + c11 * by + c12 * bz) * inv_det,
            (c02 * bx + c12 * by + c22 * bz) * inv_det,
        };
    }

    // Evaluate N(mu_diff; 0, Sigma_total).
    //
    //   mu_diff       = (mx, my, mz)
    //   Sigma_total   = symmetric 3x3 (already summed: Sigma1 + Sigma2 + r^2*I + ...)
    //
    // Returns the (un-weighted) Gaussian density at the given offset.
    inline auto
    gaussian3_density(float mx, float my, float mz, const Sym3 &sigma_total) noexcept -> float
    {
        constexpr float TWO_PI_POW_3_2 = 15.749609945722419F;  // (2*pi)^(3/2)

        float det;
        const auto sol = sym3_solve(sigma_total, mx, my, mz, det);
        const auto quad = mx * sol[0] + my * sol[1] + mz * sol[2];

        if (det <= 0.F)
        {
            // Degenerate Sigma — caller should add a body-kernel floor.
            // Returning 0 keeps the integral well-defined (a delta-mass
            // obstacle at a point off the sphere has measure zero in
            // the probability sense).
            return 0.F;
        }

        const auto norm = 1.F / (TWO_PI_POW_3_2 * vamp::collision::sqrt(det));
        return norm * vamp::collision::exp(-0.5F * quad);
    }
}  // namespace vamp::collision
