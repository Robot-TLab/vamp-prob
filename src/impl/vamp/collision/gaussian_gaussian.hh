#pragma once

// The atomic probabilistic-collision primitive: the probability that two
// UNCERTAIN SPHERES overlap.  A robot body sphere and an obstacle are each a
// hard sphere (physical radius ``r``) whose CENTRE is Gaussian-uncertain
// (mean ``μ``, position covariance ``Σ``).  They collide iff their centres come
// within ``R = r_a + r_o`` of each other, and the centre gap
// ``d = x_a − x_o`` is Gaussian with mean ``Δ = μ_a − μ_o`` and covariance
// ``S = Σ_a + Σ_o``.  The per-pair collision probability is therefore the mass
// of that gap distribution inside the ball of radius ``R``:
//
//     p = P(‖d‖ ≤ R),   d ~ N(Δ, S).
//
// This is a genuine probability in [0, 1] that SATURATES to 1 for a deep,
// certain collision (Δ = 0, R ≫ √S) and decays to 0 far away — unlike the old
// density-overlap kernel, whose value was an unbounded expected-overlap *mass*
// scaled by a hand-tuned occupancy weight and never approached 1 in genuine
// collision.  The physical extents ``r_a, r_o`` provide the saturation scale;
// the uncertainties ``Σ_a, Σ_o`` only soften the boundary — so a single nearby
// obstacle can carry the whole risk (no pile-up of dense samples needed).
//
// Isotropic projection.  A closed form for the ball integral exists only for an
// isotropic gap covariance, so we project ``S`` onto its isotropic equivalent
// ``s² = tr(S)/3`` — the same convention ``Gaussian3::iso_sigma`` /
// ``visibility.hh`` already use.  With ``m = ‖Δ‖`` the exact 3-D result is
//
//     p = Φ((R−m)/s) + Φ((R+m)/s) − 1
//         − (s/(m·√(2π)))·[ e^{−(R−m)²/2s²} − e^{−(R+m)²/2s²} ],
//
// where the last term is replaced by its finite ``m→0`` limit
// ``√(2/π)·(R/s)·e^{−R²/2s²}`` near contact.  Written ``DataT``-generic so the
// same routine serves a whole rake of lanes; the SIMD batching (FK, obstacle
// iteration) lives in the caller, exactly as for the sphere primitives.  Only
// the ``FloatVector`` instantiations are used (never scalar ``float``), so the
// ``.max``/``.blend``/``.less_than`` lane ops below are always available.

#include <vamp/collision/gaussian.hh>
#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

namespace vamp::collision
{
    // Floor on the isotropic gap std ``s`` (metres).  As ``s → 0`` the pair
    // degrades to the deterministic hard-sphere test (p = 1 iff m ≤ R); the
    // floor keeps the ``1/s`` terms finite with a ~0.1 mm soft boundary.
    inline constexpr float kSphereSigmaFloor = 1e-4F;
    inline constexpr float kSphereSigmaFloorSq = kSphereSigmaFloor * kSphereSigmaFloor;

    // Probability that two uncertain spheres overlap (see file header).
    template <typename DataT>
    inline auto gaussian_gaussian(const Gaussian3<DataT> &a, const Gaussian3<DataT> &b) noexcept
        -> DataT
    {
        constexpr float INV_SQRT_2PI = 0.3989422804014327F;  // 1 / √(2π)

        // Centre gap Δ = μ_a − μ_b and its magnitude m = ‖Δ‖.  The +ε under the
        // root keeps the rsqrt-based ``sqrt`` off exact zero (it returns NaN at
        // 0); at coincident centres m ≈ 1e-6, which correctly trips the m→0
        // contact branch below.
        const DataT dx = a.mx - b.mx;
        const DataT dy = a.my - b.my;
        const DataT dz = a.mz - b.mz;
        const DataT m = sqrt((dx * dx + dy * dy + dz * dz) + DataT::fill(1e-12F));

        // Isotropic combined position spread s² = tr(Σ_a + Σ_b)/3, floored so
        // the pair never divides by zero (hard-sphere limit).
        const DataT tr =
            (a.sigma_xx + b.sigma_xx) + (a.sigma_yy + b.sigma_yy) + (a.sigma_zz + b.sigma_zz);
        const DataT s_sq = (tr * (1.0F / 3.0F)).max(DataT::fill(kSphereSigmaFloorSq));
        const DataT s = sqrt(s_sq);
        const DataT inv_s = rcp(s);

        // Minkowski radius R = r_a + r_o and the two normalised offsets.
        const DataT R = a.radius + b.radius;
        const DataT u = (R - m) * inv_s;  // (R − m)/s
        const DataT v = (R + m) * inv_s;  // (R + m)/s

        // Φ((R−m)/s) + Φ((R+m)/s) − 1.
        const DataT main = (normal_cdf(u) + normal_cdf(v)) - DataT::fill(1.0F);

        // Gaussian factors e^{−½u²} = e^{−(R−m)²/2s²}, likewise for v.
        const DataT eu = exp(u * u * -0.5F);
        const DataT ev = exp(v * v * -0.5F);

        // Correction term (s/(m√(2π)))·(eu − ev); near m→0 it tends to
        // √(2/π)·(R/s)·e^{−R²/2s²} = 2·(R/s)·e^{−½(R/s)²}/√(2π).
        const DataT m_div = m.max(DataT::fill(1e-9F));  // guard the unused-lane 1/m
        const DataT tail_far = (s * rcp(m_div)) * (eu - ev) * INV_SQRT_2PI;
        const DataT ros = R * inv_s;
        const DataT tail_near = (ros * exp(ros * ros * -0.5F)) * (2.0F * INV_SQRT_2PI);
        // Blend to the m→0 limit where m is tiny relative to s (contact).
        const DataT tail = tail_far.blend(tail_near, m.less_than(s * 1e-2F));

        // Clamp to [0, 1]: removes fp noise so (1 − p) is a well-formed product
        // factor, AND sanitises the tree's padding lanes — a +inf mean gives
        // sqrt(inf)=NaN → NaN here, and clamp(NaN, 0, 1) = 0 (max/min return the
        // non-NaN operand), i.e. a padding obstacle contributes p = 0 (factor 1).
        return (main - tail).clamp(0.0F, 1.0F);
    }
}  // namespace vamp::collision
