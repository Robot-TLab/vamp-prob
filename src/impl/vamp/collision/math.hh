#pragma once

#include <algorithm>
#include <cmath>

namespace vamp::collision
{
    using Point = std::array<float, 3>;

    template <typename DataT>
    inline constexpr auto dot_2(const DataT &ax, const DataT &ay, const DataT &bx, const DataT &by) -> DataT
    {
        return (ax * bx) + (ay * by);
    }

    template <typename DataT>
    inline constexpr auto dot_3(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &bx,
        const DataT &by,
        const DataT &bz) -> DataT
    {
        return (ax * bx) + (ay * by) + (az * bz);
    }

    template <typename DataT>
    inline constexpr auto sql2_3(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &bx,
        const DataT &by,
        const DataT &bz) -> DataT
    {
        const auto xs = (ax - bx);
        const auto ys = (ay - by);
        const auto zs = (az - bz);

        return dot_3(xs, ys, zs, xs, ys, zs);
    }

    template <typename DataT>
    inline constexpr auto clamp(const DataT &v, const DataT &lower, const DataT &upper) -> DataT
    {
        return v.clamp(lower, upper);
    }

    template <>
    inline constexpr auto clamp<float>(const float &v, const float &lower, const float &upper) -> float
    {
        return std::max(std::min(v, upper), lower);
    }

    template <typename DataT>
    inline constexpr auto sqrt(const DataT &v) -> DataT
    {
        return v.sqrt();
    }

    template <>
    inline constexpr auto sqrt<float>(const float &v) -> float
    {
        return std::sqrt(v);
    }

    // Scalar exp / standard-normal CDF used by the probabilistic
    // collision-checking primitives in ``collision/gaussian.hh``,
    // ``collision/gaussian_gaussian.hh`` and ``collision/visibility.hh``.
    //
    // Both functions operate per-sample (one float in → one float out).
    // The probabilistic primitives are themselves applied per sphere /
    // per kernel; we never need rake-vectorized ``exp``/``erf`` here
    // because the SIMD parallelism in the risk evaluator is over
    // *configurations* (the existing FK rake path), not over per-sphere
    // Gaussian terms.  If profiling later shows the inner Gaussian
    // evaluation becoming a bottleneck, vectorized polynomial-approx
    // ``exp``/``erf`` overloads can be added without changing callers.

    template <typename DataT>
    inline constexpr auto exp(const DataT &v) -> DataT
    {
        return v.exp();
    }

    template <>
    inline constexpr auto exp<float>(const float &v) -> float
    {
        return std::exp(v);
    }

    // Reciprocal 1/v.  ``float`` is exact; the SIMD-lane form uses the hardware
    // ``rcp`` (≈12-bit) refined by one Newton step (≈24-bit ≈ full fp32) — a
    // deliberate speed-over-``operator/`` choice, ample for the per-element
    // Gaussian-density math.  Lets ``DataT``-generic code read like the scalar.
    template <typename DataT>
    inline auto rcp(const DataT &v) -> DataT
    {
        const DataT r = v.rcp();
        return r * (DataT::fill(2.0F) - v * r);
    }

    template <>
    inline auto rcp<float>(const float &v) -> float
    {
        return 1.0F / v;
    }

    // Standard-normal CDF Φ(x) = 0.5·(1 + erf(x / √2)).
    template <typename DataT>
    inline constexpr auto normal_cdf(const DataT &v) -> DataT
    {
        // No SIMD-lane fallback yet (would require .erf() on DataT);
        // see the comment above this block.  Until then, the scalar
        // ``float`` specialization is the only supported instantiation.
        static_assert(
            sizeof(DataT) == 0,
            "normal_cdf is only specialized for ``float``; rake-vectorized "
            "callers should reduce to scalar before invoking.");
        return v;
    }

    template <>
    inline constexpr auto normal_cdf<float>(const float &v) -> float
    {
        constexpr float INV_SQRT_2 = 0.7071067811865475F;  // 1 / sqrt(2)
        return 0.5F * (1.0F + std::erf(v * INV_SQRT_2));
    }

    // 3-D cross product: c = a × b.  Templated like ``dot_3`` so it
    // works on either scalar floats or SIMD lane types that overload
    // arithmetic operators.
    template <typename DataT>
    inline constexpr auto cross_3(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &bx,
        const DataT &by,
        const DataT &bz,
        DataT &cx,
        DataT &cy,
        DataT &cz) -> void
    {
        cx = ay * bz - az * by;
        cy = az * bx - ax * bz;
        cz = ax * by - ay * bx;
    }

    // Non-central chi-squared (k = 2) CDF via Helstrom's series.  Used
    // by ``collision/visibility.hh`` for the rigorous 3-D angular
    // fraction: a Gaussian kernel viewed through a circular FoV cone
    // contributes its inside-cone probability mass, which under the
    // small-angle approximation is exactly P[χ²_2(λ) ≤ z] with
    // ``λ = (δ_k r_k / σ_k)²``, ``z = (ψ/2 · r_k / σ_k)²``.
    //
    // N = 40 Poisson-mixture terms give tail < 1e-8 for λ ≤ 50 and
    // < 1e-7 absolute error vs ``scipy.stats.ncx2.cdf(z, 2, λ)`` over
    // (λ, z) ∈ [0, 50]².  Inner loop is pure recurrences — no pow,
    // tgamma, or branch.  Scalar only; same rationale as ``normal_cdf``
    // above (rake parallelism lives at the FK layer, not here).
    inline auto ncx2_2_cdf(float z, float lam) noexcept -> float
    {
        if (z <= 0.0F)
        {
            return 0.0F;
        }

        // Guard against the deep upper tail where ``exp(-z/2)`` underflows
        // to 0 against a runaway ``cum_z = Σ (z/2)^i / i!``: P[χ²_2(λ) ≤ z]
        // saturates to 1 well before the floats break down.  Use a loose
        // 12·std cutoff so the early return only triggers when we're
        // genuinely past the support, and let the numerically-stable
        // adaptive series below handle the rest.
        const float mean_dist = 2.0F + lam;
        const float std_dist = std::sqrt(4.0F + 4.0F * lam);
        if (z >= mean_dist + 12.0F * std_dist + 50.0F)
        {
            return 1.0F;
        }

        // Switch to double internally for the accumulators: float32 loses
        // > 7 digits to cancellation in the high-λ + high-z regime
        // (chi2k_cdf = 1 − exp(−z/2)·cum_z where cum_z → exp(z/2) and the
        // two are nearly equal), which is exactly where visibility lands
        // for kernels at the cone boundary with small σ.
        const double half_z_d = 0.5 * static_cast<double>(z);
        const double half_lam_d = 0.5 * static_cast<double>(lam);
        const double exp_neg_half_z = std::exp(-half_z_d);
        const double exp_neg_half_lam = std::exp(-half_lam_d);

        // Adaptive truncation: the Poisson weight on j peaks near
        // half_lam, the inner cumulant on i peaks near half_z.  We need
        // N to comfortably bracket whichever is larger; +25 buys ~1e-10
        // Poisson tail.  Cap at 1024 so worst-case rake-of-configs
        // callers stay bounded (most calls land far below).
        const int N_target = static_cast<int>(
            std::ceil(std::max(half_z_d, half_lam_d))) + 25;
        const int N = (N_target > 1024) ? 1024 : (N_target < 40 ? 40 : N_target);

        // ``pois_term`` carries the actual Poisson PMF
        // ``exp(-λ/2)·(λ/2)^j/j!`` (folding ``exp_neg_half_lam`` into
        // the initial value), so it stays bounded by 1 even for large λ
        // where the un-normalised ``(λ/2)^j/j!`` would overflow around
        // j ≈ 100.
        double pois_term = exp_neg_half_lam;
        double z_term = 1.0;
        double cum_z = 1.0;
        double chi2k_cdf = 1.0 - exp_neg_half_z * cum_z;
        double acc = pois_term * chi2k_cdf;

        for (int j = 1; j <= N; ++j)
        {
            const auto jf = static_cast<double>(j);
            pois_term *= half_lam_d / jf;
            z_term *= half_z_d / jf;
            cum_z += z_term;
            chi2k_cdf = 1.0 - exp_neg_half_z * cum_z;
            if (chi2k_cdf < 0.0)
            {
                // Cancellation can push chi2k_cdf very slightly
                // negative for j ≫ z; clamp so the accumulator stays
                // monotone.
                chi2k_cdf = 0.0;
            }
            acc += pois_term * chi2k_cdf;
        }
        return static_cast<float>(acc);
    }
}  // namespace vamp::collision
