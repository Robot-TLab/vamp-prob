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
    // ``collision/sphere_gaussian.hh`` and ``collision/visibility.hh``.
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
}  // namespace vamp::collision
