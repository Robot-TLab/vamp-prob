#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <vamp/vector.hh>

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

    // exp / standard-normal CDF / non-central χ²₂ CDF used by the
    // probabilistic collision-checking and visibility primitives in
    // ``collision/gaussian.hh``, ``collision/gaussian_gaussian.hh`` and
    // ``collision/visibility.hh``.
    //
    // Each comes in two forms: a scalar ``float`` path (exact libm) and a
    // ``DataT``-generic / rake overload that packs a whole SIMD vector of
    // samples.  The collision-risk path packs *stored Gaussians* per lane
    // (``GaussianTree``); ``observation_reward`` packs *kernels* the same
    // way, so its inner loop needs ``exp``/``erf``/``ncx2`` per lane.  The
    // float forms stay for scalar callers and as the exact reference the
    // rake forms are validated against (and, for ``ncx2_2_cdf``, the
    // per-lane fp64 fallback).

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
    //
    // Rake form: vectorized erf via the Abramowitz & Stegun 7.1.26
    // rational-times-Gaussian approximation (max abs error 1.5e-7 ≈ fp32 ε),
    // built from the lane type's exp / abs / blend.  ``observation_reward``
    // routes its radial fraction Φ((d_max − r)/σ) through this, lane-packed
    // over kernels.  The scalar ``float`` specialization below keeps libm erf.
    template <typename DataT>
    inline auto normal_cdf(const DataT &v) -> DataT
    {
        constexpr float INV_SQRT_2 = 0.7071067811865475F;  // 1 / sqrt(2)
        constexpr float P = 0.3275911F;
        constexpr float A1 = 0.254829592F;
        constexpr float A2 = -0.284496736F;
        constexpr float A3 = 1.421413741F;
        constexpr float A4 = -1.453152027F;
        constexpr float A5 = 1.061405429F;

        const DataT x = v * INV_SQRT_2;  // erf argument
        const DataT ax = x.abs();
        const DataT t = rcp(DataT::fill(1.0F) + ax * P);
        DataT poly = DataT::fill(A5);
        poly = poly * t + A4;
        poly = poly * t + A3;
        poly = poly * t + A2;
        poly = poly * t + A1;
        poly = poly * t;  // (a1 t + … + a5 t⁵)
        const DataT erf_abs = DataT::fill(1.0F) - poly * (-(x * x)).exp();

        // erf is odd; Φ(x) = 0.5·(1 ± erf(|x|)) for x ≷ 0.
        const DataT phi_hi = (erf_abs + 1.0F) * 0.5F;
        const DataT phi_lo = (DataT::fill(1.0F) - erf_abs) * 0.5F;
        return phi_hi.blend(phi_lo, x.less_than(DataT::fill(0.0F)));
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

    // atan2(y, x) for SIMD lane types (scalar callers use ``std::atan2``).
    // Range-reduces to atan on [0, 1] via the standard |y| ≷ |x| swap,
    // evaluates a degree-9 odd minimax polynomial (max abs error ~1.3e-5
    // rad), then folds in the quadrant from the signs of x and y.
    // ``visibility.hh`` feeds it ‖u × n‖ ≥ 0 and ⟨u, n⟩, so the result lands
    // in [0, π] — the same value the scalar ``std::atan2(cross_norm, dot)``
    // returns there; the y < 0 branch is kept for a correct general atan2.
    template <typename DataT>
    inline auto atan2(const DataT &y, const DataT &x) -> DataT
    {
        constexpr float HALF_PI = 1.5707963267948966F;
        constexpr float PI = 3.141592653589793F;

        const DataT ax = x.abs();
        const DataT ay = y.abs();
        const DataT swap = ay.greater_than(ax);  // |y| > |x|
        const DataT num = ay.blend(ax, swap);    // min(|x|, |y|)
        const DataT den = ax.blend(ay, swap);    // max(|x|, |y|)
        const DataT a = num * rcp(den.max(DataT::fill(1.0e-30F)));
        const DataT a2 = a * a;

        // atan(a), a ∈ [0, 1].
        DataT p = DataT::fill(0.0208351F);
        p = p * a2 + (-0.0851330F);
        p = p * a2 + 0.1801410F;
        p = p * a2 + (-0.3302995F);
        p = p * a2 + 0.9998660F;
        p = p * a;

        // Undo the swap, then fold the quadrant from sign(x), sign(y).
        p = p.blend(HALF_PI - p, swap);
        p = p.blend(PI - p, x.less_than(DataT::fill(0.0F)));
        p = p.blend(-p, y.less_than(DataT::fill(0.0F)));
        return p;
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
        const int N_target = static_cast<int>(std::ceil(std::max(half_z_d, half_lam_d))) + 25;
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

    // Rake counterpart of ``ncx2_2_cdf`` above: the non-central χ²₂ CDF for a
    // whole SIMD vector of (z, λ) lanes at once, so ``observation_reward`` can
    // route its angular fraction through SIMD the way the collision-risk path
    // runs ``gaussian_gaussian`` over packed Gaussians.
    //
    // This is the Sankaran (1959, 1963) cube-root normal approximation rather
    // than the scalar path's Poisson-mixture series.  The series is exact but
    // single precision can't hold it: ``exp(-λ/2)`` underflows past λ≈170 and
    // the ``(z/2)^i/i!`` partial sums overflow past z≈1400 (where even the
    // scalar fp64 form NaNs) — and a visibility scene of tight kernels lives
    // mostly in that high-(z, λ) tail (off-axis splats).  Sankaran maps
    // ``(X/(k+λ))^h`` to a standard normal; it is closed-form (no per-lane
    // loop), smooth, monotone, and NaN-free across the whole range — z ≤ 0 → 0
    // and z ≫ mean → 1 fall straight out of the Φ.  Max abs error vs SciPy is
    // ~1e-2 near the cone boundary at small λ (where the reward only feeds an
    // argmax) and ~1e-3 in the large-λ tail; the scalar overload above stays
    // for the SciPy-tested binding and any exact scalar caller.
    template <std::size_t rake>
    inline auto ncx2_2_cdf(const FloatVector<rake> &z, const FloatVector<rake> &lam) noexcept
        -> FloatVector<rake>
    {
        using FV = FloatVector<rake>;
        const FV a = lam + 2.0F;         // k + λ      (k = 2)
        const FV b = lam * 2.0F + 2.0F;  // k + 2λ
        const FV c = lam * 3.0F + 2.0F;  // k + 3λ

        const FV inv_b = rcp(b);
        const FV h = FV::fill(1.0F) - (a * c) * (inv_b * inv_b) * (2.0F / 3.0F);
        const FV p = b * (rcp(a) * rcp(a));  // (k + 2λ) / (k + λ)²
        const FV m = (h - 1.0F) * (FV::fill(1.0F) - h * 3.0F);

        // (X / (k+λ))^h via exp(h·log(·)); the floor keeps log finite at z = 0.
        const FV xa = (z * rcp(a)).max(FV::fill(1.0e-30F));
        const FV base = (h * xa.log()).exp();

        const FV num = base - (FV::fill(1.0F) + h * p * (h - 1.0F) - h * p * m * 0.5F);
        const FV den = h * (p * 2.0F).sqrt() * (FV::fill(1.0F) + m * p * 0.5F);
        return normal_cdf(num * rcp(den));
    }
}  // namespace vamp::collision
