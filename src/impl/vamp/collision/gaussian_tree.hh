#pragma once

// ``GaussianTree`` — a spatial index over a set of 3-D Gaussians
// (``Gaussian3<float>``, any covariance) that returns the summed
// Gaussian-product overlap against a query Gaussian.
//
// It is the accelerated, *complete* counterpart of the linear gaussian-obstacle
// scan in ``risk_validity.hh``: rather than evaluating ``gaussian_gaussian``
// against every stored Gaussian, it descends a bucketed k-d tree over the
// Gaussian means and prunes any subtree past the query's reach.  It is fully
// generic in what it stores — a dense static-map surface enters as
// zero-covariance Diracs, a tracked obstacle as a blob with Σ_obs > 0 — and is
// instantiated and filled by the caller (the planner), not auto-built as a vamp
// primitive.
//
// Completeness is the point: a risk is a *sum*, so (unlike the CAPT afford-set
// boolean, which preserves "some point is near" but not the full near set) no
// in-reach Gaussian may be dropped.  The descent only prunes Gaussians whose
// contribution is < e^{-kRiskSigma²/2}.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <vamp/collision/capt.hh>  // Point, Volume
#include <vamp/collision/gaussian.hh>
#include <vamp/collision/gaussian_gaussian.hh>
#include <vamp/vector.hh>

namespace vamp::collision
{
    struct GaussianTree
    {
        using FVectorT = FloatVector<>;

        // Mahalanobis cull radius: past ``kRiskSigma`` standard deviations of
        // the combined covariance the overlap is < e^{-kRiskSigma²/2} and is
        // dropped.  The reach is taken along √tr(Σ) ≥ √λ_max — a safe
        // over-estimate that never prunes a contributing Gaussian.
        static constexpr float kRiskSigma = 5.0F;

        // Leaf bucket size: a node with at most this many Gaussians becomes a
        // leaf whose members are summed as contiguous SIMD blocks.
        static constexpr std::size_t kBucket = 32;

        struct Node
        {
            Volume box;  // bounds the stored Gaussian *means* in this subtree
            std::int32_t left = -1;
            std::int32_t right = -1;
            std::uint32_t block_begin = 0;
            std::uint32_t block_count = 0;
        };

        std::vector<Node> nodes;

        // Struct-of-arrays leaf storage: ``FVectorT::num_scalars`` Gaussians per
        // block, fields in ``Gaussian3`` order (mx, my, mz, σxx, σxy, σxz, σyy,
        // σyz, σzz, α).  Padding lanes carry a +inf mean (zero overlap).
        std::array<std::vector<FVectorT>, 10> leaf_blocks;

        // Largest tr(Σ) over the stored Gaussians; widens the query reach so a
        // blob whose spread reaches the query is never pruned.
        float max_trace = 0.0F;

        GaussianTree() = default;

        explicit GaussianTree(std::vector<Gaussian3<float>> gaussians) noexcept
        {
            if (gaussians.empty())
            {
                return;
            }
            for (const auto &g : gaussians)
            {
                max_trace = std::max(max_trace, g.sigma_xx + g.sigma_yy + g.sigma_zz);
            }
            nodes.reserve(2 * gaussians.size() / kBucket + 16);
            build(gaussians, 0, static_cast<int>(gaussians.size()));
        }

        // Σ over every stored Gaussian g of gaussian_gaussian(query, g).  Scalar
        // in the query (a rake's lanes take distinct tree paths, so the rake
        // batching is the caller's per-lane loop); SIMD over the stored set.
        [[nodiscard]] auto sum_overlap(const Gaussian3<float> &query) const noexcept -> float
        {
            if (nodes.empty())
            {
                return 0.0F;
            }
            const float trace = query.sigma_xx + query.sigma_yy + query.sigma_zz;
            const float reach = kRiskSigma * std::sqrt(trace + max_trace);
            const Point center{query.mx, query.my, query.mz};
            const Gaussian3<FVectorT> query_v{
                FVectorT::fill(query.mx),       FVectorT::fill(query.my),
                FVectorT::fill(query.mz),       FVectorT::fill(query.sigma_xx),
                FVectorT::fill(query.sigma_xy), FVectorT::fill(query.sigma_xz),
                FVectorT::fill(query.sigma_yy), FVectorT::fill(query.sigma_yz),
                FVectorT::fill(query.sigma_zz), FVectorT::fill(query.alpha)};
            return descend(0, center, reach * reach, query_v);
        }

      private:
        static auto mean_of(const Gaussian3<float> &g) noexcept -> Point
        {
            return Point{g.mx, g.my, g.mz};
        }

        auto build(std::vector<Gaussian3<float>> &gs, int begin, int end) noexcept -> std::int32_t
        {
            constexpr float kInf = std::numeric_limits<float>::infinity();
            Volume box{{kInf, kInf, kInf}, {-kInf, -kInf, -kInf}};
            for (int i = begin; i < end; ++i)
            {
                box.extend(mean_of(gs[i]));
            }

            const auto id = static_cast<std::int32_t>(nodes.size());
            nodes.push_back(Node{box});

            if (static_cast<std::size_t>(end - begin) > kBucket)
            {
                std::uint8_t dim = 0;
                float widest = -1.0F;
                for (std::uint8_t k = 0; k < 3; ++k)
                {
                    const float w = box.upper[k] - box.lower[k];
                    if (w > widest)
                    {
                        widest = w;
                        dim = k;
                    }
                }

                const int mid = begin + (end - begin) / 2;
                std::nth_element(
                    gs.begin() + begin, gs.begin() + mid, gs.begin() + end,
                    [dim](const Gaussian3<float> &a, const Gaussian3<float> &b) noexcept
                    { return mean_of(a)[dim] < mean_of(b)[dim]; });

                const auto l = build(gs, begin, mid);
                const auto r = build(gs, mid, end);
                nodes[id].left = l;
                nodes[id].right = r;
            }
            else
            {
                nodes[id].block_begin = static_cast<std::uint32_t>(leaf_blocks[0].size());
                pack_leaf(gs, begin, end);
                nodes[id].block_count =
                    static_cast<std::uint32_t>(leaf_blocks[0].size()) - nodes[id].block_begin;
            }
            return id;
        }

        void pack_leaf(const std::vector<Gaussian3<float>> &gs, int begin, int end) noexcept
        {
            constexpr float kInf = std::numeric_limits<float>::infinity();
            constexpr int width = static_cast<int>(FVectorT::num_scalars);
            for (int i = begin; i < end; i += width)
            {
                std::array<std::array<float, FVectorT::num_scalars>, 10> f{};
                for (int l = 0; l < width; ++l)
                {
                    const int j = i + l;
                    if (j < end)
                    {
                        const auto &g = gs[j];
                        f[0][l] = g.mx;       f[1][l] = g.my;       f[2][l] = g.mz;
                        f[3][l] = g.sigma_xx; f[4][l] = g.sigma_xy; f[5][l] = g.sigma_xz;
                        f[6][l] = g.sigma_yy; f[7][l] = g.sigma_yz; f[8][l] = g.sigma_zz;
                        f[9][l] = g.alpha;
                    }
                    else
                    {
                        // +inf mean ⇒ zero overlap; remaining fields irrelevant.
                        f[0][l] = kInf; f[1][l] = kInf; f[2][l] = kInf;
                    }
                }
                for (std::size_t c = 0; c < 10; ++c)
                {
                    leaf_blocks[c].emplace_back(f[c]);
                }
            }
        }

        [[nodiscard]] auto descend(
            std::int32_t n,
            const Point &center,
            float reach_sq,
            const Gaussian3<FVectorT> &query_v) const noexcept -> float
        {
            const Node &nd = nodes[n];
            if (nd.box.distsq_to(center) > reach_sq)
            {
                return 0.0F;
            }

            if (nd.left < 0)
            {
                const std::uint32_t end = nd.block_begin + nd.block_count;
                float acc = 0.0F;
                for (std::uint32_t b = nd.block_begin; b < end; ++b)
                {
                    const Gaussian3<FVectorT> stored{
                        leaf_blocks[0][b], leaf_blocks[1][b], leaf_blocks[2][b],
                        leaf_blocks[3][b], leaf_blocks[4][b], leaf_blocks[5][b],
                        leaf_blocks[6][b], leaf_blocks[7][b], leaf_blocks[8][b],
                        leaf_blocks[9][b]};
                    const auto contrib = gaussian_gaussian(query_v, stored).to_array();
                    for (const float v : contrib)
                    {
                        acc += v;
                    }
                }
                return acc;
            }

            return descend(nd.left, center, reach_sq, query_v) +
                   descend(nd.right, center, reach_sq, query_v);
        }
    };
}  // namespace vamp::collision
