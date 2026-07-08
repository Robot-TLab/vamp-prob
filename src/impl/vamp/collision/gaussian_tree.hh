#pragma once

// ``GaussianTree`` — a spatial index over a set of uncertain obstacle spheres
// (``Gaussian3<float>``: a Gaussian-uncertain centre + a physical ``radius``)
// that returns the probability the query body sphere collides with NONE of
// them — i.e. the noisy-OR "no-collision" product ∏_o (1 − p_o), where each
// ``p_o`` is the two-uncertain-spheres collision probability
// (``gaussian_gaussian.hh``).  The caller turns that into a collision
// probability ``P = 1 − ∏(1 − p_o)`` (noisy-OR across obstacles AND body
// spheres); nothing is summed and nothing is counted.
//
// This is the "CAPT near obstacles" query the risk evaluator wants: because a
// far obstacle has ``p_o ≈ 0`` (factor ≈ 1, no effect on the product), the
// descent prunes any subtree whose obstacle means are past the query's reach
// ``R + prob_sigma·s`` and only ever touches the handful of near obstacles.
// Unlike the old density SUM (which demanded completeness — "no in-reach
// Gaussian may be dropped"), a saturating probability only needs the near ones:
// a single coincident obstacle drives its factor to 0 and the product to 0
// (P → 1), so a genuine collision reads ~1 from just a few spheres.
//
// It stores whatever the caller fills it with — a dense static-map surface as
// small-radius spheres, a tracked obstacle as a blob with its own Σ and body
// radius — and is instantiated by the planner, not auto-built as a vamp
// primitive.

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

        // Reach cull: an obstacle at centre-distance d contributes a collision
        // probability that decays once (d − R)/s exceeds a few standard
        // deviations of the combined position spread s.  Past ``prob_sigma`` the
        // per-pair p is < ~e^{−prob_sigma²/2} and the subtree is dropped.  The
        // reach is taken along √tr(Σ)/√3 ≥ √λ_min and widened by the largest
        // stored radius/spread, a safe over-estimate that never prunes a
        // contributing obstacle.  Configurable per tree via the constructor's
        // ``prob_sigma`` argument; ``kDefaultProbSigma`` is the historical 4.0.
        // A larger value culls less (more obstacles evaluated, more accurate P);
        // a smaller value culls more aggressively (faster, drops weaker tails).
        static constexpr float kDefaultProbSigma = 4.0F;
        float prob_sigma = kDefaultProbSigma;

        // Leaf bucket size: a node with at most this many spheres becomes a
        // leaf whose members are summed as contiguous SIMD blocks.
        static constexpr std::size_t kBucket = 32;

        struct Node
        {
            Volume box;  // bounds the stored obstacle *centres* in this subtree
            std::int32_t left = -1;
            std::int32_t right = -1;
            std::uint32_t block_begin = 0;
            std::uint32_t block_count = 0;
        };

        std::vector<Node> nodes;

        // Struct-of-arrays leaf storage: ``FVectorT::num_scalars`` spheres per
        // block, fields in ``Gaussian3`` order EXCEPT the 10th slot carries the
        // physical ``radius`` (the risk path never uses ``alpha``): (mx, my, mz,
        // σxx, σxy, σxz, σyy, σyz, σzz, radius).  Padding lanes carry a +inf mean
        // (zero collision probability ⇒ factor 1 in the product).
        std::array<std::vector<FVectorT>, 10> leaf_blocks;

        // Largest tr(Σ) and largest radius over the stored spheres; widen the
        // query reach so a spread-out or large obstacle is never pruned.
        float max_trace = 0.0F;
        float max_radius = 0.0F;

        GaussianTree() = default;

        explicit GaussianTree(
            std::vector<Gaussian3<float>> gaussians,
            float prob_sigma_arg = kDefaultProbSigma) noexcept
          : prob_sigma(prob_sigma_arg)
        {
            if (gaussians.empty())
            {
                return;
            }
            for (const auto &g : gaussians)
            {
                max_trace = std::max(max_trace, g.sigma_xx + g.sigma_yy + g.sigma_zz);
                max_radius = std::max(max_radius, g.radius);
            }
            nodes.reserve(2 * gaussians.size() / kBucket + 16);
            build(gaussians, 0, static_cast<int>(gaussians.size()));
        }

        // ∏ over every stored obstacle o of (1 − gaussian_gaussian(query, o)) —
        // the probability the query sphere collides with none of them.  Scalar
        // in the query (a rake's lanes take distinct tree paths); SIMD over the
        // stored set.
        [[nodiscard]] auto no_collision_product(
            const Gaussian3<float> &query, float min_no_collision = 0.0F) const noexcept
            -> float
        {
            if (nodes.empty())
            {
                return 1.0F;
            }
            const float trace = query.sigma_xx + query.sigma_yy + query.sigma_zz;
            // Floor the spread to match the primitive (kSphereSigmaFloorSq) so
            // the reach never under-estimates for near-zero covariance.
            const float s =
                std::sqrt(std::max((trace + max_trace) / 3.0F, kSphereSigmaFloorSq));
            const float reach = query.radius + max_radius + prob_sigma * s;
            const Point center{query.mx, query.my, query.mz};
            const Gaussian3<FVectorT> query_v{
                FVectorT::fill(query.mx),       FVectorT::fill(query.my),
                FVectorT::fill(query.mz),       FVectorT::fill(query.sigma_xx),
                FVectorT::fill(query.sigma_xy), FVectorT::fill(query.sigma_xz),
                FVectorT::fill(query.sigma_yy), FVectorT::fill(query.sigma_yz),
                FVectorT::fill(query.sigma_zz), FVectorT::fill(query.alpha),
                FVectorT::fill(query.radius)};
            return descend(0, center, reach * reach, query_v, min_no_collision);
        }

        // Packed counterpart of ``no_collision_product``: the query carries
        // ``rake`` distinct lanes (e.g. the rake of *adjacent* edge waypoints of
        // one body sphere), and the return is the per-lane no-collision product.
        // ONE descent serves every lane — a subtree is pruned only when *all*
        // lanes are out of reach — and each leaf broadcasts its stored spheres
        // against the packed query, so the lanes stay packed end to end.
        // Requires the caller to pack *spatially close* lanes, else the shared
        // descent over-includes.
        //
        // ``min_no_collision`` is the noisy-OR reject early-out floor: once every
        // lane's accumulated no-collision product has dropped to ``≤ floor`` (i.e.
        // every lane's P ≥ 1 − floor is already over the risk budget), the descent
        // stops — the remaining obstacles can only lower the product further, so
        // the edge is already infeasible.  ``0`` disables the early-out (full,
        // exact product).  The floor is a safe over-approximation: it never stops
        // before P genuinely exceeds the threshold, so the feasibility verdict is
        // unchanged; only the reported P of an already-rejected lane is a lower
        // bound (it is ``≥`` the threshold, hence still rejected).
        template <std::size_t rake>
        [[nodiscard]] auto no_collision_product(
            const Gaussian3<FloatVector<rake>> &query,
            float min_no_collision = 0.0F) const noexcept -> FloatVector<rake>
        {
            using FV = FloatVector<rake>;
            if (nodes.empty())
            {
                return FV::fill(1.0F);
            }
            // Per-lane reach² = (r_query + max_radius + kProbSigma·s)², with the
            // isotropic spread s floored so ``sqrt`` never sees a zero (rsqrt).
            const FV var =
                ((query.sigma_xx + query.sigma_yy + query.sigma_zz) + FV::fill(max_trace)) *
                (1.0F / 3.0F);
            const FV s = var.max(FV::fill(kSphereSigmaFloorSq)).sqrt();
            const FV reach = (query.radius + FV::fill(max_radius)) + s * FV::fill(prob_sigma);
            const FV reach_sq = reach * reach;
            FV prod = FV::fill(1.0F);
            descend_simd<rake>(0, query, reach_sq, prod, min_no_collision);
            return prod;
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
                        f[9][l] = g.radius;  // 10th slot = physical radius, not alpha
                    }
                    else
                    {
                        // +inf mean ⇒ zero collision probability; remaining
                        // fields irrelevant (radius left 0).
                        f[0][l] = kInf; f[1][l] = kInf; f[2][l] = kInf;
                    }
                }
                for (std::size_t c = 0; c < 10; ++c)
                {
                    leaf_blocks[c].emplace_back(f[c]);
                }
            }
        }

        // Scalar-query descent: returns ∏(1 − p) over near obstacles in this
        // subtree.  Prune ⇒ factor 1 (identity for the product).
        [[nodiscard]] auto descend(
            std::int32_t n,
            const Point &center,
            float reach_sq,
            const Gaussian3<FVectorT> &query_v,
            float min_no_collision) const noexcept -> float
        {
            const Node &nd = nodes[n];
            if (nd.box.distsq_to(center) > reach_sq)
            {
                return 1.0F;
            }

            if (nd.left < 0)
            {
                const std::uint32_t end = nd.block_begin + nd.block_count;
                float prod = 1.0F;
                for (std::uint32_t b = nd.block_begin; b < end; ++b)
                {
                    const Gaussian3<FVectorT> stored{
                        leaf_blocks[0][b], leaf_blocks[1][b], leaf_blocks[2][b],
                        leaf_blocks[3][b], leaf_blocks[4][b], leaf_blocks[5][b],
                        leaf_blocks[6][b], leaf_blocks[7][b], leaf_blocks[8][b],
                        FVectorT::fill(1.0F),  // alpha (unused by collision prob)
                        leaf_blocks[9][b]};    // radius
                    const auto p = gaussian_gaussian(query_v, stored).to_array();
                    for (const float pv : p)
                    {
                        prod *= (1.0F - pv);
                    }
                    // Reject early-out: this subtree alone already saturates P.
                    if (min_no_collision > 0.0F and prod <= min_no_collision)
                    {
                        return prod;
                    }
                }
                return prod;
            }

            const float pl = descend(nd.left, center, reach_sq, query_v, min_no_collision);
            if (min_no_collision > 0.0F and pl <= min_no_collision)
            {
                return pl;
            }
            return pl * descend(nd.right, center, reach_sq, query_v, min_no_collision);
        }

        // Packed descent: multiply per-lane (1 − p) for every near obstacle by
        // broadcasting each stored sphere against the rake-packed query.  Prune
        // a subtree only when *every* lane is out of reach.
        template <std::size_t rake>
        void descend_simd(
            std::int32_t n,
            const Gaussian3<FloatVector<rake>> &query,
            const FloatVector<rake> &reach_sq,
            FloatVector<rake> &prod,
            float min_no_collision) const noexcept
        {
            using FV = FloatVector<rake>;

            // Reject early-out: every lane's no-collision product is already at or
            // below the floor (every lane's P is at/over the risk budget), so no
            // remaining obstacle can change the verdict.  This entry check also
            // propagates the stop across the recursion — once the left subtree
            // saturates every lane, the right subtree returns here immediately.
            if (min_no_collision > 0.0F and
                prod.test_all_less_equal(FV::fill(min_no_collision)))
            {
                return;
            }

            const Node &nd = nodes[n];

            // Distance² from the node's centre-bounding box to each lane's query
            // centre, all lanes at once.
            const FV dx = query.mx - query.mx.clamp(nd.box.lower[0], nd.box.upper[0]);
            const FV dy = query.my - query.my.clamp(nd.box.lower[1], nd.box.upper[1]);
            const FV dz = query.mz - query.mz.clamp(nd.box.lower[2], nd.box.upper[2]);
            const FV dsq = dx * dx + dy * dy + dz * dz;
            // Prune the whole subtree only when no lane can reach it.
            if (dsq.test_all_greater_equal(reach_sq))
            {
                return;
            }

            if (nd.left < 0)
            {
                const std::uint32_t end = nd.block_begin + nd.block_count;
                for (std::uint32_t b = nd.block_begin; b < end; ++b)
                {
                    const auto mx = leaf_blocks[0][b].to_array();
                    const auto my = leaf_blocks[1][b].to_array();
                    const auto mz = leaf_blocks[2][b].to_array();
                    const auto sxx = leaf_blocks[3][b].to_array();
                    const auto sxy = leaf_blocks[4][b].to_array();
                    const auto sxz = leaf_blocks[5][b].to_array();
                    const auto syy = leaf_blocks[6][b].to_array();
                    const auto syz = leaf_blocks[7][b].to_array();
                    const auto szz = leaf_blocks[8][b].to_array();
                    const auto rad = leaf_blocks[9][b].to_array();
                    for (std::size_t l = 0; l < FVectorT::num_scalars; ++l)
                    {
                        // Padding lanes carry a +inf mean ⇒ p = 0 ⇒ factor 1.
                        const Gaussian3<FV> stored{
                            FV::fill(mx[l]),  FV::fill(my[l]),  FV::fill(mz[l]),
                            FV::fill(sxx[l]), FV::fill(sxy[l]), FV::fill(sxz[l]),
                            FV::fill(syy[l]), FV::fill(syz[l]), FV::fill(szz[l]),
                            FV::fill(1.0F),   FV::fill(rad[l])};
                        prod = prod * (FV::fill(1.0F) - gaussian_gaussian(query, stored));
                    }
                    // Reject early-out after each stored block: bail once every
                    // lane is over budget instead of finishing this leaf.
                    if (min_no_collision > 0.0F and
                        prod.test_all_less_equal(FV::fill(min_no_collision)))
                    {
                        return;
                    }
                }
                return;
            }

            descend_simd<rake>(nd.left, query, reach_sq, prod, min_no_collision);
            descend_simd<rake>(nd.right, query, reach_sq, prod, min_no_collision);
        }
    };
}  // namespace vamp::collision
