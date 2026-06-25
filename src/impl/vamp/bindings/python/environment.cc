#include <vamp_python_init.hh>

#include <vamp/collision/environment.hh>
#include <vamp/collision/filter.hh>
#include <vamp/collision/capt.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/gaussian.hh>
#include <vamp/collision/shapes.hh>
#include <vamp/collision/visibility.hh>

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
namespace vc = vamp::collision;
namespace vf = vamp::collision::factory;
using namespace nb::literals;

void vamp::binding::init_environment(nanobind::module_ &pymodule)
{
    nb::class_<vc::Sphere<float>>(pymodule, "Sphere")
        .def(
            "__init__",
            [](vc::Sphere<float> *q, const std::array<float, 3> &center, float radius) noexcept
            { new (q) vc::Sphere<float>(vf::sphere::array(center, radius)); },
            "Constructor from center and radius.")
        .def_static("make_flat", &vf::sphere::flat)
        .def_static("make", &vf::sphere::array)
        .def_ro("x", &vc::Sphere<float>::x)
        .def_ro("y", &vc::Sphere<float>::y)
        .def_ro("z", &vc::Sphere<float>::z)
        .def_ro("r", &vc::Sphere<float>::r)
        .def_prop_ro(
            "position",
            [](vc::Sphere<float> &sphere) { return std::array<float, 3>{sphere.x, sphere.y, sphere.z}; })
        .def_ro("min_distance", &vc::Sphere<float>::min_distance)
        .def_rw("name", &vc::Sphere<float>::name);

    nb::class_<vc::Cylinder<float>>(pymodule, "Cylinder")
        .def(
            "__init__",
            [](vc::Cylinder<float> *q,
               const std::array<float, 3> &center,
               const std::array<float, 3> &euler_xyz,
               float radius,
               float length) noexcept
            { new (q) vc::Cylinder<float>(vf::cylinder::center::array(center, euler_xyz, radius, length)); },
            "Constructor from center, Euler XYZ orientation, radius, and length.")
        .def(
            "__init__",
            [](vc::Cylinder<float> *q,
               const std::array<float, 3> &endpoint1,
               const std::array<float, 3> &endpoint2,
               float radius) noexcept { *q = vf::cylinder::endpoints::array(endpoint1, endpoint2, radius); },
            "Constructor from endpoints and radius.")
        .def_ro("x1", &vc::Cylinder<float>::x1)
        .def_ro("y1", &vc::Cylinder<float>::y1)
        .def_ro("z1", &vc::Cylinder<float>::z1)
        .def_prop_ro("x2", &vc::Cylinder<float>::x2)
        .def_prop_ro("y2", &vc::Cylinder<float>::y2)
        .def_prop_ro("z2", &vc::Cylinder<float>::z2)
        .def_ro("xv", &vc::Cylinder<float>::xv)
        .def_ro("yv", &vc::Cylinder<float>::yv)
        .def_ro("zv", &vc::Cylinder<float>::zv)
        .def_ro("r", &vc::Cylinder<float>::r)
        .def_ro("rdv", &vc::Cylinder<float>::rdv)
        .def_ro("min_distance", &vc::Cylinder<float>::min_distance)
        .def_rw("name", &vc::Cylinder<float>::name);

    nb::class_<vc::Cuboid<float>>(pymodule, "Cuboid")
        .def(
            "__init__",
            [](vc::Cuboid<float> *q,
               const std::array<float, 3> &center,
               const std::array<float, 3> &euler_xyz,
               const std::array<float, 3> &half_extents) noexcept
            { new (q) vc::Cuboid<float>(vf::cuboid::array(center, euler_xyz, half_extents)); },
            "Constructor from center, Euler XYZ orientation, and XYZ half-extents.")
        .def_ro("x", &vc::Cuboid<float>::x)
        .def_ro("y", &vc::Cuboid<float>::y)
        .def_ro("z", &vc::Cuboid<float>::z)
        .def_ro("axis_1_x", &vc::Cuboid<float>::axis_1_x)
        .def_ro("axis_1_y", &vc::Cuboid<float>::axis_1_y)
        .def_ro("axis_1_z", &vc::Cuboid<float>::axis_1_z)
        .def_ro("axis_2_x", &vc::Cuboid<float>::axis_2_x)
        .def_ro("axis_2_y", &vc::Cuboid<float>::axis_2_y)
        .def_ro("axis_2_z", &vc::Cuboid<float>::axis_2_z)
        .def_ro("axis_3_x", &vc::Cuboid<float>::axis_3_x)
        .def_ro("axis_3_y", &vc::Cuboid<float>::axis_3_y)
        .def_ro("axis_3_z", &vc::Cuboid<float>::axis_3_z)
        .def_ro("axis_1_r", &vc::Cuboid<float>::axis_1_r)
        .def_ro("axis_2_r", &vc::Cuboid<float>::axis_2_r)
        .def_ro("axis_3_r", &vc::Cuboid<float>::axis_3_r)
        .def_ro("min_distance", &vc::Cuboid<float>::min_distance)
        .def_rw("name", &vc::Cuboid<float>::name);

    // 3-D Gaussian distribution: the primary type vamp uses for every
    // primitive that integrates over a Gaussian (sphere-vs-Gaussian
    // collision risk, visibility / observation reward, ...).  Mean
    // (mx, my, mz), symmetric covariance laid out as upper triangle
    // (sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz),
    // and a scalar weight alpha (defaults to 1.0).  This is the one
    // Gaussian type: the risk path sums it and the visibility primitives
    // consume it directly.
    nb::class_<vc::Gaussian3<float>>(pymodule, "Gaussian3")
        .def(
            "__init__",
            [](vc::Gaussian3<float> *q,
               const std::array<float, 3> &mean,
               const std::array<float, 6> &sigma_upper,
               float alpha) noexcept
            {
                new (q) vc::Gaussian3<float>(
                    mean[0],
                    mean[1],
                    mean[2],
                    sigma_upper[0],
                    sigma_upper[1],
                    sigma_upper[2],
                    sigma_upper[3],
                    sigma_upper[4],
                    sigma_upper[5],
                    alpha);
            },
            "mean"_a,
            "sigma_upper"_a,
            "alpha"_a = 1.0F,
            "Constructor: mean (xyz), covariance upper triangle "
            "(xx, xy, xz, yy, yz, zz), occupancy weight alpha.")
        .def_rw("mx", &vc::Gaussian3<float>::mx)
        .def_rw("my", &vc::Gaussian3<float>::my)
        .def_rw("mz", &vc::Gaussian3<float>::mz)
        .def_rw("sigma_xx", &vc::Gaussian3<float>::sigma_xx)
        .def_rw("sigma_xy", &vc::Gaussian3<float>::sigma_xy)
        .def_rw("sigma_xz", &vc::Gaussian3<float>::sigma_xz)
        .def_rw("sigma_yy", &vc::Gaussian3<float>::sigma_yy)
        .def_rw("sigma_yz", &vc::Gaussian3<float>::sigma_yz)
        .def_rw("sigma_zz", &vc::Gaussian3<float>::sigma_zz)
        .def_rw("alpha", &vc::Gaussian3<float>::alpha)
        .def_prop_ro(
            "mean",
            [](vc::Gaussian3<float> &g)
            { return std::array<float, 3>{g.mx, g.my, g.mz}; })
        .def_prop_ro(
            "sigma_upper",
            [](vc::Gaussian3<float> &g) {
                return std::array<float, 6>{
                    g.sigma_xx, g.sigma_xy, g.sigma_xz, g.sigma_yy, g.sigma_yz, g.sigma_zz};
            })
        .def("trace_sigma", &vc::Gaussian3<float>::trace_sigma)
        .def("iso_sigma", &vc::Gaussian3<float>::iso_sigma);


    pymodule.def("make_heightfield", &vf::heightfield::array);

    nb::class_<vc::HeightField<float>>(pymodule, "HeightField")
        .def_ro("x", &vc::HeightField<float>::x)
        .def_ro("y", &vc::HeightField<float>::y)
        .def_ro("z", &vc::HeightField<float>::z)
        .def_ro("xs", &vc::HeightField<float>::xs)
        .def_ro("ys", &vc::HeightField<float>::ys)
        .def_ro("zs", &vc::HeightField<float>::zs)
        .def_ro("data", &vc::HeightField<float>::data);

    nb::class_<vc::Environment<float>>(pymodule, "Environment")
        .def(nb::init<>())
        .def(
            "add_sphere",
            [](vc::Environment<float> &e, const vc::Sphere<float> &s)
            {
                e.spheres.emplace_back(s);
                e.sort();
            })
        .def(
            "add_cuboid",
            [](vc::Environment<float> &e, const vc::Cuboid<float> &s)
            {
                if (s.axis_3_z == 1.)
                {
                    e.z_aligned_cuboids.emplace_back(s);
                }
                else
                {
                    e.cuboids.emplace_back(s);
                }
                e.sort();
            })
        .def(
            "add_capsule",
            [](vc::Environment<float> &e, const vc::Cylinder<float> &s)
            {
                if (s.xv == 0. and s.yv == 0.)
                {
                    e.z_aligned_capsules.emplace_back(s);
                }
                else
                {
                    e.capsules.emplace_back(s);
                }
                e.sort();
            })
        .def(
            "add_heightfield",
            [](vc::Environment<float> &e, const vc::HeightField<float> &s)
            { e.heightfields.emplace_back(s); })
        .def(
            "add_pointcloud",
            [](vc::Environment<float> &e,
               const std::vector<collision::Point> &pc,
               float r_min,
               float r_max,
               float r_point)
            {
                auto start_time = std::chrono::steady_clock::now();
                e.pointclouds.emplace_back(pc, r_min, r_max, r_point);
                return vamp::utils::get_elapsed_nanoseconds(start_time);
            })
        .def(
            "attach",
            [](vc::Environment<float> &e, const vc::Attachment<float> &a) { e.attachments.emplace(a); })
        .def("detach", [](vc::Environment<float> &e) { e.attachments.reset(); })
        .def_ro("spheres", &vc::Environment<float>::spheres)
        .def_ro("cuboids", &vc::Environment<float>::cuboids)
        .def_ro("z_aligned_cuboids", &vc::Environment<float>::z_aligned_cuboids)
        .def_ro("capsules", &vc::Environment<float>::capsules)
        .def_ro("z_aligned_capsules", &vc::Environment<float>::z_aligned_capsules)
        .def_ro("heightfields", &vc::Environment<float>::heightfields)
        .def_ro("pointclouds", &vc::Environment<float>::pointclouds);

    pymodule.def(
        "filter_pointcloud",
        [](const std::vector<collision::Point> &pc,
           float min_dist,
           float max_range,
           const collision::Point &origin,
           const collision::Point &workcell_min,
           const collision::Point &workcell_max,
           bool cull) -> std::pair<std::vector<collision::Point>, std::size_t>
        {
            auto start_time = std::chrono::steady_clock::now();
            auto filtered =
                vc::filter_pointcloud(pc, min_dist, max_range, origin, workcell_min, workcell_max, cull);
            return {filtered, vamp::utils::get_elapsed_nanoseconds(start_time)};
        });

    pymodule.def(
        "filter_pointcloud",
        [](const nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> &pc,
           float min_dist,
           float max_range,
           const collision::Point &origin,
           const collision::Point &workcell_min,
           const collision::Point &workcell_max,
           bool cull) -> std::pair<std::vector<collision::Point>, std::size_t>
        {
            auto start_time = std::chrono::steady_clock::now();
            auto filtered =
                vc::filter_pointcloud(pc, min_dist, max_range, origin, workcell_min, workcell_max, cull);
            return {filtered, vamp::utils::get_elapsed_nanoseconds(start_time)};
        });

    // ── Probabilistic CC + visibility math primitives ─────────────────
    // Robot-agnostic free functions that consumers can call directly to
    // evaluate the per-sphere Gaussian terms, the observation-cone
    // integral, or the optimal-gaze line search — without going through
    // a planner instance.

    pymodule.def(
        "gaussian3_density",
        [](const std::array<float, 3> &mean,
           const std::array<float, 6> &sigma_upper) -> float
        {
            const auto sigma = vc::Sym3{
                sigma_upper[0], sigma_upper[1], sigma_upper[2],
                sigma_upper[3], sigma_upper[4], sigma_upper[5]};
            return vc::gaussian3_density(mean[0], mean[1], mean[2], sigma);
        },
        "mean"_a, "sigma_upper"_a,
        "Evaluate N(mean; 0, sigma) for a 3D Gaussian with symmetric "
        "covariance given as upper-triangle row-major "
        "(xx, xy, xz, yy, yz, zz).");

    // 3-D observation reward.  Camera at world position (cx, cy, cz)
    // looking along unit vector (nx, ny, nz).  Per-kernel σ comes from
    // each Gaussian's covariance (no separate sigma_rho).  Accepts
    // ``List[Gaussian3]``.
    pymodule.def(
        "observation_reward",
        [](float cx, float cy, float cz,
           float nx, float ny, float nz,
           float d_max, float psi,
           const std::vector<vc::Gaussian3<float>> &gaussians) -> float
        {
            return vc::observation_reward(
                cx, cy, cz, nx, ny, nz, d_max, psi,
                gaussians.data(), gaussians.size());
        },
        "cx"_a, "cy"_a, "cz"_a, "nx"_a, "ny"_a, "nz"_a,
        "d_max"_a, "psi"_a, "gaussians"_a,
        "Camera-cone observation reward O(c, n_gaze) for a 3-D camera at "
        "world position (cx, cy, cz) looking along unit vector (nx, ny, nz). "
        "Per-kernel angular fraction is the non-central chi-squared (k=2) "
        "tail under the small-angle approximation (Marcum-Q form); radial "
        "fraction is the standard-normal CDF mass within d_max.  σ per "
        "kernel is sqrt(tr(Σ)/3).");

    // 3-D optimal gaze.  Searches (az, el) head-frame offsets within the
    // head-sweep window and the per-axis joint-limit clamps.  R_head is
    // row-major 3×3.  Returns (az*, el*, O*).
    pymodule.def(
        "optimal_gaze",
        [](float cx, float cy, float cz,
           const std::array<float, 9> &R_head,
           const std::array<float, 3> &n_ref,
           float d_max, float psi,
           float psi_h_az, float psi_h_el,
           float az_min, float az_max,
           float el_min, float el_max,
           const std::vector<vc::Gaussian3<float>> &gaussians,
           int n_grid, int n_refine) -> std::tuple<float, float, float>
        {
            return vc::optimal_gaze(
                cx, cy, cz,
                R_head[0], R_head[1], R_head[2],
                R_head[3], R_head[4], R_head[5],
                R_head[6], R_head[7], R_head[8],
                n_ref[0], n_ref[1], n_ref[2],
                d_max, psi,
                psi_h_az, psi_h_el,
                az_min, az_max, el_min, el_max,
                gaussians.data(), gaussians.size(),
                n_grid, n_refine);
        },
        "cx"_a, "cy"_a, "cz"_a,
        "R_head"_a, "n_ref"_a,
        "d_max"_a, "psi"_a,
        "psi_h_az"_a, "psi_h_el"_a,
        "az_min"_a, "az_max"_a, "el_min"_a, "el_max"_a,
        "gaussians"_a, "n_grid"_a = 9, "n_refine"_a = 20,
        "Bounded 2-D search for the optimal (az*, el*) head-frame gaze "
        "offsets maximising the observation reward.  R_head is row-major "
        "3×3; gaze = R_head · Rz(az) · Ry(el) · n_ref.  Search strategy: "
        "coarse n_grid×n_grid uniform scan over the window "
        "[-psi_h_*/2, +psi_h_*/2] ∩ [*_min, *_max], then per-axis "
        "golden-section refinement around the best cell (n_refine "
        "iterations each).  Returns (az*, el*, O*).");

    // Direct binding for the non-central chi-squared (k=2) CDF used by
    // observation_reward.  Exposed for unit testing against scipy.
    pymodule.def(
        "ncx2_2_cdf",
        [](float z, float lam) -> float
        { return vc::ncx2_2_cdf(z, lam); },
        "z"_a, "lam"_a,
        "P[χ²_2(λ) ≤ z] via Helstrom series (N=40 Poisson-mixture "
        "terms).  Used internally by observation_reward for the 3-D "
        "angular fraction.");

    nb::class_<vc::Attachment<float>>(pymodule, "Attachment")
        .def(
            "__init__",
            [](vc::Attachment<float> *q, Eigen::Matrix4f &tf) noexcept
            {
                Eigen::Isometry3f iso;
                iso.matrix() = tf;
                new (q) vc::Attachment<float>(iso);
            },
            "Constructor for an attachment centered at a relative transform from the end-effector.")
        .def_prop_ro("relative_frame", [](vc::Attachment<float> &a) { return a.tf; })
        .def(
            "add_sphere",
            [](vc::Attachment<float> &a, collision::Sphere<float> &sphere)
            { a.spheres.emplace_back(sphere); })
        .def(
            "add_spheres",
            [](vc::Attachment<float> &a, std::vector<collision::Sphere<float>> &spheres)
            { a.spheres.insert(a.spheres.end(), spheres.cbegin(), spheres.cend()); })
        .def(
            "set_ee_pose",
            [](vc::Attachment<float> &a, Eigen::Matrix4f &tf)
            {
                Eigen::Isometry3f iso;
                iso.matrix() = tf;
                a.pose(iso);
            },
            "tf"_a)
        .def_ro("posed_spheres", &vc::Attachment<float>::posed_spheres);
}
