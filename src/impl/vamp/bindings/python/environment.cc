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

    // 3D Gaussian obstacle for probabilistic collision checking.  Mean
    // (mx, my, mz), symmetric covariance laid out as upper triangle
    // (sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz),
    // and an occupancy weight alpha (defaults to 1.0).
    nb::class_<vc::GaussianObstacle<float>>(pymodule, "GaussianObstacle")
        .def(
            "__init__",
            [](vc::GaussianObstacle<float> *q,
               const std::array<float, 3> &mean,
               const std::array<float, 6> &sigma_upper,
               float alpha) noexcept
            {
                new (q) vc::GaussianObstacle<float>(
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
        .def_ro("mx", &vc::GaussianObstacle<float>::mx)
        .def_ro("my", &vc::GaussianObstacle<float>::my)
        .def_ro("mz", &vc::GaussianObstacle<float>::mz)
        .def_ro("sigma_xx", &vc::GaussianObstacle<float>::sigma_xx)
        .def_ro("sigma_xy", &vc::GaussianObstacle<float>::sigma_xy)
        .def_ro("sigma_xz", &vc::GaussianObstacle<float>::sigma_xz)
        .def_ro("sigma_yy", &vc::GaussianObstacle<float>::sigma_yy)
        .def_ro("sigma_yz", &vc::GaussianObstacle<float>::sigma_yz)
        .def_ro("sigma_zz", &vc::GaussianObstacle<float>::sigma_zz)
        .def_ro("alpha", &vc::GaussianObstacle<float>::alpha)
        .def_ro("three_sigma_extent", &vc::GaussianObstacle<float>::three_sigma_extent)
        .def_prop_ro(
            "mean",
            [](vc::GaussianObstacle<float> &g)
            { return std::array<float, 3>{g.mx, g.my, g.mz}; })
        .def_prop_ro(
            "sigma_upper",
            [](vc::GaussianObstacle<float> &g) {
                return std::array<float, 6>{
                    g.sigma_xx, g.sigma_xy, g.sigma_xz, g.sigma_yy, g.sigma_yz, g.sigma_zz};
            })
        .def_ro("min_distance", &vc::GaussianObstacle<float>::min_distance)
        .def_rw("name", &vc::GaussianObstacle<float>::name);

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
            "add_gaussian_obstacle",
            [](vc::Environment<float> &e, const vc::GaussianObstacle<float> &g)
            {
                e.gaussian_obstacles.emplace_back(g);
                e.sort();
            })
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
        .def_ro("pointclouds", &vc::Environment<float>::pointclouds)
        .def_ro("gaussian_obstacles", &vc::Environment<float>::gaussian_obstacles);

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

    pymodule.def(
        "observation_reward",
        [](float qx, float qy, float phi,
           float sigma_rho, float d_max, float psi,
           const std::vector<std::array<float, 4>> &kernels) -> float
        {
            std::vector<vc::RiskKernel> kv;
            kv.reserve(kernels.size());
            for (const auto &k : kernels)
            {
                kv.push_back(vc::RiskKernel{k[0], k[1], k[2], k[3]});
            }
            return vc::observation_reward(qx, qy, phi, sigma_rho, d_max, psi,
                                          kv.data(), kv.size());
        },
        "qx"_a, "qy"_a, "phi"_a, "sigma_rho"_a, "d_max"_a, "psi"_a,
        "kernels"_a,
        "Camera-cone observation reward O(q, phi) for a sensor at "
        "(qx, qy) with gaze ``phi`` evaluating a list of risk kernels "
        "[x, y, z, weight].");

    pymodule.def(
        "optimal_gaze",
        [](float qx, float qy, float theta,
           float sigma_rho, float d_max, float psi, float psi_h,
           const std::vector<std::array<float, 4>> &kernels,
           int n_iter) -> std::pair<float, float>
        {
            std::vector<vc::RiskKernel> kv;
            kv.reserve(kernels.size());
            for (const auto &k : kernels)
            {
                kv.push_back(vc::RiskKernel{k[0], k[1], k[2], k[3]});
            }
            return vc::optimal_gaze(qx, qy, theta, sigma_rho, d_max, psi, psi_h,
                                    kv.data(), kv.size(), n_iter);
        },
        "qx"_a, "qy"_a, "theta"_a, "sigma_rho"_a, "d_max"_a, "psi"_a,
        "psi_h"_a, "kernels"_a, "n_iter"_a = 20,
        "Bounded golden-section search for the optimal gaze direction "
        "phi* over the head-sweep range [theta - psi_h/2, theta + psi_h/2]. "
        "Returns (phi_star, O_star).");

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
