from collections.abc import Sequence
from typing import Annotated, overload

from numpy.typing import ArrayLike

import _core_ext


def dimension() -> int:
    """Dimension of configuration space for this robot."""

def resolution() -> int:
    """Collision checking resolution for this robot."""

def n_spheres() -> int:
    """Number of spheres in robot collision model."""

def space_measure() -> float:
    """Measure of robot's C-space."""

def min_max_radii() -> tuple[float, float]:
    """Minimum and maximum radii sizes of robot spheres."""

def joint_names() -> list[str]:
    """Joint names for the robot in order of DoF"""

def end_effector() -> str:
    """End-effector frame name."""

def upper_bounds() -> Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]: ...

def lower_bounds() -> Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]: ...

class RNG:
    """RNG for robot configurations."""

    def reset(self) -> None:
        """Reset the RNG to initial state and seed."""

    def next(self) -> Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]:
        """Sample the next configuration. Modifies internal RNG state."""

    def skip(self, arg: int, /) -> None:
        """Skip the next n iterations."""

class ProlateHyperspheroid:
    """Prolate Hyperspheroid for Robot."""

    def __init__(self, arg0: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], arg1: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], /) -> None:
        """Construct from two loci."""

    def set_transverse_diameter(self, arg: float, /) -> None: ...

    def transform(self, arg: "vamp::Vector<vamp::SIMDVector<float __vector(8)>, 1ul, 14ul>", /) -> "vamp::Vector<vamp::SIMDVector<float __vector(8)>, 1ul, 14ul>": ...

def halton() -> RNG:
    """Creates a new Halton sampler."""

def phs_sampler(arg0: ProlateHyperspheroid, arg1: RNG, /) -> RNG:
    """Creates a new PHS sampler."""

def xorshift() -> RNG:
    """Creates a new XORShift sampler."""

class Path:
    """Path in configuration space represented as discrete waypoints."""

    def __init__(self) -> None:
        """Default constructor, creates empty path."""

    def __len__(self) -> int:
        """Return the number of waypoints in the path."""

    def __getitem__(self, arg: int, /) -> Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]:
        """Get the i-th configuration in the path."""

    @overload
    def __setitem__(self, arg0: int, arg1: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], /) -> None:
        """Set the i-th configuration of the the path."""

    @overload
    def __setitem__(self, arg0: int, arg1: Sequence[float], /) -> None: ...

    @overload
    def append(self, arg: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], /) -> None:
        """Append a configuration to the end of this path."""

    @overload
    def append(self, arg: Sequence[float], /) -> None: ...

    @overload
    def insert(self, arg0: int, arg1: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], /) -> None:
        """Append a configuration to the end of this path."""

    @overload
    def insert(self, arg0: int, arg1: Sequence[float], /) -> None: ...

    def cost(self) -> float:
        """Compute the total path length (by the l2-norm) of the path."""

    def subdivide(self) -> None:
        """
        Subdivide the path by inserting a configuration at the midpoint of all existing segments.
        """

    def interpolate_to_resolution(self, arg: int, /) -> None:
        """
        Refine the path by interpolating all segments up to the resolution provided.
        """

    def interpolate_to_n_states(self, arg: int, /) -> None:
        """Refine the path by interpolating to n states as even as possible."""

    def validate(self, arg: _core_ext.Environment, /) -> bool:
        """Validate the path in an environment."""

    def numpy(self) -> Annotated[ArrayLike, dict(dtype='float32', device='cpu', writable=False)]:
        """Convert this path to a numpy matrix."""

class PlanningResult:
    """Result of a planning query."""

    def __init__(self) -> None:
        """Empty constructor."""

    @property
    def solved(self) -> bool:
        """Returns true if solution found."""

    @property
    def path(self) -> Path:
        """The solution path, if the path is found."""

    @property
    def nanoseconds(self) -> int:
        """Nanoseconds taken to find the path."""

    @property
    def iterations(self) -> int:
        """Number of planner iterations used to find the path."""

    @property
    def size(self) -> list[int]:
        """Size of the internal planner datastructures."""

class Roadmap:
    """Undirected graph in configuration space."""

    def __init__(self) -> None:
        """Empty constructor."""

    def __len__(self) -> int:
        """Return the number of vertices in the roadmap."""

    def __getitem__(self, arg: int, /) -> Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]:
        """Get the i-th vertex."""

    @property
    def vertices(self) -> list["vamp::Vector<vamp::SIMDVector<float __vector(8)>, 1ul, 14ul>"]:
        """List of all vertices (configurations) in the roadmap."""

    @property
    def edges(self) -> list[list[int]]:
        """List of all undirected edge pairs, by vertex index."""

    @property
    def nanoseconds(self) -> int:
        """Nanoseconds taken to construct roadmap."""

    @property
    def iterations(self) -> int:
        """Number of iterations taken to construct roadmap."""

def simplify(path: Path, environment: _core_ext.Environment, settings: _core_ext.SimplifySettings, rng: RNG) -> PlanningResult:
    """Simplification heuristics to post-process a path."""

@overload
def fk(configuration: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]) -> list[_core_ext.Sphere]:
    """
    Computes the forward kinematics of the robot. Returns array of all collision sphere positions.
    """

@overload
def fk(configuration: Sequence[float]) -> list[_core_ext.Sphere]: ...

@overload
def eefk(configuration: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]) -> Annotated[ArrayLike, dict(dtype='float32', shape=(4, 4), order='F')]:
    """
    Computes the forward kinematics of the robot's end-effector. Returns XYZ and a XYZW quaternion.
    """

@overload
def eefk(configuration: Sequence[float]) -> Annotated[ArrayLike, dict(dtype='float32', shape=(4, 4), order='F')]: ...

@overload
def debug(configuration: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment = ...) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """Check which spheres of a robot configuration are in collision."""

@overload
def debug(configuration: Sequence[float], environment: _core_ext.Environment = ...) -> tuple[list[list[str]], list[tuple[int, int]]]: ...

@overload
def validate(configuration: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment = ..., check_bounds: bool = False) -> bool:
    """Check if a configuration is valid. Returns true if valid."""

@overload
def validate(configuration: Sequence[float], environment: _core_ext.Environment = ..., check_bounds: bool = False) -> bool: ...

@overload
def validate_motion(configuration_in: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], configuration_out: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment = ..., check_bounds: bool = True) -> bool:
    """Check if a configuration is valid. Returns true if valid."""

@overload
def validate_motion(configuration_in: Sequence[float], configuration_out: Sequence[float], environment: _core_ext.Environment = ..., check_bounds: bool = True) -> bool: ...

@overload
def filter_self_from_pointcloud(pc: Sequence[Sequence[float]], point_radius: float, configuration: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment = ...) -> list[list[float]]:
    """
    Removes points from pointcloud which collide with the robot and environment.
    """

@overload
def filter_self_from_pointcloud(pc: Sequence[Sequence[float]], point_radius: float, configuration: Sequence[float], environment: _core_ext.Environment = ...) -> list[list[float]]: ...

@overload
def roadmap(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> Roadmap:
    """PRM roadmap construction."""

@overload
def roadmap(start: Sequence[float], goal: Sequence[float], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> Roadmap: ...

@overload
def rrtc(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment, settings: _core_ext.RRTCSettings, rng: RNG) -> PlanningResult:
    """RRTConnect"""

@overload
def rrtc(start: Sequence[float], goal: Sequence[float], environment: _core_ext.Environment, settings: _core_ext.RRTCSettings, rng: RNG) -> PlanningResult: ...

@overload
def rrtc(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Sequence[Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]], environment: _core_ext.Environment, settings: _core_ext.RRTCSettings, rng: RNG) -> PlanningResult: ...

@overload
def rrtc(start: Sequence[float], goal: Sequence[Sequence[float]], environment: _core_ext.Environment, settings: _core_ext.RRTCSettings, rng: RNG) -> PlanningResult: ...

@overload
def prm(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> PlanningResult:
    """PRM"""

@overload
def prm(start: Sequence[float], goal: Sequence[float], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> PlanningResult: ...

@overload
def prm(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Sequence[Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> PlanningResult: ...

@overload
def prm(start: Sequence[float], goal: Sequence[Sequence[float]], environment: _core_ext.Environment, settings: _core_ext.PRMSettings, rng: RNG) -> PlanningResult: ...

@overload
def fcit(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment, settings: _core_ext.FCITSettings, rng: RNG) -> PlanningResult:
    """FCIT"""

@overload
def fcit(start: Sequence[float], goal: Sequence[float], environment: _core_ext.Environment, settings: _core_ext.FCITSettings, rng: RNG) -> PlanningResult: ...

@overload
def fcit(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Sequence[Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]], environment: _core_ext.Environment, settings: _core_ext.FCITSettings, rng: RNG) -> PlanningResult: ...

@overload
def fcit(start: Sequence[float], goal: Sequence[Sequence[float]], environment: _core_ext.Environment, settings: _core_ext.FCITSettings, rng: RNG) -> PlanningResult: ...

@overload
def aorrtc(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], environment: _core_ext.Environment, settings: _core_ext.AORRTCSettings, rng: RNG) -> PlanningResult:
    """AORRTC"""

@overload
def aorrtc(start: Sequence[float], goal: Sequence[float], environment: _core_ext.Environment, settings: _core_ext.AORRTCSettings, rng: RNG) -> PlanningResult: ...

@overload
def aorrtc(start: Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')], goal: Sequence[Annotated[ArrayLike, dict(dtype='float32', shape=(14), device='cpu')]], environment: _core_ext.Environment, settings: _core_ext.AORRTCSettings, rng: RNG) -> PlanningResult: ...

@overload
def aorrtc(start: Sequence[float], goal: Sequence[Sequence[float]], environment: _core_ext.Environment, settings: _core_ext.AORRTCSettings, rng: RNG) -> PlanningResult: ...
