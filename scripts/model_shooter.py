#!/usr/bin/env python3
"""Advanced 2026 FRC FUEL shooter model and calibration tool.

This is an offline design/calibration model, not a replacement for on-robot
closed-loop control. RPM ALWAYS means the measured flywheel/launcher-roller RPM.
The robot PID is responsible for reaching that RPM; motor count, motor torque,
current limits, and gearing are intentionally not part of this ballistic model.

It combines:

* 3-D time-domain projectile dynamics.
* Reynolds-number-dependent quadratic drag.
* Full 3-D ball spin, Magnus lift, and spin decay.
* Robot-center -> shooter-exit geometry and shooting-while-moving compensation.
* Refined yaw lead solved against the simulated trajectory.
* Regular-hex HUB opening plus rim and tapered-funnel contact simulation.
* Monotonic PCHIP RPM -> exit-speed and RPM -> spin calibration.
* Automatic fitting of selected physics parameters from real shot logs.
* Learned shot-to-shot covariance from repeated measurements.
* Sobol quasi-Monte-Carlo uncertainty sweeps.
* Numba-JIT batch trajectory/contact simulation.
* Adaptive Monte Carlo around competitive RPM/angle candidates.
* Multiprocessing or threading with tqdm progress bars.
* Dense continuous distance lookup generation.

Fallback calibration
--------------------
If no measured ``rpm,exit_speed_mps`` calibration is supplied, the physical
speed scale is anchored so that the trusted 2.00 m / 1400 RPM / 70 degree shot
crosses the HUB center while descending.  When the default fixed 70 degree
configuration is used, the final lookup is also pinned to exactly 1400 RPM at
2.00 m.

Recommended calibration CSV columns
-----------------------------------
Only the columns you have are required.  The most useful are::

    distance_m,rpm,exit_speed_mps,spin_rpm,result
    actual_rpm,actual_angle_deg,yaw_error_deg
    entry_angle_deg,flight_time_s
    lip_x_m,lip_y_m
    distance_error_m,lateral_error_m,exit_height_error_m

Here ``rpm`` means measured flywheel/launcher-roller RPM, not motor-command percent.

``result`` values such as ``center``, ``centered``, ``bullseye`` and
``good-center`` are treated as centered/good shots.  ``recommended_rpm`` can
also be supplied to build an empirical residual correction.

Examples
--------
Default fixed-angle model::

    python shooter_model_2026_everybot_pid.py --workers 8

Adjustable hood::

    python shooter_model_2026_everybot_pid.py --optimize-angle --workers 8

Measured calibration and automatic fitting::

    python shooter_model_2026_everybot_pid.py --calibration-csv shots.csv --auto-calibrate --workers 8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.special import ndtri
from scipy.stats import qmc

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("Missing dependency 'tqdm'. Install with: pip install tqdm") from exc

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional slow fallback
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator


# ===========================================================================
# OFFICIAL / ENVIRONMENTAL CONSTANTS
# ===========================================================================

G_MPS2 = 9.80665
AIR_DENSITY_KG_M3 = 1.225
AIR_DYNAMIC_VISCOSITY_PA_S = 1.81e-5

# 2026 FUEL nominal dimensions used by the original model.
FUEL_DIAMETER_M = 0.150
FUEL_MASS_KG = 0.215

# HUB top opening values used by the original model.
HUB_OPENING_FLAT_TO_FLAT_M = 41.7 * 0.0254
HUB_LIP_HEIGHT_M = 72.0 * 0.0254


# ===========================================================================
# YOUR 2026 EVERYBOT SHOOTER CONSTANTS
# ===========================================================================
# RPM semantics:
#   Every RPM value in this file is the ACTUAL MEASURED FLYWHEEL / LAUNCHER-ROLLER
#   RPM. Your robot-side velocity PID handles however many motors are attached and
#   keeps the roller at the requested setpoint. Motor count and electrical details
#   therefore do not belong in this projectile model.
#
# Coordinate convention:
#   +x = robot forward / toward HUB when aimed
#   +y = robot left
#   +z = up
#
# These geometry values are the assumptions requested for your robot.

# Carpet -> CENTER of FUEL at the instant it leaves the shooter.
ROBOT_SHOOTER_EXIT_HEIGHT_M = 0.600

# Robot pose/rotation center -> FUEL release point.
ROBOT_SHOOTER_FORWARD_OFFSET_M = 0.000
ROBOT_SHOOTER_LEFT_OFFSET_M = 0.000

# Actual initial FUEL velocity angle above horizontal.
ROBOT_NOMINAL_LAUNCH_ANGLE_DEG = 70.0

# Stock Everybot launcher wheel diameter. This is useful metadata and for relating
# measured flywheel RPM to surface speed, but measured RPM -> ball-speed calibration
# remains authoritative.
FLYWHEEL_DIAMETER_M = 4.0 * 0.0254  # 0.1016 m
FLYWHEEL_COUNT = 6                   # metadata only; not used in trajectory physics

# Fallback ball-spin prior used only when no measured ``rpm,spin_rpm`` calibration
# is supplied. This is BALL spin RPM per measured FLYWHEEL RPM.
SHOOTER_FALLBACK_BACKSPIN_RPM_PER_FLYWHEEL_RPM = 0.50

# HUB collision geometry. Verify these against official CAD / your practice HUB if
# you need quantitative rim/funnel-contact predictions.
HUB_RIM_RADIAL_WIDTH_M = 0.040
HUB_RIM_THICKNESS_M = 0.012
HUB_FUNNEL_DEPTH_M = 0.330
HUB_FUNNEL_THROAT_FLAT_TO_FLAT_M = 0.610
HUB_CONTACT_RESTITUTION = 0.28
HUB_CONTACT_FRICTION = 0.22

# Aerodynamic priors.  Prefer --auto-calibrate after collecting real shots.
AERO_CD_REFERENCE = 0.47
AERO_CD_LOG_RE_SLOPE = 0.025
AERO_REFERENCE_REYNOLDS = 8.0e4
AERO_MAGNUS_LIFT_SLOPE = 0.10
AERO_MAGNUS_RE_SLOPE = 0.0
AERO_MAX_CL = 0.28
AERO_SPIN_DECAY_TAU_S = 2.5

# Trusted fallback anchor retained from the original script.
CALIBRATION_DISTANCE_M = 2.0
CALIBRATION_RPM = 1400.0

# Lookup/output defaults.
DISTANCES_M = np.arange(1.0, 6.01, 0.5)
ROUND_TO_RPM = 25.0
DENSE_LOOKUP_STEP_M = 0.05


CENTERED_RESULT_LABELS = {
    "center",
    "centered",
    "bullseye",
    "good-center",
    "good_center",
    "good",
    "score-center",
}


# ===========================================================================
# SMALL HELPERS
# ===========================================================================


def _as_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_clip_probability(value: np.ndarray) -> np.ndarray:
    # Sobol may include 0; ndtri(0) is -inf.
    eps = np.finfo(float).eps
    return np.clip(value, eps, 1.0 - eps)


def _round_rpm(value: np.ndarray | float) -> np.ndarray:
    return np.round(np.asarray(value, dtype=float) / ROUND_TO_RPM) * ROUND_TO_RPM


def _hex_normals_numpy() -> np.ndarray:
    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((np.cos(angles), np.sin(angles)))


HEX_NORMALS = _hex_normals_numpy()


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


@dataclass(frozen=True)
class ShooterGeometry:
    exit_height_m: float = ROBOT_SHOOTER_EXIT_HEIGHT_M
    forward_offset_m: float = ROBOT_SHOOTER_FORWARD_OFFSET_M
    left_offset_m: float = ROBOT_SHOOTER_LEFT_OFFSET_M


@dataclass(frozen=True)
class RobotState:
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    omega_rad_s: float = 0.0


@dataclass(frozen=True)
class HubGeometry:
    opening_flat_to_flat_m: float = HUB_OPENING_FLAT_TO_FLAT_M
    lip_height_m: float = HUB_LIP_HEIGHT_M
    rim_radial_width_m: float = HUB_RIM_RADIAL_WIDTH_M
    rim_thickness_m: float = HUB_RIM_THICKNESS_M
    funnel_depth_m: float = HUB_FUNNEL_DEPTH_M
    throat_flat_to_flat_m: float = HUB_FUNNEL_THROAT_FLAT_TO_FLAT_M
    restitution: float = HUB_CONTACT_RESTITUTION
    friction: float = HUB_CONTACT_FRICTION
    desired_extra_clearance_m: float = 0.015

    @property
    def opening_apothem_m(self) -> float:
        return 0.5 * self.opening_flat_to_flat_m

    @property
    def throat_apothem_m(self) -> float:
        return 0.5 * self.throat_flat_to_flat_m

    @property
    def funnel_bottom_z_m(self) -> float:
        return self.lip_height_m - self.rim_thickness_m - self.funnel_depth_m

    def signed_opening_margin_m(
        self, x_m: np.ndarray | float, y_m: np.ndarray | float, radius_m: np.ndarray | float
    ) -> np.ndarray:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        radius = np.asarray(radius_m, dtype=float)
        points = np.stack(np.broadcast_arrays(x, y), axis=-1)
        projections = points @ HEX_NORMALS.T
        center_margin = self.opening_apothem_m - np.max(projections, axis=-1)
        return center_margin - radius


@dataclass(frozen=True)
class AeroModel:
    cd_reference: float = AERO_CD_REFERENCE
    cd_log_re_slope: float = AERO_CD_LOG_RE_SLOPE
    reference_reynolds: float = AERO_REFERENCE_REYNOLDS
    magnus_lift_slope: float = AERO_MAGNUS_LIFT_SLOPE
    magnus_re_slope: float = AERO_MAGNUS_RE_SLOPE
    max_cl: float = AERO_MAX_CL
    spin_decay_tau_s: float = AERO_SPIN_DECAY_TAU_S
    launch_angle_bias_deg: float = 0.0
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.0
    wind_z_mps: float = 0.0


@dataclass(frozen=True)
class BallModel:
    mass_kg: float = FUEL_MASS_KG
    diameter_m: float = FUEL_DIAMETER_M

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def area_m2(self) -> float:
        return math.pi * self.radius_m**2


@dataclass(frozen=True)
class PiecewiseCubicCalibration:
    """Pickle-friendly monotonic PCHIP representation.

    SciPy PchipInterpolator stores coefficients ``c`` such that on interval i,
    with dx = x - knots[i]:

        y = c[0,i] dx^3 + c[1,i] dx^2 + c[2,i] dx + c[3,i]
    """

    knots: tuple[float, ...]
    coefficients_flat: tuple[float, ...]
    coefficient_shape: tuple[int, int]
    source: str
    extrapolate: bool = False

    @classmethod
    def from_points(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        source: str,
        extrapolate: bool = False,
    ) -> "PiecewiseCubicCalibration":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        # Collapse repeated x values to medians.
        unique_x = np.unique(x_arr)
        unique_y = np.array([np.median(y_arr[np.isclose(x_arr, xx)]) for xx in unique_x])
        if len(unique_x) < 2:
            raise ValueError("PCHIP calibration needs at least two unique x values")
        if np.any(np.diff(unique_y) <= 0.0):
            # Isotonic-lite: enforce tiny strictly increasing increments.  This is
            # intentionally conservative; the calibration report calls it out.
            unique_y = np.maximum.accumulate(unique_y)
            for i in range(1, len(unique_y)):
                if unique_y[i] <= unique_y[i - 1]:
                    unique_y[i] = unique_y[i - 1] + 1e-6
        interpolator = PchipInterpolator(unique_x, unique_y, extrapolate=True)
        c = np.asarray(interpolator.c, dtype=float)
        return cls(
            knots=tuple(float(v) for v in unique_x),
            coefficients_flat=tuple(float(v) for v in c.ravel()),
            coefficient_shape=tuple(int(v) for v in c.shape),
            source=source,
            extrapolate=extrapolate,
        )

    @property
    def x_min(self) -> float:
        return self.knots[0]

    @property
    def x_max(self) -> float:
        return self.knots[-1]

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        knots = np.asarray(self.knots, dtype=float)
        c = np.asarray(self.coefficients_flat, dtype=float).reshape(self.coefficient_shape)
        flat = values.ravel()
        indices = np.searchsorted(knots, flat, side="right") - 1
        indices = np.clip(indices, 0, len(knots) - 2)
        if not self.extrapolate:
            if np.any(flat < knots[0]) or np.any(flat > knots[-1]):
                # Clamp rather than unstable polynomial extrapolation; users get a warning.
                flat = np.clip(flat, knots[0], knots[-1])
                indices = np.searchsorted(knots, flat, side="right") - 1
                indices = np.clip(indices, 0, len(knots) - 2)
        dx = flat - knots[indices]
        result = (
            ((c[0, indices] * dx + c[1, indices]) * dx + c[2, indices]) * dx
            + c[3, indices]
        )
        return result.reshape(values.shape)


@dataclass(frozen=True)
class ShooterCalibration:
    speed_curve: PiecewiseCubicCalibration | None
    spin_curve: PiecewiseCubicCalibration | None
    fallback_speed_per_rpm: float
    fallback_spin_per_rpm: float
    source: str

    def speed_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        if self.speed_curve is not None:
            return self.speed_curve.evaluate(rpm)
        return np.asarray(rpm, dtype=float) * self.fallback_speed_per_rpm

    def spin_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        if self.spin_curve is not None:
            return self.spin_curve.evaluate(rpm)
        return np.asarray(rpm, dtype=float) * self.fallback_spin_per_rpm

    def rpm_from_speed(self, speed_mps: float, bounds: tuple[float, float]) -> float:
        low, high = bounds

        def error(rpm: float) -> float:
            return float(self.speed_from_rpm(rpm) - speed_mps)

        if error(low) * error(high) > 0:
            raise ValueError("Requested speed is outside the available RPM calibration")
        return float(brentq(error, low, high))

    @property
    def uses_fallback_anchor(self) -> bool:
        return self.speed_curve is None


@dataclass(frozen=True)
class EmpiricalCorrection:
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    max_abs_correction_rpm: float = 400.0

    def correction_rpm(self, distance_m: np.ndarray | float) -> np.ndarray:
        value = np.asarray(distance_m, dtype=float)
        if len(self.x) < 2:
            return np.zeros_like(value)
        pchip = PchipInterpolator(np.asarray(self.x), np.asarray(self.y), extrapolate=True)
        corrected = np.asarray(pchip(value), dtype=float)
        return np.clip(corrected, -self.max_abs_correction_rpm, self.max_abs_correction_rpm)


@dataclass(frozen=True)
class UncertaintyModel:
    # Model uncertainty.
    mass_sigma_kg: float = 0.006
    cd_reference_sigma: float = 0.035
    cd_log_re_slope_sigma: float = 0.012
    magnus_slope_sigma: float = 0.022
    spin_decay_fraction_sigma: float = 0.18
    calibration_speed_scale_sigma: float = 0.015

    # Default independent shot noise.  Real covariance can override this.
    rpm_sigma: float = 12.0
    exit_speed_fraction_sigma: float = 0.018
    launch_angle_sigma_deg: float = 0.70
    yaw_sigma_deg: float = 0.65
    distance_sigma_m: float = 0.025
    lateral_sigma_m: float = 0.020
    exit_height_sigma_m: float = 0.008
    robot_vx_sigma_mps: float = 0.040
    robot_vy_sigma_mps: float = 0.040
    robot_omega_sigma_rad_s: float = math.radians(2.0)


SHOT_ERROR_NAMES = (
    "rpm_error",
    "exit_speed_fraction_error",
    "angle_error_deg",
    "yaw_error_deg",
    "distance_error_m",
    "lateral_error_m",
    "exit_height_error_m",
    "robot_vx_error_mps",
    "robot_vy_error_mps",
    "robot_omega_error_rad_s",
)


@dataclass(frozen=True)
class ShotErrorDistribution:
    mean: tuple[float, ...]
    covariance_flat: tuple[float, ...]
    dimension: int
    source_rows: int = 0

    @classmethod
    def from_diagonal(cls, uncertainty: UncertaintyModel) -> "ShotErrorDistribution":
        sigma = np.array(
            [
                uncertainty.rpm_sigma,
                uncertainty.exit_speed_fraction_sigma,
                uncertainty.launch_angle_sigma_deg,
                uncertainty.yaw_sigma_deg,
                uncertainty.distance_sigma_m,
                uncertainty.lateral_sigma_m,
                uncertainty.exit_height_sigma_m,
                uncertainty.robot_vx_sigma_mps,
                uncertainty.robot_vy_sigma_mps,
                uncertainty.robot_omega_sigma_rad_s,
            ],
            dtype=float,
        )
        covariance = np.diag(sigma**2)
        return cls(tuple(np.zeros(len(sigma))), tuple(covariance.ravel()), len(sigma), 0)

    def covariance(self) -> np.ndarray:
        return np.asarray(self.covariance_flat, dtype=float).reshape(self.dimension, self.dimension)

    def mean_array(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=float)


@dataclass(frozen=True)
class CalibrationRows:
    rows: tuple[dict[str, str], ...]

    def column(self, name: str) -> np.ndarray:
        return np.asarray([_as_float(row.get(name)) for row in self.rows], dtype=float)

    def text_column(self, name: str) -> list[str]:
        return [str(row.get(name, "")).strip().lower() for row in self.rows]

    @property
    def count(self) -> int:
        return len(self.rows)


@dataclass
class TrajectoryResult:
    reached_hub: bool
    scored: bool
    classification: str
    position_at_lip_m: np.ndarray
    velocity_at_lip_mps: np.ndarray
    flight_time_s: float
    entry_angle_deg: float
    impact_speed_mps: float
    peak_height_m: float
    opening_margin_m: float
    launch_speed_mps: float
    initial_spin_rpm_xyz: np.ndarray
    final_spin_rpm_xyz: np.ndarray
    yaw_deg: float
    rim_contacts: int
    funnel_contacts: int
    t_s: np.ndarray | None = None
    states: np.ndarray | None = None


@dataclass
class CandidateStatistics:
    rpm: np.ndarray
    angle_deg: np.ndarray
    probability: np.ndarray
    clean_probability: np.ndarray
    rim_probability: np.ndarray
    funnel_probability: np.ndarray
    margin_p10_m: np.ndarray
    margin_p50_m: np.ndarray
    margin_p90_m: np.ndarray


@dataclass
class OptimizedShot:
    distance_m: float
    rpm: float
    angle_deg: float
    learned_correction_rpm: float
    combined_probability: float
    model_only_probability: float
    shot_only_probability: float
    clean_probability: float
    contact_probability: float
    probability_band_low_rpm: float
    probability_band_high_rpm: float
    nominal: TrajectoryResult



# ===========================================================================
# CALIBRATION INPUT / PCHIP CURVES
# ===========================================================================


def load_calibration_rows(path: Path | None) -> CalibrationRows:
    if path is None:
        return CalibrationRows(())
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Calibration CSV has no header: {path}")
        rows = tuple({str(k): str(v) for k, v in row.items()} for row in reader)
    return CalibrationRows(rows)


def _valid_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def fit_measured_calibration(rows: CalibrationRows) -> tuple[PiecewiseCubicCalibration | None, PiecewiseCubicCalibration | None]:
    rpm = rows.column("rpm")
    speed = rows.column("exit_speed_mps")
    spin = rows.column("spin_rpm")

    speed_rpm, speed_values = _valid_xy(rpm, speed)
    spin_rpm, spin_values = _valid_xy(rpm, spin)

    speed_curve = None
    spin_curve = None
    if len(np.unique(speed_rpm)) >= 2:
        speed_curve = PiecewiseCubicCalibration.from_points(
            speed_rpm,
            speed_values,
            source=f"PCHIP fit to {len(speed_rpm)} measured exit-speed rows",
            extrapolate=False,
        )
    if len(np.unique(spin_rpm)) >= 2:
        spin_curve = PiecewiseCubicCalibration.from_points(
            spin_rpm,
            spin_values,
            source=f"PCHIP fit to {len(spin_rpm)} measured spin rows",
            extrapolate=False,
        )
    return speed_curve, spin_curve


# ===========================================================================
# GEOMETRY / AERODYNAMICS
# ===========================================================================


def shooter_exit_position(distance_m: float, geometry: ShooterGeometry) -> np.ndarray:
    return np.array(
        [
            -distance_m + geometry.forward_offset_m,
            geometry.left_offset_m,
            geometry.exit_height_m,
        ],
        dtype=float,
    )


def shooter_exit_velocity(robot: RobotState, geometry: ShooterGeometry) -> np.ndarray:
    rotational = np.array(
        [
            -robot.omega_rad_s * geometry.left_offset_m,
            robot.omega_rad_s * geometry.forward_offset_m,
            0.0,
        ],
        dtype=float,
    )
    return np.array([robot.vx_mps, robot.vy_mps, 0.0], dtype=float) + rotational


def launch_unit(angle_deg: float, yaw_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    yaw = math.radians(yaw_deg)
    horizontal = math.cos(angle)
    return np.array(
        [horizontal * math.cos(yaw), horizontal * math.sin(yaw), math.sin(angle)],
        dtype=float,
    )


def default_spin_axis_from_yaw(yaw_deg: float) -> np.ndarray:
    # Positive backspin axis points launch-left for an upward Magnus force.
    yaw = math.radians(yaw_deg)
    return np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=float)


def _cd_from_reynolds(reynolds: float, aero: AeroModel) -> float:
    reynolds = max(reynolds, 1.0)
    cd = aero.cd_reference + aero.cd_log_re_slope * math.log10(reynolds / aero.reference_reynolds)
    return float(np.clip(cd, 0.15, 1.2))


def _cl_from_spin(reynolds: float, spin_ratio: float, aero: AeroModel) -> float:
    re_factor = 1.0 + aero.magnus_re_slope * math.log10(max(reynolds, 1.0) / aero.reference_reynolds)
    cl = aero.magnus_lift_slope * spin_ratio * re_factor
    return float(np.clip(cl, -aero.max_cl, aero.max_cl))


def acceleration_and_spin_derivative(
    velocity_mps: np.ndarray,
    spin_rad_s_xyz: np.ndarray,
    ball: BallModel,
    aero: AeroModel,
) -> tuple[np.ndarray, np.ndarray]:
    wind = np.array([aero.wind_x_mps, aero.wind_y_mps, aero.wind_z_mps], dtype=float)
    relative = velocity_mps - wind
    speed = float(np.linalg.norm(relative))
    acceleration = np.array([0.0, 0.0, -G_MPS2], dtype=float)
    if speed > 1e-9:
        reynolds = AIR_DENSITY_KG_M3 * speed * ball.diameter_m / AIR_DYNAMIC_VISCOSITY_PA_S
        cd = _cd_from_reynolds(reynolds, aero)
        q_area_over_mass = 0.5 * AIR_DENSITY_KG_M3 * ball.area_m2 / ball.mass_kg
        acceleration += -q_area_over_mass * cd * speed * relative

        spin_mag = float(np.linalg.norm(spin_rad_s_xyz))
        if spin_mag > 1e-9:
            spin_ratio = spin_mag * ball.radius_m / speed
            cl = _cl_from_spin(reynolds, spin_ratio, aero)
            # Magnus direction follows omega x v. Sign is retained by spin vector.
            lift = np.cross(spin_rad_s_xyz, relative)
            lift_norm = float(np.linalg.norm(lift))
            if lift_norm > 1e-12:
                lift /= lift_norm
                acceleration += q_area_over_mass * cl * speed**2 * lift

    tau = max(aero.spin_decay_tau_s, 0.05)
    spin_dot = -spin_rad_s_xyz / tau
    return acceleration, spin_dot


# ===========================================================================
# HUB CONTACT MODEL
# ===========================================================================


def _max_hex_projection(x: float, y: float) -> tuple[float, np.ndarray]:
    projections = HEX_NORMALS @ np.array([x, y], dtype=float)
    index = int(np.argmax(projections))
    return float(projections[index]), HEX_NORMALS[index]


def _funnel_apothem_at_z(z_m: float, hub: HubGeometry) -> float:
    top_z = hub.lip_height_m - hub.rim_thickness_m
    bottom_z = hub.funnel_bottom_z_m
    if z_m >= top_z:
        return hub.opening_apothem_m
    if z_m <= bottom_z:
        return hub.throat_apothem_m
    fraction = (top_z - z_m) / max(hub.funnel_depth_m, 1e-9)
    return hub.opening_apothem_m + fraction * (hub.throat_apothem_m - hub.opening_apothem_m)


def _resolve_funnel_contact(
    position: np.ndarray,
    velocity: np.ndarray,
    ball: BallModel,
    hub: HubGeometry,
) -> tuple[np.ndarray, np.ndarray, bool]:
    z = float(position[2])
    if z > hub.lip_height_m or z < hub.funnel_bottom_z_m:
        return position, velocity, False

    allowed = _funnel_apothem_at_z(z, hub) - ball.radius_m
    projection, normal_xy = _max_hex_projection(float(position[0]), float(position[1]))
    penetration = projection - allowed
    if penetration <= 0.0:
        return position, velocity, False

    # Approximate sloped-wall normal.  apothem grows with z; the wall normal has
    # an inward/downward component when viewed from the valid interior volume.
    da_dz = (hub.opening_apothem_m - hub.throat_apothem_m) / max(hub.funnel_depth_m, 1e-9)
    outward = np.array([normal_xy[0], normal_xy[1], -da_dz], dtype=float)
    outward /= max(float(np.linalg.norm(outward)), 1e-12)

    # Push sphere center back into valid interior.
    position = position - penetration * np.array([normal_xy[0], normal_xy[1], 0.0])

    vn = float(np.dot(velocity, outward))
    if vn > 0.0:
        # Moving outward into wall: reflect normal component and damp tangent.
        normal_component = vn * outward
        tangent = velocity - normal_component
        velocity = (1.0 - hub.friction) * tangent - hub.restitution * normal_component
    return position, velocity, True


def simulate_hub_contact_after_lip(
    position_at_lip: np.ndarray,
    velocity_at_lip: np.ndarray,
    spin_rad_s_xyz: np.ndarray,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    dt_s: float = 0.0015,
    max_time_s: float = 0.65,
) -> tuple[bool, str, int, int, np.ndarray, np.ndarray]:
    """Continue a descending shot through rim/funnel contacts.

    This is a compact rigid-contact approximation, not a compliant foam/contact
    FEA model.  It is intentionally parameterized so restitution/friction can be
    fitted from real rim-hit data later.
    """
    position = np.asarray(position_at_lip, dtype=float).copy()
    velocity = np.asarray(velocity_at_lip, dtype=float).copy()
    spin = np.asarray(spin_rad_s_xyz, dtype=float).copy()
    rim_contacts = 0
    funnel_contacts = 0

    clean_margin = float(hub.signed_opening_margin_m(position[0], position[1], ball.radius_m))
    initially_clean = clean_margin >= 0.0

    steps = int(math.ceil(max_time_s / dt_s))
    previous = position.copy()
    for _ in range(steps):
        acceleration, spin_dot = acceleration_and_spin_derivative(velocity, spin, ball, aero)
        velocity += dt_s * acceleration
        spin += dt_s * spin_dot
        previous[:] = position
        position += dt_s * velocity

        # Horizontal rim plate occupies the annular region around the top opening
        # and has real vertical thickness.  Detect downward intersection with its
        # top/bottom slab while the sphere overlaps the annulus.
        projection, normal_xy = _max_hex_projection(float(position[0]), float(position[1]))
        inner = hub.opening_apothem_m - ball.radius_m
        outer = hub.opening_apothem_m + hub.rim_radial_width_m + ball.radius_m
        rim_top = hub.lip_height_m + 0.5 * hub.rim_thickness_m
        rim_bottom = hub.lip_height_m - 0.5 * hub.rim_thickness_m
        sphere_overlaps_rim_z = (
            position[2] - ball.radius_m <= rim_top
            and position[2] + ball.radius_m >= rim_bottom
        )
        in_rim_annulus = projection > inner and projection < outer
        if sphere_overlaps_rim_z and in_rim_annulus and velocity[2] < 0.0:
            rim_contacts += 1
            # Put center just above rim top and bounce vertical velocity.  Also
            # damp horizontal tangent to represent soft foam/polycarbonate contact.
            position[2] = rim_top + ball.radius_m + 1e-5
            velocity[2] = -hub.restitution * velocity[2]
            velocity[:2] *= max(0.0, 1.0 - hub.friction)
            # Small inward component if hit is close enough to the inner lip.
            if projection < hub.opening_apothem_m + 0.5 * hub.rim_radial_width_m:
                velocity[:2] -= 0.25 * abs(velocity[2]) * normal_xy

        position, velocity, contacted = _resolve_funnel_contact(position, velocity, ball, hub)
        if contacted:
            funnel_contacts += 1

        # Successful exit through funnel throat.
        if position[2] <= hub.funnel_bottom_z_m - ball.radius_m:
            projection, _ = _max_hex_projection(float(position[0]), float(position[1]))
            if projection <= hub.throat_apothem_m - 0.15 * ball.radius_m and velocity[2] < 0.0:
                if rim_contacts == 0 and funnel_contacts == 0 and initially_clean:
                    classification = "clean_score"
                elif rim_contacts > 0 and funnel_contacts > 0:
                    classification = "rim_funnel_score"
                elif rim_contacts > 0:
                    classification = "rim_score"
                else:
                    classification = "funnel_score"
                return True, classification, rim_contacts, funnel_contacts, position, velocity
            return False, "miss", rim_contacts, funnel_contacts, position, velocity

        # Escaped above the top after a rim bounce or exited too far radially.
        max_projection, _ = _max_hex_projection(float(position[0]), float(position[1]))
        if position[2] > hub.lip_height_m + 0.35 and velocity[2] > 0.0:
            return False, "miss", rim_contacts, funnel_contacts, position, velocity
        if max_projection > hub.opening_apothem_m + hub.rim_radial_width_m + 0.50:
            return False, "miss", rim_contacts, funnel_contacts, position, velocity

    return False, "miss", rim_contacts, funnel_contacts, position, velocity


# ===========================================================================
# ACCURATE SINGLE-SHOT SIMULATION
# ===========================================================================


def simulate_to_lip(
    distance_m: float,
    flywheel_rpm: float,
    angle_deg: float,
    yaw_deg: float,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    shooter: ShooterGeometry,
    robot: RobotState,
    hub: HubGeometry,
    launch_speed_scale: float = 1.0,
    spin_scale: float = 1.0,
    spin_vector_rpm_xyz: np.ndarray | None = None,
    save_trajectory: bool = False,
    max_time_s: float = 3.0,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray | None, np.ndarray | None]:
    position0 = shooter_exit_position(distance_m, shooter)
    effective_angle = angle_deg + aero.launch_angle_bias_deg
    launch_speed = float(calibration.speed_from_rpm(flywheel_rpm)) * launch_speed_scale
    if launch_speed <= 0.0:
        raise ValueError("Shooter calibration produced a non-positive launch speed")

    velocity0 = launch_speed * launch_unit(effective_angle, yaw_deg) + shooter_exit_velocity(robot, shooter)

    if spin_vector_rpm_xyz is None:
        spin_rpm = float(calibration.spin_from_rpm(flywheel_rpm)) * spin_scale
        spin_axis = default_spin_axis_from_yaw(yaw_deg)
        spin0 = spin_axis * spin_rpm * 2.0 * math.pi / 60.0
    else:
        spin0 = np.asarray(spin_vector_rpm_xyz, dtype=float) * 2.0 * math.pi / 60.0

    state0 = np.concatenate((position0, velocity0, spin0))

    def derivative(_t: float, state: np.ndarray) -> np.ndarray:
        acceleration, spin_dot = acceleration_and_spin_derivative(state[3:6], state[6:9], ball, aero)
        return np.concatenate((state[3:6], acceleration, spin_dot))

    def descending_lip(_t: float, state: np.ndarray) -> float:
        return float(state[2] - hub.lip_height_m)

    descending_lip.terminal = True  # type: ignore[attr-defined]
    descending_lip.direction = -1.0  # type: ignore[attr-defined]

    def ground(_t: float, state: np.ndarray) -> float:
        return float(state[2] - ball.radius_m)

    ground.terminal = True  # type: ignore[attr-defined]
    ground.direction = -1.0  # type: ignore[attr-defined]

    solution = solve_ivp(
        derivative,
        (0.0, max_time_s),
        state0,
        events=(descending_lip, ground),
        rtol=4e-8,
        atol=4e-10,
        max_step=0.008,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    peak = float(np.max(solution.y[2]))
    reached = len(solution.t_events[0]) > 0
    if not reached:
        return (
            False,
            np.full(3, np.nan),
            np.full(3, np.nan),
            np.full(3, np.nan),
            float("nan"),
            peak,
            solution.t if save_trajectory else None,
            solution.y.T if save_trajectory else None,
        )

    state = np.asarray(solution.y_events[0][0], dtype=float)
    return (
        True,
        state[:3],
        state[3:6],
        state[6:9],
        float(solution.t_events[0][0]),
        peak,
        solution.t if save_trajectory else None,
        solution.y.T if save_trajectory else None,
    )


def first_order_lead_yaw_deg(
    distance_m: float,
    rpm: float,
    angle_deg: float,
    calibration: ShooterCalibration,
    aero: AeroModel,
    shooter: ShooterGeometry,
    robot: RobotState,
) -> float:
    p0 = shooter_exit_position(distance_m, shooter)
    target = -p0[:2]
    length = float(np.linalg.norm(target))
    if length < 1e-12:
        return 0.0
    target_unit = target / length
    target_yaw = math.atan2(target_unit[1], target_unit[0])
    perpendicular = np.array([-target_unit[1], target_unit[0]])
    robot_v = shooter_exit_velocity(robot, shooter)[:2]
    v_perp = float(np.dot(robot_v, perpendicular))
    launch_speed = float(calibration.speed_from_rpm(rpm))
    horizontal = max(launch_speed * math.cos(math.radians(angle_deg + aero.launch_angle_bias_deg)), 1e-6)
    delta = math.asin(float(np.clip(-v_perp / horizontal, -0.98, 0.98)))
    return math.degrees(target_yaw + delta)


def solve_exact_lead_yaw_deg(
    distance_m: float,
    rpm: float,
    angle_deg: float,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    shooter: ShooterGeometry,
    robot: RobotState,
    hub: HubGeometry,
) -> float:
    guess = first_order_lead_yaw_deg(distance_m, rpm, angle_deg, calibration, aero, shooter, robot)

    def y_error(yaw: float) -> float:
        try:
            reached, position, *_ = simulate_to_lip(
                distance_m,
                rpm,
                angle_deg,
                yaw,
                calibration,
                ball,
                aero,
                shooter,
                robot,
                hub,
                save_trajectory=False,
            )
        except (ValueError, RuntimeError):
            return 10.0
        if not reached:
            return 10.0
        return float(position[1])

    scan = np.linspace(guess - 12.0, guess + 12.0, 13)
    values = np.array([y_error(float(v)) for v in scan])
    for left, right, fl, fr in zip(scan[:-1], scan[1:], values[:-1], values[1:], strict=True):
        if np.isfinite(fl) and np.isfinite(fr) and fl * fr <= 0.0:
            try:
                return float(brentq(y_error, float(left), float(right), xtol=2e-5))
            except ValueError:
                pass

    # No robust sign change: minimize absolute lateral miss around first-order guess.
    fit = minimize_scalar(lambda yaw: abs(y_error(float(yaw))), bounds=(guess - 15.0, guess + 15.0), method="bounded")
    return float(fit.x if fit.success else guess)


def simulate_shot(
    distance_m: float,
    flywheel_rpm: float,
    angle_deg: float,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    yaw_deg: float | None = None,
    exact_lead: bool = True,
    save_trajectory: bool = False,
) -> TrajectoryResult:
    if yaw_deg is None:
        if exact_lead:
            yaw_deg = solve_exact_lead_yaw_deg(
                distance_m, flywheel_rpm, angle_deg, calibration, ball, aero, shooter, robot, hub
            )
        else:
            yaw_deg = first_order_lead_yaw_deg(
                distance_m, flywheel_rpm, angle_deg, calibration, aero, shooter, robot
            )

    reached, position, velocity, spin_rad, flight_time, peak, t_s, states = simulate_to_lip(
        distance_m,
        flywheel_rpm,
        angle_deg,
        yaw_deg,
        calibration,
        ball,
        aero,
        shooter,
        robot,
        hub,
        save_trajectory=save_trajectory,
    )
    launch_speed = float(calibration.speed_from_rpm(flywheel_rpm))
    initial_spin_rpm = default_spin_axis_from_yaw(yaw_deg) * float(calibration.spin_from_rpm(flywheel_rpm))

    if not reached:
        return TrajectoryResult(
            reached_hub=False,
            scored=False,
            classification="miss",
            position_at_lip_m=np.full(3, np.nan),
            velocity_at_lip_mps=np.full(3, np.nan),
            flight_time_s=float("nan"),
            entry_angle_deg=float("nan"),
            impact_speed_mps=float("nan"),
            peak_height_m=peak,
            opening_margin_m=-float("inf"),
            launch_speed_mps=launch_speed,
            initial_spin_rpm_xyz=initial_spin_rpm,
            final_spin_rpm_xyz=np.full(3, np.nan),
            yaw_deg=float(yaw_deg),
            rim_contacts=0,
            funnel_contacts=0,
            t_s=t_s,
            states=states,
        )

    margin = float(hub.signed_opening_margin_m(position[0], position[1], ball.radius_m))
    horizontal = float(np.hypot(velocity[0], velocity[1]))
    entry = math.degrees(math.atan2(-velocity[2], max(horizontal, 1e-12)))
    impact = float(np.linalg.norm(velocity))

    scored, classification, rim_contacts, funnel_contacts, _, final_velocity = simulate_hub_contact_after_lip(
        position,
        velocity,
        spin_rad,
        ball,
        aero,
        hub,
    )

    return TrajectoryResult(
        reached_hub=True,
        scored=bool(scored),
        classification=classification,
        position_at_lip_m=position,
        velocity_at_lip_mps=velocity,
        flight_time_s=flight_time,
        entry_angle_deg=entry,
        impact_speed_mps=impact,
        peak_height_m=peak,
        opening_margin_m=margin,
        launch_speed_mps=launch_speed,
        initial_spin_rpm_xyz=initial_spin_rpm,
        final_spin_rpm_xyz=spin_rad * 60.0 / (2.0 * math.pi),
        yaw_deg=float(yaw_deg),
        rim_contacts=rim_contacts,
        funnel_contacts=funnel_contacts,
        t_s=t_s,
        states=states,
    )


# ===========================================================================
# FALLBACK 2 m / 1400 RPM CALIBRATION
# ===========================================================================


def fallback_calibration(
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
) -> ShooterCalibration:
    robot = RobotState()
    p0 = shooter_exit_position(CALIBRATION_DISTANCE_M, shooter)
    yaw = math.degrees(math.atan2(-p0[1], -p0[0]))

    # Ball speed is solved directly; a temporary linear calibration converts 1400 RPM
    # into each trial speed.  Root target is x=0 at descending lip crossing.
    def x_error(speed_mps: float) -> float:
        temp = ShooterCalibration(
            speed_curve=None,
            spin_curve=None,
            fallback_speed_per_rpm=speed_mps / CALIBRATION_RPM,
            fallback_spin_per_rpm=SHOOTER_FALLBACK_BACKSPIN_RPM_PER_FLYWHEEL_RPM,
            source="temporary fallback root",
        )
        reached, position, *_ = simulate_to_lip(
            CALIBRATION_DISTANCE_M,
            CALIBRATION_RPM,
            ROBOT_NOMINAL_LAUNCH_ANGLE_DEG,
            yaw,
            temp,
            ball,
            aero,
            shooter,
            robot,
            hub,
        )
        return float(position[0]) if reached else -5.0

    # Usually the full physical speed bounds already bracket the descending-lip
    # center root; use that directly to avoid dozens of expensive solve_ivp trials.
    low_speed, high_speed = 2.0, 35.0
    f_low = x_error(low_speed)
    f_high = x_error(high_speed)
    root = None
    if f_low * f_high <= 0.0:
        root = float(brentq(x_error, low_speed, high_speed, xtol=1e-8))
    else:
        # Defensive fallback for unusual geometry/angles.
        grid = np.linspace(low_speed, high_speed, 24)
        values = np.array([x_error(float(speed)) for speed in grid])
        for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
            if fl * fr <= 0.0:
                root = float(brentq(x_error, float(left), float(right), xtol=1e-8))
                break
    if root is None:
        raise ValueError("Could not anchor fallback calibration at 2.0 m / 1400 RPM")

    return ShooterCalibration(
        speed_curve=None,
        spin_curve=None,
        fallback_speed_per_rpm=root / CALIBRATION_RPM,
        fallback_spin_per_rpm=SHOOTER_FALLBACK_BACKSPIN_RPM_PER_FLYWHEEL_RPM,
        source="fallback linear speed model anchored to 2.00 m = 1400 RPM",
    )


def build_calibration(
    rows: CalibrationRows,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
) -> ShooterCalibration:
    speed_curve, spin_curve = fit_measured_calibration(rows)
    fallback = fallback_calibration(ball, aero, hub, shooter)
    if speed_curve is None:
        return replace(fallback, spin_curve=spin_curve)
    return ShooterCalibration(
        speed_curve=speed_curve,
        spin_curve=spin_curve,
        fallback_speed_per_rpm=fallback.fallback_speed_per_rpm,
        fallback_spin_per_rpm=fallback.fallback_spin_per_rpm,
        source=speed_curve.source,
    )


# ===========================================================================
# REAL-SHOT COVARIANCE LEARNING
# ===========================================================================


def estimate_shot_error_distribution(
    rows: CalibrationRows,
    calibration: ShooterCalibration,
    uncertainty: UncertaintyModel,
) -> ShotErrorDistribution:
    default = ShotErrorDistribution.from_diagonal(uncertainty)
    if rows.count < 8:
        return default

    commanded_rpm = rows.column("rpm")
    actual_rpm = rows.column("actual_rpm")
    speed = rows.column("exit_speed_mps")
    command_angle = rows.column("angle_deg")
    actual_angle = rows.column("actual_angle_deg")
    yaw_error = rows.column("yaw_error_deg")
    distance_error = rows.column("distance_error_m")
    lateral_error = rows.column("lateral_error_m")
    height_error = rows.column("exit_height_error_m")
    robot_vx_error = rows.column("robot_vx_error_mps")
    robot_vy_error = rows.column("robot_vy_error_mps")
    robot_omega_error_deg = rows.column("robot_omega_error_deg_s")

    matrix = np.full((rows.count, len(SHOT_ERROR_NAMES)), np.nan, dtype=float)
    matrix[:, 0] = actual_rpm - commanded_rpm

    predicted_speed = calibration.speed_from_rpm(commanded_rpm)
    valid_prediction = np.isfinite(predicted_speed) & (predicted_speed > 1e-6)
    matrix[valid_prediction, 1] = (speed[valid_prediction] - predicted_speed[valid_prediction]) / predicted_speed[valid_prediction]
    matrix[:, 2] = actual_angle - command_angle
    matrix[:, 3] = yaw_error
    matrix[:, 4] = distance_error
    matrix[:, 5] = lateral_error
    matrix[:, 6] = height_error
    matrix[:, 7] = robot_vx_error
    matrix[:, 8] = robot_vy_error
    matrix[:, 9] = np.deg2rad(robot_omega_error_deg)

    # Fill columns with insufficient measurements from the default distribution.
    default_cov = default.covariance()
    default_mean = default.mean_array()
    usable_columns = []
    means = np.zeros(matrix.shape[1], dtype=float)
    standard = np.sqrt(np.diag(default_cov))
    for col in range(matrix.shape[1]):
        finite = np.isfinite(matrix[:, col])
        if np.count_nonzero(finite) >= 6:
            means[col] = float(np.mean(matrix[finite, col]))
            measured_std = float(np.std(matrix[finite, col], ddof=1))
            standard[col] = max(measured_std, 1e-9)
            usable_columns.append(col)
        else:
            matrix[:, col] = np.where(np.isfinite(matrix[:, col]), matrix[:, col], default_mean[col])

    covariance = default_cov.copy()
    if len(usable_columns) >= 2:
        complete_mask = np.all(np.isfinite(matrix[:, usable_columns]), axis=1)
        if np.count_nonzero(complete_mask) >= 6:
            measured = matrix[complete_mask][:, usable_columns]
            measured_cov = np.cov(measured, rowvar=False)
            if measured_cov.ndim == 0:
                measured_cov = np.array([[float(measured_cov)]])
            for i, col_i in enumerate(usable_columns):
                for j, col_j in enumerate(usable_columns):
                    covariance[col_i, col_j] = float(measured_cov[i, j])

    # Ensure positive semi-definite via eigenvalue clipping.
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-9, 1e-12)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T

    measured_rows = int(max((np.count_nonzero(np.isfinite(matrix[:, c])) for c in usable_columns), default=0))
    return ShotErrorDistribution(
        mean=tuple(float(v) for v in means),
        covariance_flat=tuple(float(v) for v in covariance.ravel()),
        dimension=len(SHOT_ERROR_NAMES),
        source_rows=measured_rows,
    )


# ===========================================================================
# OPTIONAL AUTOMATIC PHYSICS FITTING
# ===========================================================================


def _centered_row_mask(rows: CalibrationRows) -> np.ndarray:
    labels = rows.text_column("result")
    return np.asarray([label in CENTERED_RESULT_LABELS for label in labels], dtype=bool)


def fit_physics_from_logs(
    rows: CalibrationRows,
    calibration: ShooterCalibration,
    ball: BallModel,
    initial_aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    max_rows: int = 40,
) -> AeroModel:
    centered = _centered_row_mask(rows)
    distance = rows.column("distance_m")
    rpm = rows.column("rpm")
    angle = rows.column("angle_deg")
    entry = rows.column("entry_angle_deg")
    flight = rows.column("flight_time_s")
    lip_x = rows.column("lip_x_m")
    lip_y = rows.column("lip_y_m")

    mask = centered & np.isfinite(distance) & np.isfinite(rpm)
    indices = np.flatnonzero(mask)[:max_rows]
    if len(indices) < 4:
        warnings.warn("--auto-calibrate: fewer than 4 centered rows; skipping physics fit")
        return initial_aero

    robot = RobotState()

    def residuals(vector: np.ndarray) -> np.ndarray:
        cd_ref, cd_slope, magnus, tau, angle_bias = vector
        aero = replace(
            initial_aero,
            cd_reference=float(cd_ref),
            cd_log_re_slope=float(cd_slope),
            magnus_lift_slope=float(magnus),
            spin_decay_tau_s=float(tau),
            launch_angle_bias_deg=float(angle_bias),
        )
        residual: list[float] = []
        for index in indices:
            commanded_angle = float(angle[index]) if np.isfinite(angle[index]) else ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
            try:
                yaw = first_order_lead_yaw_deg(
                    float(distance[index]), float(rpm[index]), commanded_angle, calibration, aero, shooter, robot
                )
                reached, position, velocity, _, time_s, _, _, _ = simulate_to_lip(
                    float(distance[index]),
                    float(rpm[index]),
                    commanded_angle,
                    yaw,
                    calibration,
                    ball,
                    aero,
                    shooter,
                    robot,
                    hub,
                )
            except (ValueError, RuntimeError):
                residual.extend([8.0, 8.0])
                continue
            if not reached:
                residual.extend([8.0, 8.0])
                continue

            target_x = float(lip_x[index]) if np.isfinite(lip_x[index]) else 0.0
            target_y = float(lip_y[index]) if np.isfinite(lip_y[index]) else 0.0
            residual.append((float(position[0]) - target_x) / 0.05)
            residual.append((float(position[1]) - target_y) / 0.05)

            horizontal = max(float(np.hypot(velocity[0], velocity[1])), 1e-9)
            predicted_entry = math.degrees(math.atan2(-velocity[2], horizontal))
            if np.isfinite(entry[index]):
                residual.append((predicted_entry - float(entry[index])) / 3.0)
            if np.isfinite(flight[index]):
                residual.append((time_s - float(flight[index])) / 0.05)
        return np.asarray(residual, dtype=float)

    fit = least_squares(
        residuals,
        x0=np.array(
            [
                initial_aero.cd_reference,
                initial_aero.cd_log_re_slope,
                initial_aero.magnus_lift_slope,
                initial_aero.spin_decay_tau_s,
                initial_aero.launch_angle_bias_deg,
            ],
            dtype=float,
        ),
        bounds=(
            np.array([0.15, -0.15, 0.0, 0.20, -6.0]),
            np.array([1.10, 0.20, 0.45, 20.0, 6.0]),
        ),
        max_nfev=70,
        verbose=0,
    )
    cd_ref, cd_slope, magnus, tau, angle_bias = fit.x
    return replace(
        initial_aero,
        cd_reference=float(cd_ref),
        cd_log_re_slope=float(cd_slope),
        magnus_lift_slope=float(magnus),
        spin_decay_tau_s=float(tau),
        launch_angle_bias_deg=float(angle_bias),
    )


# ===========================================================================
# EMPIRICAL DISTANCE -> RPM RESIDUAL CORRECTION
# ===========================================================================


def nominal_center_rpm_for_angle(
    distance_m: float,
    angle_deg: float,
    rpm_bounds: tuple[float, float],
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
) -> float | None:
    low, high = rpm_bounds

    def x_error(rpm: float) -> float:
        yaw = first_order_lead_yaw_deg(distance_m, rpm, angle_deg, calibration, aero, shooter, robot)
        try:
            reached, position, *_ = simulate_to_lip(
                distance_m, rpm, angle_deg, yaw, calibration, ball, aero, shooter, robot, hub
            )
        except (ValueError, RuntimeError):
            return -5.0
        return float(position[0]) if reached else -5.0

    # Most geometries bracket the center root over the configured RPM bounds.
    # Try the cheap two-endpoint test first, then only scan if needed.
    f_low = x_error(low)
    f_high = x_error(high)
    if np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high <= 0.0:
        try:
            return float(brentq(x_error, low, high, xtol=1e-4))
        except (ValueError, RuntimeError):
            pass

    grid = np.linspace(low, high, 18)
    values = np.array([x_error(float(v)) for v in grid])
    for l, h, fl, fh in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True):
        if fl * fh <= 0.0:
            try:
                return float(brentq(x_error, float(l), float(h), xtol=1e-4))
            except (ValueError, RuntimeError):
                continue
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices):
        idx = finite_indices[np.argmin(np.abs(values[finite_indices]))]
        return float(grid[idx])
    return None


def fit_empirical_correction(
    rows: CalibrationRows,
    rpm_bounds: tuple[float, float],
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
) -> EmpiricalCorrection:
    distance = rows.column("distance_m")
    recommended = rows.column("recommended_rpm")
    rpm = rows.column("rpm")
    centered = _centered_row_mask(rows)

    empirical_distance: list[float] = []
    empirical_rpm: list[float] = []
    for i in range(rows.count):
        if np.isfinite(distance[i]) and np.isfinite(recommended[i]):
            empirical_distance.append(float(distance[i]))
            empirical_rpm.append(float(recommended[i]))
        elif np.isfinite(distance[i]) and np.isfinite(rpm[i]) and centered[i]:
            empirical_distance.append(float(distance[i]))
            empirical_rpm.append(float(rpm[i]))

    if len(empirical_distance) < 2:
        return EmpiricalCorrection()

    robot = RobotState()
    unique = np.unique(empirical_distance)
    residual_x: list[float] = []
    residual_y: list[float] = []
    for d in unique:
        observed = np.median([r for dd, r in zip(empirical_distance, empirical_rpm, strict=True) if math.isclose(dd, d, abs_tol=1e-9)])
        baseline = nominal_center_rpm_for_angle(
            float(d),
            ROBOT_NOMINAL_LAUNCH_ANGLE_DEG,
            rpm_bounds,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
        )
        if baseline is not None:
            residual_x.append(float(d))
            residual_y.append(float(observed - baseline))

    if len(residual_x) < 2:
        return EmpiricalCorrection()
    return EmpiricalCorrection(tuple(residual_x), tuple(residual_y))


# ===========================================================================
# NUMBA BATCH PHYSICS + HUB CONTACT
# ===========================================================================


@njit(cache=True, fastmath=True)
def _numba_hex_max_projection(x: float, y: float) -> tuple[float, int]:
    max_proj = -1e30
    max_index = 0
    for i in range(6):
        angle = i * math.pi / 3.0
        nx = math.cos(angle)
        ny = math.sin(angle)
        proj = nx * x + ny * y
        if proj > max_proj:
            max_proj = proj
            max_index = i
    return max_proj, max_index


@njit(cache=True, fastmath=True)
def _numba_accel_spin(
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
    mass: float,
    diameter: float,
    cd_ref: float,
    cd_slope: float,
    ref_re: float,
    magnus_slope: float,
    magnus_re_slope: float,
    max_cl: float,
    spin_tau: float,
    wind_x: float,
    wind_y: float,
    wind_z: float,
) -> tuple[float, float, float, float, float, float]:
    rx = vx - wind_x
    ry = vy - wind_y
    rz = vz - wind_z
    speed = math.sqrt(rx * rx + ry * ry + rz * rz)
    ax = 0.0
    ay = 0.0
    az = -G_MPS2
    if speed > 1e-9:
        reynolds = AIR_DENSITY_KG_M3 * speed * diameter / AIR_DYNAMIC_VISCOSITY_PA_S
        cd = cd_ref + cd_slope * math.log10(max(reynolds, 1.0) / ref_re)
        cd = min(1.2, max(0.15, cd))
        radius = 0.5 * diameter
        area = math.pi * radius * radius
        qam = 0.5 * AIR_DENSITY_KG_M3 * area / mass
        drag_factor = -qam * cd * speed
        ax += drag_factor * rx
        ay += drag_factor * ry
        az += drag_factor * rz

        spin_mag = math.sqrt(wx * wx + wy * wy + wz * wz)
        if spin_mag > 1e-9:
            spin_ratio = spin_mag * radius / speed
            re_factor = 1.0 + magnus_re_slope * math.log10(max(reynolds, 1.0) / ref_re)
            cl = magnus_slope * spin_ratio * re_factor
            cl = min(max_cl, max(-max_cl, cl))
            # omega x v
            lx = wy * rz - wz * ry
            ly = wz * rx - wx * rz
            lz = wx * ry - wy * rx
            lnorm = math.sqrt(lx * lx + ly * ly + lz * lz)
            if lnorm > 1e-12:
                scale = qam * cl * speed * speed / lnorm
                ax += scale * lx
                ay += scale * ly
                az += scale * lz

    tau = max(spin_tau, 0.05)
    return ax, ay, az, -wx / tau, -wy / tau, -wz / tau


@njit(cache=True, fastmath=True)
def _numba_simulate_one(
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
    mass: float,
    diameter: float,
    cd_ref: float,
    cd_slope: float,
    ref_re: float,
    magnus_slope: float,
    magnus_re_slope: float,
    max_cl: float,
    spin_tau: float,
    wind_x: float,
    wind_y: float,
    wind_z: float,
    opening_apothem: float,
    lip_height: float,
    rim_width: float,
    rim_thickness: float,
    funnel_depth: float,
    throat_apothem: float,
    restitution: float,
    friction: float,
    dt: float,
    max_time: float,
) -> tuple[int, int, float, float, float, float, float, float, float, float, int, int]:
    """Return (score, class_code, margin, lip x/y, lip vx/vy/vz, time, peak, rim, funnel)."""
    radius = 0.5 * diameter
    peak = z
    previous_z = z
    time_s = 0.0
    crossed_lip = False
    lip_x = math.nan
    lip_y = math.nan
    lip_vx = math.nan
    lip_vy = math.nan
    lip_vz = math.nan
    lip_time = math.nan
    lip_margin = -1e9
    rim_contacts = 0
    funnel_contacts = 0
    initially_clean = False

    funnel_bottom = lip_height - rim_thickness - funnel_depth
    rim_top = lip_height + 0.5 * rim_thickness
    rim_bottom = lip_height - 0.5 * rim_thickness

    steps = int(max_time / dt) + 1
    for _ in range(steps):
        ax, ay, az, dwx, dwy, dwz = _numba_accel_spin(
            vx, vy, vz, wx, wy, wz, mass, diameter, cd_ref, cd_slope, ref_re,
            magnus_slope, magnus_re_slope, max_cl, spin_tau, wind_x, wind_y, wind_z
        )
        # RK2-ish velocity midpoint for position; acceleration evaluated once keeps
        # JIT kernel compact and is accurate enough at 2-3 ms steps.
        vx_mid = vx + 0.5 * dt * ax
        vy_mid = vy + 0.5 * dt * ay
        vz_mid = vz + 0.5 * dt * az
        old_x = x
        old_y = y
        old_z = z
        x += dt * vx_mid
        y += dt * vy_mid
        z += dt * vz_mid
        vx += dt * ax
        vy += dt * ay
        vz += dt * az
        wx += dt * dwx
        wy += dt * dwy
        wz += dt * dwz
        time_s += dt
        if z > peak:
            peak = z

        if not crossed_lip and old_z >= lip_height and z < lip_height and vz < 0.0:
            denom = old_z - z
            frac = 0.0 if abs(denom) < 1e-12 else (old_z - lip_height) / denom
            lip_x = old_x + frac * (x - old_x)
            lip_y = old_y + frac * (y - old_y)
            lip_vx = vx
            lip_vy = vy
            lip_vz = vz
            lip_time = time_s - dt + frac * dt
            proj, _ = _numba_hex_max_projection(lip_x, lip_y)
            lip_margin = opening_apothem - proj - radius
            initially_clean = lip_margin >= 0.0
            crossed_lip = True

        if crossed_lip:
            proj, normal_index = _numba_hex_max_projection(x, y)
            angle = normal_index * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            inner = opening_apothem - radius
            outer = opening_apothem + rim_width + radius
            overlaps_z = (z - radius <= rim_top) and (z + radius >= rim_bottom)
            in_annulus = proj > inner and proj < outer
            if overlaps_z and in_annulus and vz < 0.0:
                rim_contacts += 1
                z = rim_top + radius + 1e-5
                vz = -restitution * vz
                vx *= max(0.0, 1.0 - friction)
                vy *= max(0.0, 1.0 - friction)
                if proj < opening_apothem + 0.5 * rim_width:
                    inward = 0.25 * abs(vz)
                    vx -= inward * nx
                    vy -= inward * ny

            # Funnel wall collision.
            if z <= lip_height and z >= funnel_bottom:
                top_z = lip_height - rim_thickness
                if z >= top_z:
                    allowed = opening_apothem - radius
                elif z <= funnel_bottom:
                    allowed = throat_apothem - radius
                else:
                    fraction = (top_z - z) / max(funnel_depth, 1e-9)
                    apothem = opening_apothem + fraction * (throat_apothem - opening_apothem)
                    allowed = apothem - radius
                proj, normal_index = _numba_hex_max_projection(x, y)
                if proj > allowed:
                    funnel_contacts += 1
                    angle = normal_index * math.pi / 3.0
                    nx = math.cos(angle)
                    ny = math.sin(angle)
                    penetration = proj - allowed
                    x -= penetration * nx
                    y -= penetration * ny
                    da_dz = (opening_apothem - throat_apothem) / max(funnel_depth, 1e-9)
                    onx = nx
                    ony = ny
                    onz = -da_dz
                    norm = math.sqrt(onx * onx + ony * ony + onz * onz)
                    onx /= norm
                    ony /= norm
                    onz /= norm
                    vn = vx * onx + vy * ony + vz * onz
                    if vn > 0.0:
                        tx = vx - vn * onx
                        ty = vy - vn * ony
                        tz = vz - vn * onz
                        vx = (1.0 - friction) * tx - restitution * vn * onx
                        vy = (1.0 - friction) * ty - restitution * vn * ony
                        vz = (1.0 - friction) * tz - restitution * vn * onz

            if z <= funnel_bottom - radius:
                proj, _ = _numba_hex_max_projection(x, y)
                if proj <= throat_apothem - 0.15 * radius and vz < 0.0:
                    if initially_clean and rim_contacts == 0 and funnel_contacts == 0:
                        class_code = 1  # clean
                    elif rim_contacts > 0 and funnel_contacts > 0:
                        class_code = 4  # rim+funnel
                    elif rim_contacts > 0:
                        class_code = 2  # rim
                    else:
                        class_code = 3  # funnel
                    return 1, class_code, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts
                return 0, 0, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts

            if z > lip_height + 0.35 and vz > 0.0:
                return 0, 0, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts
            proj, _ = _numba_hex_max_projection(x, y)
            if proj > opening_apothem + rim_width + 0.50:
                return 0, 0, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts

        if z <= radius and vz < 0.0:
            return 0, 0, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts

        previous_z = z

    return 0, 0, lip_margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim_contacts, funnel_contacts


@njit(cache=True, fastmath=True)
def _numba_batch_kernel(
    position0: np.ndarray,
    velocity0: np.ndarray,
    spin0: np.ndarray,
    mass: np.ndarray,
    diameter: np.ndarray,
    cd_ref: np.ndarray,
    cd_slope: np.ndarray,
    ref_re: float,
    magnus_slope: np.ndarray,
    magnus_re_slope: float,
    max_cl: float,
    spin_tau: np.ndarray,
    wind: np.ndarray,
    opening_apothem: float,
    lip_height: float,
    rim_width: float,
    rim_thickness: float,
    funnel_depth: float,
    throat_apothem: float,
    restitution: float,
    friction: float,
    dt: float,
    max_time: float,
):
    n = position0.shape[0]
    score = np.zeros(n, dtype=np.int8)
    class_code = np.zeros(n, dtype=np.int8)
    margin = np.full(n, -1e9)
    lip_x = np.full(n, np.nan)
    lip_y = np.full(n, np.nan)
    lip_vx = np.full(n, np.nan)
    lip_vy = np.full(n, np.nan)
    lip_vz = np.full(n, np.nan)
    lip_time = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    rim = np.zeros(n, dtype=np.int16)
    funnel = np.zeros(n, dtype=np.int16)
    for i in range(n):
        result = _numba_simulate_one(
            position0[i, 0], position0[i, 1], position0[i, 2],
            velocity0[i, 0], velocity0[i, 1], velocity0[i, 2],
            spin0[i, 0], spin0[i, 1], spin0[i, 2],
            mass[i], diameter[i], cd_ref[i], cd_slope[i], ref_re,
            magnus_slope[i], magnus_re_slope, max_cl, spin_tau[i],
            wind[i, 0], wind[i, 1], wind[i, 2],
            opening_apothem, lip_height, rim_width, rim_thickness, funnel_depth,
            throat_apothem, restitution, friction, dt, max_time,
        )
        score[i] = result[0]
        class_code[i] = result[1]
        margin[i] = result[2]
        lip_x[i] = result[3]
        lip_y[i] = result[4]
        lip_vx[i] = result[5]
        lip_vy[i] = result[6]
        lip_vz[i] = result[7]
        lip_time[i] = result[8]
        peak[i] = result[9]
        rim[i] = result[10]
        funnel[i] = result[11]
    return score, class_code, margin, lip_x, lip_y, lip_vx, lip_vy, lip_vz, lip_time, peak, rim, funnel


# ===========================================================================
# SOBOL / CORRELATED MONTE CARLO
# ===========================================================================


UncertaintyMode = Literal["combined", "model", "shot"]


def sobol_standard_normals(sample_count: int, dimension: int, seed: int) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    # random_base2 gives best Sobol balance; draw next power of two then trim.
    m = int(math.ceil(math.log2(sample_count)))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    uniform = sampler.random_base2(m)
    uniform = _safe_clip_probability(uniform[:sample_count])
    return ndtri(uniform)


def correlated_shot_errors(
    sample_count: int,
    distribution: ShotErrorDistribution,
    seed: int,
) -> np.ndarray:
    z = sobol_standard_normals(sample_count, distribution.dimension, seed)
    covariance = distribution.covariance()
    try:
        transform = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(covariance)
        transform = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-12)))
    return distribution.mean_array()[None, :] + z @ transform.T


def monte_carlo_candidates(
    distance_m: float,
    candidate_rpm: Sequence[float] | np.ndarray,
    candidate_angle_deg: Sequence[float] | np.ndarray,
    samples_per_candidate: int,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    mode: UncertaintyMode,
    seed: int,
    dt_s: float = 0.003,
) -> CandidateStatistics:
    rpms = np.asarray(candidate_rpm, dtype=float)
    angles = np.asarray(candidate_angle_deg, dtype=float)
    if rpms.shape != angles.shape:
        raise ValueError("candidate RPM and angle arrays must have equal shape")
    candidate_count = len(rpms)
    n = samples_per_candidate

    include_model = mode in ("combined", "model")
    include_shot = mode in ("combined", "shot")

    # Common random numbers across candidates reduce ranking noise.
    model_z = sobol_standard_normals(n, 6, seed + 11)
    shot_error = correlated_shot_errors(n, shot_distribution, seed + 29)
    if not include_shot:
        shot_error[:] = 0.0

    total = candidate_count * n
    rpm_command = np.repeat(rpms, n)
    angle_command = np.repeat(angles, n)

    def tile_col(array: np.ndarray, col: int) -> np.ndarray:
        return np.tile(array[:, col], candidate_count)

    mass = np.full(total, ball.mass_kg, dtype=float)
    cd_ref = np.full(total, aero.cd_reference, dtype=float)
    cd_slope = np.full(total, aero.cd_log_re_slope, dtype=float)
    magnus = np.full(total, aero.magnus_lift_slope, dtype=float)
    spin_tau = np.full(total, aero.spin_decay_tau_s, dtype=float)
    speed_scale = np.ones(total, dtype=float)
    if include_model:
        mass += uncertainty.mass_sigma_kg * tile_col(model_z, 0)
        mass = np.clip(mass, 0.195, 0.235)
        cd_ref += uncertainty.cd_reference_sigma * tile_col(model_z, 1)
        cd_ref = np.clip(cd_ref, 0.15, 1.1)
        cd_slope += uncertainty.cd_log_re_slope_sigma * tile_col(model_z, 2)
        magnus += uncertainty.magnus_slope_sigma * tile_col(model_z, 3)
        magnus = np.clip(magnus, 0.0, 0.45)
        spin_tau *= np.maximum(0.2, 1.0 + uncertainty.spin_decay_fraction_sigma * tile_col(model_z, 4))
        speed_scale *= np.maximum(0.5, 1.0 + uncertainty.calibration_speed_scale_sigma * tile_col(model_z, 5))

    error = np.tile(shot_error, (candidate_count, 1))
    actual_rpm = rpm_command + error[:, 0]
    speed_scale *= np.maximum(0.5, 1.0 + error[:, 1])
    actual_angle = angle_command + aero.launch_angle_bias_deg + error[:, 2]
    yaw_error = error[:, 3]
    actual_distance = distance_m + error[:, 4]
    lateral = error[:, 5]
    exit_height = shooter.exit_height_m + error[:, 6]
    robot_vx = robot.vx_mps + error[:, 7]
    robot_vy = robot.vy_mps + error[:, 8]
    robot_omega = robot.omega_rad_s + error[:, 9]

    launch_speed = calibration.speed_from_rpm(actual_rpm) * speed_scale
    launch_speed = np.maximum(0.2, launch_speed)
    spin_rpm = calibration.spin_from_rpm(actual_rpm)

    # Refined lead is too expensive per MC sample.  Solve exact lead once per
    # candidate at nominal conditions, then inject measured yaw error around it.
    # Candidate sweeps use the inexpensive kinematic lead.  The final selected
    # nominal shot is re-solved with the exact trajectory-root lead, so this avoids
    # dozens of nested solve_ivp calls without removing exact lead compensation
    # from the delivered setpoint/diagnostics.
    nominal_yaw = np.array(
        [
            first_order_lead_yaw_deg(
                distance_m, float(rpm), float(angle), calibration, aero, shooter, robot
            )
            for rpm, angle in zip(rpms, angles, strict=True)
        ],
        dtype=float,
    )
    yaw = np.repeat(nominal_yaw, n) + yaw_error

    position0 = np.column_stack(
        (
            -actual_distance + shooter.forward_offset_m,
            shooter.left_offset_m + lateral,
            exit_height,
        )
    )

    angle_rad = np.deg2rad(actual_angle)
    yaw_rad = np.deg2rad(yaw)
    horizontal = np.cos(angle_rad)
    units = np.column_stack(
        (horizontal * np.cos(yaw_rad), horizontal * np.sin(yaw_rad), np.sin(angle_rad))
    )

    rotational_vx = -robot_omega * shooter.left_offset_m
    rotational_vy = robot_omega * shooter.forward_offset_m
    robot_exit_v = np.column_stack(
        (robot_vx + rotational_vx, robot_vy + rotational_vy, np.zeros(total))
    )
    velocity0 = launch_speed[:, None] * units + robot_exit_v

    # Full 3-D spin vector.  Backspin axis follows each perturbed yaw; logs can
    # later provide explicit spin_xyz columns for a more specialized extension.
    spin_axis = np.column_stack((np.sin(yaw_rad), -np.cos(yaw_rad), np.zeros(total)))
    spin0 = spin_axis * (spin_rpm * 2.0 * math.pi / 60.0)[:, None]

    diameter = np.full(total, ball.diameter_m, dtype=float)
    wind = np.tile(
        np.array([aero.wind_x_mps, aero.wind_y_mps, aero.wind_z_mps], dtype=float),
        (total, 1),
    )

    result = _numba_batch_kernel(
        position0,
        velocity0,
        spin0,
        mass,
        diameter,
        cd_ref,
        cd_slope,
        aero.reference_reynolds,
        magnus,
        aero.magnus_re_slope,
        aero.max_cl,
        spin_tau,
        wind,
        hub.opening_apothem_m,
        hub.lip_height_m,
        hub.rim_radial_width_m,
        hub.rim_thickness_m,
        hub.funnel_depth_m,
        hub.throat_apothem_m,
        hub.restitution,
        hub.friction,
        dt_s,
        3.0,
    )
    score, class_code, margin = result[0], result[1], result[2]
    score = score.reshape(candidate_count, n)
    class_code = class_code.reshape(candidate_count, n)
    margin = margin.reshape(candidate_count, n)
    finite_margin = np.where(np.isfinite(margin), margin, -1.0)

    probability = np.mean(score, axis=1)
    clean_probability = np.mean(class_code == 1, axis=1)
    rim_probability = np.mean((class_code == 2) | (class_code == 4), axis=1)
    funnel_probability = np.mean((class_code == 3) | (class_code == 4), axis=1)
    return CandidateStatistics(
        rpm=rpms,
        angle_deg=angles,
        probability=probability,
        clean_probability=clean_probability,
        rim_probability=rim_probability,
        funnel_probability=funnel_probability,
        margin_p10_m=np.percentile(finite_margin, 10, axis=1),
        margin_p50_m=np.percentile(finite_margin, 50, axis=1),
        margin_p90_m=np.percentile(finite_margin, 90, axis=1),
    )


# ===========================================================================
# ADAPTIVE ROBUST OPTIMIZATION
# ===========================================================================


def _stats_subset_replace(base: CandidateStatistics, indices: np.ndarray, refined: CandidateStatistics) -> CandidateStatistics:
    result = CandidateStatistics(
        rpm=base.rpm.copy(),
        angle_deg=base.angle_deg.copy(),
        probability=base.probability.copy(),
        clean_probability=base.clean_probability.copy(),
        rim_probability=base.rim_probability.copy(),
        funnel_probability=base.funnel_probability.copy(),
        margin_p10_m=base.margin_p10_m.copy(),
        margin_p50_m=base.margin_p50_m.copy(),
        margin_p90_m=base.margin_p90_m.copy(),
    )
    for field in (
        "probability",
        "clean_probability",
        "rim_probability",
        "funnel_probability",
        "margin_p10_m",
        "margin_p50_m",
        "margin_p90_m",
    ):
        getattr(result, field)[indices] = getattr(refined, field)
    return result


def adaptive_candidate_statistics(
    distance_m: float,
    rpms: np.ndarray,
    angles: np.ndarray,
    base_samples: int,
    max_samples: int,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    mode: UncertaintyMode,
    seed: int,
    refinement_probability_window: float = 0.08,
) -> CandidateStatistics:
    base = monte_carlo_candidates(
        distance_m,
        rpms,
        angles,
        base_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        mode,
        seed,
    )
    if max_samples <= base_samples or len(rpms) <= 1:
        return base
    best = float(np.max(base.probability))
    competitive = np.flatnonzero(base.probability >= best - refinement_probability_window)
    # Keep refinement bounded if a broad plateau scores identically.
    if len(competitive) > 8:
        order = np.argsort(base.probability[competitive])[::-1][:8]
        competitive = competitive[order]
    if len(competitive) == 0:
        return base

    refined = monte_carlo_candidates(
        distance_m,
        rpms[competitive],
        angles[competitive],
        max_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        mode,
        seed + 100003,
    )
    return _stats_subset_replace(base, competitive, refined)


def choose_best_candidate(
    stats: CandidateStatistics,
    distance_m: float,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    preferred_rpm: float | None = None,
    desired_entry_angle_deg: float = 45.0,
) -> int:
    utility = 1000.0 * stats.probability
    utility += 45.0 * stats.clean_probability
    utility += 18.0 * np.clip(stats.margin_p10_m, -0.20, 0.20)
    utility -= 8.0 * stats.rim_probability

    for i, (rpm, angle) in enumerate(zip(stats.rpm, stats.angle_deg, strict=True)):
        try:
            nominal = simulate_shot(
                distance_m,
                float(rpm),
                float(angle),
                calibration,
                ball,
                aero,
                hub,
                shooter,
                robot,
                # Candidate tie-breaking uses first-order lead for speed. The final
                # selected shot is always recomputed with exact trajectory-root lead.
                exact_lead=False,
                save_trajectory=False,
            )
        except (ValueError, RuntimeError):
            utility[i] -= 1000.0
            continue
        if not nominal.reached_hub:
            utility[i] -= 1000.0
            continue
        utility[i] -= 0.05 * abs(nominal.entry_angle_deg - desired_entry_angle_deg)
        utility[i] -= 0.00035 * float(rpm)
        if preferred_rpm is not None:
            utility[i] -= 0.010 * abs(float(rpm) - preferred_rpm)
        if nominal.classification == "clean_score":
            utility[i] += 2.0
    return int(np.argmax(utility))


def optimize_shot_for_distance(
    distance_m: float,
    angle_candidates_deg: np.ndarray,
    rpm_bounds: tuple[float, float],
    calibration: ShooterCalibration,
    correction: EmpiricalCorrection,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    optimizer_samples: int,
    final_samples: int,
    adaptive_max_samples: int,
    probability_threshold: float,
    seed: int,
) -> OptimizedShot:
    correction_rpm = float(correction.correction_rpm(distance_m))

    # Preserve the trusted default anchor exactly when 70 degrees is an allowed
    # candidate and no measured exit-speed curve has replaced it.
    if (
        calibration.uses_fallback_anchor
        and abs(distance_m - CALIBRATION_DISTANCE_M) < 1e-9
        and np.any(np.isclose(angle_candidates_deg, ROBOT_NOMINAL_LAUNCH_ANGLE_DEG))
    ):
        best_angle = ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
        best_rpm = CALIBRATION_RPM
        final_stats = monte_carlo_candidates(
            distance_m,
            np.array([best_rpm]),
            np.array([best_angle]),
            max(final_samples, adaptive_max_samples),
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
            uncertainty,
            shot_distribution,
            "combined",
            seed,
        )
        model_stats = monte_carlo_candidates(
            distance_m,
            np.array([best_rpm]),
            np.array([best_angle]),
            final_samples,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
            uncertainty,
            shot_distribution,
            "model",
            seed + 2,
        )
        shot_stats = monte_carlo_candidates(
            distance_m,
            np.array([best_rpm]),
            np.array([best_angle]),
            final_samples,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
            uncertainty,
            shot_distribution,
            "shot",
            seed + 3,
        )
        nominal = simulate_shot(
            distance_m,
            best_rpm,
            best_angle,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
            exact_lead=True,
            save_trajectory=True,
        )
        p = float(final_stats.probability[0])
        return OptimizedShot(
            distance_m=distance_m,
            rpm=best_rpm,
            angle_deg=best_angle,
            learned_correction_rpm=correction_rpm,
            combined_probability=p,
            model_only_probability=float(model_stats.probability[0]),
            shot_only_probability=float(shot_stats.probability[0]),
            clean_probability=float(final_stats.clean_probability[0]),
            contact_probability=float(final_stats.rim_probability[0] + final_stats.funnel_probability[0]),
            probability_band_low_rpm=best_rpm if p >= probability_threshold else float("nan"),
            probability_band_high_rpm=best_rpm if p >= probability_threshold else float("nan"),
            nominal=nominal,
        )

    center_rpms: list[float] = []
    valid_angles: list[float] = []
    for angle in angle_candidates_deg:
        center = nominal_center_rpm_for_angle(
            distance_m,
            float(angle),
            rpm_bounds,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
        )
        if center is not None and np.isfinite(center):
            center_rpms.append(float(np.clip(center + correction_rpm, *rpm_bounds)))
            valid_angles.append(float(angle))
    if not center_rpms:
        raise ValueError(f"No valid descending shot at {distance_m:.2f} m")

    angle_stats = adaptive_candidate_statistics(
        distance_m,
        np.asarray(center_rpms),
        np.asarray(valid_angles),
        optimizer_samples,
        adaptive_max_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        "combined",
        seed,
    )
    angle_index = choose_best_candidate(
        angle_stats,
        distance_m,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
    )
    best_angle = float(angle_stats.angle_deg[angle_index])
    preferred_rpm = float(angle_stats.rpm[angle_index])

    step = ROUND_TO_RPM
    low = max(rpm_bounds[0], preferred_rpm - 250.0)
    high = min(rpm_bounds[1], preferred_rpm + 250.0)
    rpms = np.arange(math.floor(low / step) * step, math.ceil(high / step) * step + 0.1, step)
    rpms = rpms[(rpms >= rpm_bounds[0]) & (rpms <= rpm_bounds[1])]
    angles = np.full_like(rpms, best_angle)

    rpm_stats = adaptive_candidate_statistics(
        distance_m,
        rpms,
        angles,
        final_samples,
        adaptive_max_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        "combined",
        seed + 101,
    )
    rpm_index = choose_best_candidate(
        rpm_stats,
        distance_m,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        preferred_rpm=preferred_rpm,
    )
    best_rpm = float(rpm_stats.rpm[rpm_index])

    acceptable = rpm_stats.probability >= probability_threshold
    if np.any(acceptable):
        band_low = float(np.min(rpm_stats.rpm[acceptable]))
        band_high = float(np.max(rpm_stats.rpm[acceptable]))
    else:
        band_low = band_high = float("nan")

    model_stats = monte_carlo_candidates(
        distance_m,
        np.array([best_rpm]),
        np.array([best_angle]),
        final_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        "model",
        seed + 201,
    )
    shot_stats = monte_carlo_candidates(
        distance_m,
        np.array([best_rpm]),
        np.array([best_angle]),
        final_samples,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        "shot",
        seed + 301,
    )
    nominal = simulate_shot(
        distance_m,
        best_rpm,
        best_angle,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        exact_lead=True,
        save_trajectory=True,
    )
    return OptimizedShot(
        distance_m=distance_m,
        rpm=best_rpm,
        angle_deg=best_angle,
        learned_correction_rpm=correction_rpm,
        combined_probability=float(rpm_stats.probability[rpm_index]),
        model_only_probability=float(model_stats.probability[0]),
        shot_only_probability=float(shot_stats.probability[0]),
        clean_probability=float(rpm_stats.clean_probability[rpm_index]),
        contact_probability=float(rpm_stats.rim_probability[rpm_index] + rpm_stats.funnel_probability[rpm_index]),
        probability_band_low_rpm=band_low,
        probability_band_high_rpm=band_high,
        nominal=nominal,
    )


# ===========================================================================
# PARALLEL WORKER PAYLOADS
# ===========================================================================


@dataclass(frozen=True)
class OptimizeWorkerPayload:
    distance_m: float
    angle_candidates_deg: tuple[float, ...]
    rpm_bounds: tuple[float, float]
    calibration: ShooterCalibration
    correction: EmpiricalCorrection
    ball: BallModel
    aero: AeroModel
    hub: HubGeometry
    shooter: ShooterGeometry
    robot: RobotState
    uncertainty: UncertaintyModel
    shot_distribution: ShotErrorDistribution
    optimizer_samples: int
    final_samples: int
    adaptive_max_samples: int
    probability_threshold: float
    seed: int


def _optimize_worker(payload: OptimizeWorkerPayload) -> OptimizedShot:
    return optimize_shot_for_distance(
        payload.distance_m,
        np.asarray(payload.angle_candidates_deg, dtype=float),
        payload.rpm_bounds,
        payload.calibration,
        payload.correction,
        payload.ball,
        payload.aero,
        payload.hub,
        payload.shooter,
        payload.robot,
        payload.uncertainty,
        payload.shot_distribution,
        payload.optimizer_samples,
        payload.final_samples,
        payload.adaptive_max_samples,
        payload.probability_threshold,
        payload.seed,
    )


@dataclass(frozen=True)
class HeatmapWorkerPayload:
    row_index: int
    distance_m: float
    angle_deg: float
    rpm_grid: tuple[float, ...]
    sample_count: int
    calibration: ShooterCalibration
    ball: BallModel
    aero: AeroModel
    hub: HubGeometry
    shooter: ShooterGeometry
    robot: RobotState
    uncertainty: UncertaintyModel
    shot_distribution: ShotErrorDistribution
    seed: int


def _heatmap_worker(payload: HeatmapWorkerPayload) -> tuple[int, np.ndarray]:
    rpm = np.asarray(payload.rpm_grid, dtype=float)
    stats = monte_carlo_candidates(
        payload.distance_m,
        rpm,
        np.full_like(rpm, payload.angle_deg),
        payload.sample_count,
        payload.calibration,
        payload.ball,
        payload.aero,
        payload.hub,
        payload.shooter,
        payload.robot,
        payload.uncertainty,
        payload.shot_distribution,
        "combined",
        payload.seed,
    )
    return payload.row_index, stats.probability


def _executor_class(kind: str):
    return ThreadPoolExecutor if kind == "thread" else ProcessPoolExecutor


def optimize_distances_parallel(
    distances: np.ndarray,
    angle_candidates: np.ndarray,
    rpm_bounds: tuple[float, float],
    calibration: ShooterCalibration,
    correction: EmpiricalCorrection,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    optimizer_samples: int,
    final_samples: int,
    adaptive_max_samples: int,
    probability_threshold: float,
    workers: int,
    executor_kind: str,
    seed: int,
    show_progress: bool,
) -> list[OptimizedShot]:
    payloads = [
        OptimizeWorkerPayload(
            distance_m=float(distance),
            angle_candidates_deg=tuple(float(v) for v in angle_candidates),
            rpm_bounds=rpm_bounds,
            calibration=calibration,
            correction=correction,
            ball=ball,
            aero=aero,
            hub=hub,
            shooter=shooter,
            robot=robot,
            uncertainty=uncertainty,
            shot_distribution=shot_distribution,
            optimizer_samples=optimizer_samples,
            final_samples=final_samples,
            adaptive_max_samples=adaptive_max_samples,
            probability_threshold=probability_threshold,
            seed=seed + 1009 * index,
        )
        for index, distance in enumerate(distances)
    ]

    if workers <= 1:
        iterator: Iterable[OptimizeWorkerPayload] = payloads
        if show_progress:
            iterator = tqdm(payloads, desc="Optimizing shots", unit="distance")
        results = [_optimize_worker(payload) for payload in iterator]
        return sorted(results, key=lambda shot: shot.distance_m)

    executor_cls = _executor_class(executor_kind)
    results: list[OptimizedShot] = []
    with executor_cls(max_workers=workers) as executor:
        future_map = {executor.submit(_optimize_worker, payload): payload.distance_m for payload in payloads}
        iterator = as_completed(future_map)
        if show_progress:
            iterator = tqdm(iterator, total=len(payloads), desc="Optimizing shots", unit="distance")
        for future in iterator:
            results.append(future.result())
    return sorted(results, key=lambda shot: shot.distance_m)


# ===========================================================================
# OUTPUTS
# ===========================================================================


def write_lookup_csv(output_directory: Path, shots: Sequence[OptimizedShot]) -> Path:
    path = output_directory / "shooter_velocity_estimates.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "robot_center_distance_m",
                "optimal_launch_angle_deg",
                "optimal_flywheel_rpm",
                "rounded_flywheel_rpm",
                "learned_correction_rpm",
                "combined_score_probability",
                "model_only_score_probability",
                "shot_only_score_probability",
                "clean_score_probability",
                "contact_score_probability",
                "probability_band_low_rpm",
                "probability_band_high_rpm",
                "launch_speed_mps",
                "lead_yaw_deg",
                "flight_time_s",
                "entry_angle_deg_down",
                "impact_speed_mps",
                "hub_opening_margin_mm",
                "peak_height_m",
                "nominal_classification",
                "nominal_rim_contacts",
                "nominal_funnel_contacts",
                "spin_x_rpm_at_launch",
                "spin_y_rpm_at_launch",
                "spin_z_rpm_at_launch",
            ]
        )
        for shot in shots:
            r = shot.nominal
            writer.writerow(
                [
                    f"{shot.distance_m:.3f}",
                    f"{shot.angle_deg:.4f}",
                    f"{shot.rpm:.3f}",
                    f"{float(_round_rpm(shot.rpm)):.3f}",
                    f"{shot.learned_correction_rpm:.3f}",
                    f"{shot.combined_probability:.6f}",
                    f"{shot.model_only_probability:.6f}",
                    f"{shot.shot_only_probability:.6f}",
                    f"{shot.clean_probability:.6f}",
                    f"{shot.contact_probability:.6f}",
                    f"{shot.probability_band_low_rpm:.3f}",
                    f"{shot.probability_band_high_rpm:.3f}",
                    f"{r.launch_speed_mps:.6f}",
                    f"{r.yaw_deg:.5f}",
                    f"{r.flight_time_s:.6f}",
                    f"{r.entry_angle_deg:.5f}",
                    f"{r.impact_speed_mps:.6f}",
                    f"{1000.0 * r.opening_margin_m:.3f}",
                    f"{r.peak_height_m:.6f}",
                    r.classification,
                    r.rim_contacts,
                    r.funnel_contacts,
                    f"{r.initial_spin_rpm_xyz[0]:.3f}",
                    f"{r.initial_spin_rpm_xyz[1]:.3f}",
                    f"{r.initial_spin_rpm_xyz[2]:.3f}",
                ]
            )
    return path


def _monotonic_dense_curve(x: np.ndarray, y: np.ndarray, dense_x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.full_like(dense_x, y[0], dtype=float)
    # PCHIP preserves shape and avoids polynomial ringing.
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    return np.asarray(interpolator(dense_x), dtype=float)


def write_dense_lookup(
    output_directory: Path,
    shots: Sequence[OptimizedShot],
    step_m: float,
) -> Path:
    distances = np.asarray([shot.distance_m for shot in shots], dtype=float)
    rpms = np.asarray([shot.rpm for shot in shots], dtype=float)
    angles = np.asarray([shot.angle_deg for shot in shots], dtype=float)
    probabilities = np.asarray([shot.combined_probability for shot in shots], dtype=float)
    dense = np.arange(distances[0], distances[-1] + 0.5 * step_m, step_m)
    dense_rpm = _monotonic_dense_curve(distances, rpms, dense)
    dense_angle = _monotonic_dense_curve(distances, angles, dense)
    dense_probability = _monotonic_dense_curve(distances, probabilities, dense)

    path = output_directory / "shooter_lookup_dense.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["distance_m", "flywheel_rpm", "rounded_rpm", "hood_angle_deg", "interpolated_score_probability"])
        for d, rpm, angle, probability in zip(dense, dense_rpm, dense_angle, dense_probability, strict=True):
            writer.writerow(
                [
                    f"{d:.3f}",
                    f"{rpm:.3f}",
                    f"{float(_round_rpm(rpm)):.1f}",
                    f"{angle:.4f}",
                    f"{float(np.clip(probability, 0.0, 1.0)):.6f}",
                ]
            )
    return path


def write_rpm_plot(output_directory: Path, shots: Sequence[OptimizedShot]) -> Path:
    distance = np.asarray([s.distance_m for s in shots])
    rpm = np.asarray([s.rpm for s in shots])
    low = np.asarray([s.probability_band_low_rpm for s in shots])
    high = np.asarray([s.probability_band_high_rpm for s in shots])
    probability = np.asarray([s.combined_probability for s in shots])

    figure, axis = plt.subplots(figsize=(9.7, 5.8), layout="constrained")
    finite = np.isfinite(low) & np.isfinite(high)
    if np.any(finite):
        axis.fill_between(distance[finite], low[finite], high[finite], alpha=0.20, label="Robust probability band")
    axis.plot(distance, rpm, "o-", label="Optimized setpoint")
    axis.plot([CALIBRATION_DISTANCE_M], [CALIBRATION_RPM], "s", markersize=8, label="Trusted fallback anchor")
    axis.set(
        title="2026 FUEL Shooter Robust Setpoint Model",
        xlabel="Robot-center distance to HUB center (m)",
        ylabel="Flywheel setpoint (RPM)",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    twin = axis.twinx()
    twin.plot(distance, 100.0 * probability, "--", alpha=0.45, label="P(score)")
    twin.set_ylabel("Predicted score probability (%)")
    twin.set_ylim(0.0, 105.0)
    path = output_directory / "shooter_velocity_model.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_trajectory_plot(output_directory: Path, shots: Sequence[OptimizedShot], hub: HubGeometry) -> Path:
    figure, axis = plt.subplots(figsize=(10.2, 6.0), layout="constrained")
    for shot in shots:
        r = shot.nominal
        if r.states is None:
            continue
        start_x = r.states[0, 0]
        range_x = r.states[:, 0] - start_x
        axis.plot(range_x, r.states[:, 2], label=f"{shot.distance_m:.1f} m / {shot.rpm:.0f} RPM / {shot.angle_deg:.1f}°")
    axis.axhline(hub.lip_height_m, linestyle="--", linewidth=1.0, label="HUB lip")
    axis.axhline(hub.funnel_bottom_z_m, linestyle=":", linewidth=1.0, label="Funnel bottom")
    axis.set(
        title="Nominal Optimized Trajectories",
        xlabel="Forward travel from shooter exit (m)",
        ylabel="Ball-center height (m)",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    path = output_directory / "shooter_trajectories.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_probability_heatmap_parallel(
    output_directory: Path,
    distances: np.ndarray,
    rpm_grid: np.ndarray,
    angle_by_distance: np.ndarray,
    sample_count: int,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    workers: int,
    executor_kind: str,
    seed: int,
    show_progress: bool,
) -> np.ndarray:
    payloads = [
        HeatmapWorkerPayload(
            row_index=i,
            distance_m=float(distance),
            angle_deg=float(angle_by_distance[i]),
            rpm_grid=tuple(float(v) for v in rpm_grid),
            sample_count=sample_count,
            calibration=calibration,
            ball=ball,
            aero=aero,
            hub=hub,
            shooter=shooter,
            robot=robot,
            uncertainty=uncertainty,
            shot_distribution=shot_distribution,
            seed=seed + 4001 * i,
        )
        for i, distance in enumerate(distances)
    ]

    probability = np.zeros((len(distances), len(rpm_grid)), dtype=float)
    if workers <= 1:
        iterator: Iterable[HeatmapWorkerPayload] = payloads
        if show_progress:
            iterator = tqdm(payloads, desc="Heatmap", unit="row")
        for payload in iterator:
            row, values = _heatmap_worker(payload)
            probability[row] = values
    else:
        executor_cls = _executor_class(executor_kind)
        with executor_cls(max_workers=workers) as executor:
            futures = [executor.submit(_heatmap_worker, payload) for payload in payloads]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Heatmap", unit="row")
            for future in iterator:
                row, values = future.result()
                probability[row] = values

    csv_path = output_directory / "score_probability_heatmap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["distance_m"] + [f"rpm_{rpm:.0f}" for rpm in rpm_grid])
        for d, row in zip(distances, probability, strict=True):
            writer.writerow([f"{d:.3f}"] + [f"{p:.6f}" for p in row])

    figure, axis = plt.subplots(figsize=(11.0, 6.0), layout="constrained")
    y_low = float(distances[0] - 0.25) if len(distances) == 1 else float(distances[0])
    y_high = float(distances[0] + 0.25) if len(distances) == 1 else float(distances[-1])
    image = axis.imshow(
        probability,
        origin="lower",
        aspect="auto",
        extent=[float(rpm_grid[0]), float(rpm_grid[-1]), y_low, y_high],
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set(
        title="Collision-aware Monte Carlo P(score)",
        xlabel="Flywheel setpoint (RPM)",
        ylabel="Robot-center distance to HUB center (m)",
    )
    figure.colorbar(image, ax=axis, label="P(score)")
    figure.savefig(output_directory / "score_probability_heatmap.png", dpi=180)
    plt.close(figure)
    return probability


def write_calibration_plot(output_directory: Path, rows: CalibrationRows, calibration: ShooterCalibration) -> Path | None:
    rpm = rows.column("rpm")
    speed = rows.column("exit_speed_mps")
    mask = np.isfinite(rpm) & np.isfinite(speed)
    if np.count_nonzero(mask) < 2:
        return None
    x = np.linspace(float(np.min(rpm[mask])), float(np.max(rpm[mask])), 300)
    y = calibration.speed_from_rpm(x)
    figure, axis = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    axis.scatter(rpm[mask], speed[mask], label="Measured")
    axis.plot(x, y, label="Monotonic PCHIP")
    axis.set(title="RPM to Ball Exit-Speed Calibration", xlabel="Flywheel RPM", ylabel="Ball exit speed (m/s)")
    axis.grid(alpha=0.3)
    axis.legend()
    path = output_directory / "exit_speed_calibration.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_shot_log_template(output_directory: Path) -> Path:
    path = output_directory / "shot_log_template.csv"
    headers = [
        "distance_m",
        "rpm",
        "angle_deg",
        "exit_speed_mps",
        "spin_rpm",
        "result",
        "recommended_rpm",
        "actual_rpm",
        "actual_angle_deg",
        "yaw_error_deg",
        "entry_angle_deg",
        "flight_time_s",
        "lip_x_m",
        "lip_y_m",
        "distance_error_m",
        "lateral_error_m",
        "exit_height_error_m",
        "robot_vx_error_mps",
        "robot_vy_error_mps",
        "robot_omega_error_deg_s",
    ]
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
    return path


def write_calibration_report(
    output_directory: Path,
    calibration: ShooterCalibration,
    aero: AeroModel,
    shot_distribution: ShotErrorDistribution,
    correction: EmpiricalCorrection,
) -> Path:
    path = output_directory / "calibration_report.txt"
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"Calibration source: {calibration.source}\n")
        output_file.write(f"1400 flywheel RPM speed: {float(calibration.speed_from_rpm(1400.0)):.6f} m/s\n")
        output_file.write(f"Spin source: {calibration.spin_curve.source if calibration.spin_curve else 'fallback spin ratio'}\n")
        output_file.write(f"Cd reference: {aero.cd_reference:.6f}\n")
        output_file.write(f"Cd log-Re slope: {aero.cd_log_re_slope:.6f}\n")
        output_file.write(f"Magnus slope: {aero.magnus_lift_slope:.6f}\n")
        output_file.write(f"Spin decay tau: {aero.spin_decay_tau_s:.6f} s\n")
        output_file.write(f"Launch-angle bias: {aero.launch_angle_bias_deg:+.6f} deg\n")
        output_file.write(f"Shot covariance learned rows: {shot_distribution.source_rows}\n")
        output_file.write(f"Empirical correction points: {len(correction.x)}\n")
        output_file.write("\nShot-error covariance order:\n")
        for index, name in enumerate(SHOT_ERROR_NAMES):
            output_file.write(f"  {index}: {name}\n")
        output_file.write("\nCovariance matrix:\n")
        np.savetxt(output_file, shot_distribution.covariance(), fmt="% .8e")
    return path


def print_java_entries(shots: Sequence[OptimizedShot]) -> None:
    print("\nJava interpolation-map entries:")
    print("// Flywheel RPM")
    for shot in shots:
        print(f"FLYWHEEL_RPM_BY_DISTANCE.put({shot.distance_m:.2f}, {float(_round_rpm(shot.rpm)):.1f});")
    print("\n// Hood angle")
    for shot in shots:
        print(f"HOOD_ANGLE_DEG_BY_DISTANCE.put({shot.distance_m:.2f}, {shot.angle_deg:.3f});")



# ===========================================================================
# CLI
# ===========================================================================


def parse_distances(value: str) -> np.ndarray:
    value = value.strip()
    if ":" in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("distance range must be start:stop:step")
        start, stop, step = parts
        if step <= 0.0 or stop < start:
            raise argparse.ArgumentTypeError("invalid distance range")
        return np.arange(start, stop + 0.5 * step, step)
    try:
        values = np.array([float(part) for part in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(values) == 0 or np.any(values <= 0.0):
        raise argparse.ArgumentTypeError("distances must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=128, help="Base final MC samples per candidate")
    parser.add_argument("--optimizer-samples", type=int, default=64, help="Base MC samples while selecting angle")
    parser.add_argument("--adaptive-max-samples", type=int, default=512, help="Sobol samples for competitive candidates")
    parser.add_argument("--heatmap-samples", type=int, default=64, help="Samples per heatmap cell")
    parser.add_argument("--probability-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--executor", choices=("process", "thread"), default="thread", help="thread is usually fastest with the Numba kernel; process is available for isolation")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--auto-calibrate", action="store_true", help="Fit physics and shot covariance from supplied logs")
    parser.add_argument("--distances", type=parse_distances, default=DISTANCES_M, help='"1:6:0.5" or comma list')
    parser.add_argument("--rpm-min", type=float, default=800.0)
    parser.add_argument("--rpm-max", type=float, default=3000.0)
    parser.add_argument("--heatmap-rpm-step", type=float, default=50.0)
    parser.add_argument("--dense-step", type=float, default=DENSE_LOOKUP_STEP_M)

    parser.add_argument("--optimize-angle", action="store_true")
    parser.add_argument("--fixed-angle", type=float, default=None)
    parser.add_argument("--angle-min", type=float, default=55.0)
    parser.add_argument("--angle-max", type=float, default=75.0)
    parser.add_argument("--angle-step", type=float, default=2.5)

    parser.add_argument("--exit-height", type=float, default=ROBOT_SHOOTER_EXIT_HEIGHT_M)
    parser.add_argument("--shooter-forward-offset", type=float, default=ROBOT_SHOOTER_FORWARD_OFFSET_M)
    parser.add_argument("--shooter-left-offset", type=float, default=ROBOT_SHOOTER_LEFT_OFFSET_M)
    parser.add_argument("--robot-vx", type=float, default=0.0)
    parser.add_argument("--robot-vy", type=float, default=0.0)
    parser.add_argument("--robot-omega-deg-s", type=float, default=0.0)

    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--wind-y", type=float, default=0.0)
    parser.add_argument("--wind-z", type=float, default=0.0)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples < 16 or args.optimizer_samples < 16 or args.heatmap_samples < 8:
        parser.error("Monte Carlo sample counts are too small")
    if args.adaptive_max_samples < args.samples:
        parser.error("--adaptive-max-samples must be >= --samples")
    if args.rpm_max <= args.rpm_min:
        parser.error("--rpm-max must be greater than --rpm-min")
    if not (0.0 < args.probability_threshold <= 1.0):
        parser.error("--probability-threshold must be in (0, 1]")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.optimize_angle and args.fixed_angle is not None:
        parser.error("use either --optimize-angle or --fixed-angle")
    if args.dense_step <= 0.0 or args.heatmap_rpm_step <= 0.0:
        parser.error("lookup/heatmap steps must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if not NUMBA_AVAILABLE:
        warnings.warn("Numba is not installed; batch Monte Carlo will be much slower. pip install numba")

    output_directory: Path = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    ball = BallModel()
    hub = HubGeometry()
    shooter = ShooterGeometry(
        exit_height_m=float(args.exit_height),
        forward_offset_m=float(args.shooter_forward_offset),
        left_offset_m=float(args.shooter_left_offset),
    )
    robot = RobotState(
        vx_mps=float(args.robot_vx),
        vy_mps=float(args.robot_vy),
        omega_rad_s=math.radians(float(args.robot_omega_deg_s)),
    )
    aero = AeroModel(
        wind_x_mps=float(args.wind_x),
        wind_y_mps=float(args.wind_y),
        wind_z_mps=float(args.wind_z),
    )
    uncertainty = UncertaintyModel()
    rows = load_calibration_rows(args.calibration_csv)
    calibration = build_calibration(rows, ball, aero, hub, shooter)

    if args.auto_calibrate and rows.count > 0:
        aero = fit_physics_from_logs(rows, calibration, ball, aero, hub, shooter)
        # Keep fallback anchor exact after the physics fit if there was no measured
        # speed curve; measured calibration deliberately remains authoritative.
        if calibration.speed_curve is None:
            calibration = build_calibration(rows, ball, aero, hub, shooter)

    shot_distribution = estimate_shot_error_distribution(rows, calibration, uncertainty)
    rpm_bounds = (float(args.rpm_min), float(args.rpm_max))
    correction = fit_empirical_correction(rows, rpm_bounds, calibration, ball, aero, hub, shooter)

    if args.fixed_angle is not None:
        angle_candidates = np.array([float(args.fixed_angle)])
    elif args.optimize_angle:
        if args.angle_step <= 0.0 or args.angle_max < args.angle_min:
            parser.error("invalid angle range")
        angle_candidates = np.arange(args.angle_min, args.angle_max + 0.5 * args.angle_step, args.angle_step)
    else:
        angle_candidates = np.array([ROBOT_NOMINAL_LAUNCH_ANGLE_DEG])

    distances = np.asarray(args.distances, dtype=float)

    # Warm the Numba cache in the parent so fork/spawn workers are less surprising.
    if NUMBA_AVAILABLE:
        dummy = np.array([[0.0, 0.0, shooter.exit_height_m]])
        dummy_v = np.array([[1.0, 0.0, 3.0]])
        dummy_s = np.array([[0.0, -10.0, 0.0]])
        _numba_batch_kernel(
            dummy,
            dummy_v,
            dummy_s,
            np.array([ball.mass_kg]),
            np.array([ball.diameter_m]),
            np.array([aero.cd_reference]),
            np.array([aero.cd_log_re_slope]),
            aero.reference_reynolds,
            np.array([aero.magnus_lift_slope]),
            aero.magnus_re_slope,
            aero.max_cl,
            np.array([aero.spin_decay_tau_s]),
            np.zeros((1, 3)),
            hub.opening_apothem_m,
            hub.lip_height_m,
            hub.rim_radial_width_m,
            hub.rim_thickness_m,
            hub.funnel_depth_m,
            hub.throat_apothem_m,
            hub.restitution,
            hub.friction,
            0.003,
            0.01,
        )

    shots = optimize_distances_parallel(
        distances,
        angle_candidates,
        rpm_bounds,
        calibration,
        correction,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        int(args.optimizer_samples),
        int(args.samples),
        int(args.adaptive_max_samples),
        float(args.probability_threshold),
        int(args.workers),
        str(args.executor),
        int(args.seed),
        not bool(args.no_progress),
    )

    for shot in shots:
        print(
            f"{shot.distance_m:4.2f} m -> {shot.rpm:6.0f} RPM @ {shot.angle_deg:5.1f} deg | "
            f"P(score)={shot.combined_probability:6.1%} | clean={shot.clean_probability:6.1%} | "
            f"entry={shot.nominal.entry_angle_deg:5.1f} deg | {shot.nominal.classification}"
        )

    write_lookup_csv(output_directory, shots)
    write_dense_lookup(output_directory, shots, float(args.dense_step))
    write_rpm_plot(output_directory, shots)
    write_trajectory_plot(output_directory, shots, hub)
    write_shot_log_template(output_directory)
    write_calibration_plot(output_directory, rows, calibration)
    write_calibration_report(output_directory, calibration, aero, shot_distribution, correction)

    rpm_grid = np.arange(
        math.ceil(rpm_bounds[0] / args.heatmap_rpm_step) * args.heatmap_rpm_step,
        rpm_bounds[1] + 0.1,
        args.heatmap_rpm_step,
    )
    write_probability_heatmap_parallel(
        output_directory,
        distances,
        rpm_grid,
        np.asarray([shot.angle_deg for shot in shots], dtype=float),
        int(args.heatmap_samples),
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        uncertainty,
        shot_distribution,
        int(args.workers),
        str(args.executor),
        int(args.seed) + 900001,
        not bool(args.no_progress),
    )


    print_java_entries(shots)
    print("\nModel summary:")
    print("  RPM definition: measured flywheel / launcher-roller RPM")
    print(f"  Shooter geometry: z={shooter.exit_height_m:.3f} m, x-offset={shooter.forward_offset_m:.3f} m, y-offset={shooter.left_offset_m:.3f} m")
    print(f"  Nominal launch angle: {ROBOT_NOMINAL_LAUNCH_ANGLE_DEG:.1f} deg")
    print(f"  Calibration: {calibration.source}")
    print(f"  1400 RPM -> {float(calibration.speed_from_rpm(CALIBRATION_RPM)):.3f} m/s")
    print(f"  Cd(reference): {aero.cd_reference:.4f}")
    print(f"  Cd log-Re slope: {aero.cd_log_re_slope:+.4f}")
    print(f"  Magnus slope: {aero.magnus_lift_slope:.4f}")
    print(f"  Spin decay tau: {aero.spin_decay_tau_s:.3f} s")
    print(f"  Launch-angle bias: {aero.launch_angle_bias_deg:+.3f} deg")
    print(f"  Learned shot-covariance rows: {shot_distribution.source_rows}")
    print(f"  Empirical RPM correction points: {len(correction.x)}")
    print(f"  Numba JIT: {'enabled' if NUMBA_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Outputs: {output_directory}")


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows.
    main()
