#!/usr/bin/env python3
"""Advanced 2026 FRC FUEL shooter model, calibration, validation, and 3-D viewer.

This is an offline design/calibration model, not a replacement for on-robot
closed-loop control. RPM ALWAYS means the measured flywheel/launcher-roller RPM.
The robot PID is responsible for reaching that RPM; motor count, motor torque,
current limits, and gearing are intentionally not part of this ballistic model.

Major features
--------------
* 3-D time-domain projectile dynamics with Reynolds-dependent drag and Magnus lift.
* Full 3-D spin with decay, spin-axis error, and spin-aware rigid contact impulses.
* Official 2026 FUEL diameter/mass envelope and official 41.7in / 72in HUB opening.
* Robot-center -> shooter-exit geometry and per-sample moving-shot lead compensation.
* Accurate adaptive ``solve_ivp`` nominal trajectories plus RK4/JIT Monte Carlo.
* RPM-conditioned measured exit-speed/spin calibration and scatter estimation.
* Robust regularized aerodynamic fitting that never converts missing lip data to zero.
* Pairwise/shrunk shot covariance that avoids RPM/exit-speed double counting.
* Sobol quasi-Monte-Carlo uncertainty including FUEL size/mass and spin variation.
* Genuine 2-D RPM/angle robust optimization, followed by high-sample local refinement.
* Dense lookup re-simulation at the actual rounded RPM command (probability is not interpolated).
* Separate held-out validation CSV with trajectory MAE, Brier score, and log loss.
* Interactive 3-D replay, including the post-lip rim/funnel path and contacts.
* Multiprocessing/threading, Numba JIT, tqdm progress, plots, heatmaps, and Java entries.

Fallback calibration
--------------------
If no measured ``rpm,exit_speed_mps`` calibration is supplied, the speed scale is
anchored so the trusted 2.00 m / 1400 RPM / 70 degree shot crosses the HUB center
while descending. The default fixed-70-degree lookup remains pinned to exactly
1400 RPM at 2.00 m until measured speed calibration replaces that fallback.

Recommended training-log columns
--------------------------------
Only columns you can actually measure are required. Useful columns include::

    timestamp,session_id,ball_id,distance_m,rpm,actual_rpm
    angle_deg,actual_angle_deg,exit_speed_mps,spin_rpm
    result,scored,recommended_rpm,entry_angle_deg,flight_time_s
    lip_x_m,lip_y_m,distance_error_m,lateral_error_m,exit_height_error_m
    robot_vx_error_mps,robot_vy_error_mps,robot_omega_error_deg_s
    rpm_before_shot,rpm_min_during_shot,rpm_after_shot,time_since_previous_shot_s

``rpm`` is the requested/measured launcher-roller setpoint, not motor percent.
Use a completely separate ``--validation-csv`` when you want honest held-out metrics.

Examples
--------
Fixed 70 degree model::

    python shooter_model_2026_everybot_pid_v2.py --workers 8

Watch each completed distance shot in 3-D while workers continue calculating::

    python shooter_model_2026_everybot_pid_v2.py --workers 8 --live-3d

Measured calibration + robust physics fit + held-out validation::

    python shooter_model_2026_everybot_pid_v2.py --calibration-csv train.csv --validation-csv validation.csv --auto-calibrate --workers 8

Adjustable hood with true 2-D RPM/angle optimization::

    python shooter_model_2026_everybot_pid_v2.py --optimize-angle --angle-step 2.5 --workers 8 --show-3d
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
from typing import Callable, Iterable, Literal, Sequence

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
FUEL_DIAMETER_M = 5.91 * 0.0254  # official nominal diameter
FUEL_MASS_MIN_KG = 0.448 * 0.45359237
FUEL_MASS_MAX_KG = 0.500 * 0.45359237
FUEL_MASS_KG = 0.5 * (FUEL_MASS_MIN_KG + FUEL_MASS_MAX_KG)

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
        enforce_increasing: bool = True,
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
        if enforce_increasing and np.any(np.diff(unique_y) <= 0.0):
            # Isotonic-lite for physical RPM->speed/spin curves only. Scatter
            # curves may legitimately rise and fall with RPM.
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
    speed_sigma_fraction_curve: PiecewiseCubicCalibration | None = None
    spin_sigma_fraction_curve: PiecewiseCubicCalibration | None = None
    fallback_speed_sigma_fraction: float = 0.018
    fallback_spin_sigma_fraction: float = 0.10

    def speed_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        if self.speed_curve is not None:
            return self.speed_curve.evaluate(rpm)
        return np.asarray(rpm, dtype=float) * self.fallback_speed_per_rpm

    def spin_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        if self.spin_curve is not None:
            return self.spin_curve.evaluate(rpm)
        return np.asarray(rpm, dtype=float) * self.fallback_spin_per_rpm

    def speed_sigma_fraction_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        values = np.asarray(rpm, dtype=float)
        if self.speed_sigma_fraction_curve is not None:
            return np.maximum(1e-4, self.speed_sigma_fraction_curve.evaluate(values))
        return np.full_like(values, self.fallback_speed_sigma_fraction, dtype=float)

    def spin_sigma_fraction_from_rpm(self, rpm: np.ndarray | float) -> np.ndarray:
        values = np.asarray(rpm, dtype=float)
        if self.spin_sigma_fraction_curve is not None:
            return np.maximum(1e-4, self.spin_sigma_fraction_curve.evaluate(values))
        return np.full_like(values, self.fallback_spin_sigma_fraction, dtype=float)

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
    x: tuple[float, ...] = ()          # distance
    y: tuple[float, ...] = ()          # RPM residual
    angle_deg: tuple[float, ...] = ()  # angle associated with each point
    max_abs_correction_rpm: float = 400.0

    def correction_rpm(
        self,
        distance_m: np.ndarray | float,
        angle_deg: np.ndarray | float | None = None,
    ) -> np.ndarray:
        distance = np.asarray(distance_m, dtype=float)
        if len(self.x) < 2:
            return np.zeros_like(distance)

        # If calibration spans multiple angles, use smooth inverse-distance
        # weighting in (distance, angle) space. Otherwise retain the stable 1-D
        # PCHIP correction used by the original fixed-70-degree model.
        if len(self.angle_deg) == len(self.x) and len(np.unique(self.angle_deg)) >= 2 and angle_deg is not None:
            angle = np.asarray(angle_deg, dtype=float)
            d_b, a_b = np.broadcast_arrays(distance, angle)
            out = np.empty_like(d_b, dtype=float)
            pts_d = np.asarray(self.x, dtype=float)
            pts_a = np.asarray(self.angle_deg, dtype=float)
            values = np.asarray(self.y, dtype=float)
            for index in np.ndindex(d_b.shape):
                dd = (pts_d - float(d_b[index])) / 0.50
                aa = (pts_a - float(a_b[index])) / 3.0
                r2 = dd * dd + aa * aa
                exact = np.flatnonzero(r2 < 1e-12)
                if len(exact):
                    value = float(np.median(values[exact]))
                else:
                    # Four nearest points prevent distant calibration regions from
                    # dominating while remaining smooth across sparse angle grids.
                    nearest = np.argsort(r2)[: min(4, len(r2))]
                    weights = 1.0 / np.maximum(r2[nearest], 1e-6)
                    value = float(np.sum(weights * values[nearest]) / np.sum(weights))
                out[index] = value
            return np.clip(out, -self.max_abs_correction_rpm, self.max_abs_correction_rpm)

        pchip = PchipInterpolator(np.asarray(self.x), np.asarray(self.y), extrapolate=True)
        corrected = np.asarray(pchip(distance), dtype=float)
        return np.clip(corrected, -self.max_abs_correction_rpm, self.max_abs_correction_rpm)


@dataclass(frozen=True)
class UncertaintyModel:
    # Model uncertainty. The official FUEL mass range is sampled directly in the
    # Monte Carlo; these values represent remaining model uncertainty.
    mass_sigma_kg: float = 0.004
    diameter_sigma_m: float = 0.0020
    cd_reference_sigma: float = 0.030
    cd_log_re_slope_sigma: float = 0.010
    magnus_slope_sigma: float = 0.020
    spin_decay_fraction_sigma: float = 0.15
    calibration_speed_scale_sigma: float = 0.008
    spin_scale_sigma: float = 0.06
    spin_axis_tilt_sigma_deg: float = 2.0

    # Default independent shot noise. Real covariance overrides these where there
    # is enough data. Exit-speed scatter is additionally conditioned on RPM using
    # repeated chronograph/high-speed-video measurements when available.
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
    # Optional empirical P(score | lip margin, entry angle). These coefficients
    # complement the rigid contact model when enough real scored/missed shots exist.
    score_model_coefficients: tuple[float, ...] = ()
    score_model_feature_mean: tuple[float, ...] = ()
    score_model_feature_scale: tuple[float, ...] = ()
    score_model_rows: int = 0

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

    def empirical_score_probability(
        self, margin_m: np.ndarray, entry_angle_deg: np.ndarray
    ) -> np.ndarray | None:
        if len(self.score_model_coefficients) != 3:
            return None
        feature = np.column_stack((np.asarray(margin_m).ravel(), np.asarray(entry_angle_deg).ravel()))
        mean = np.asarray(self.score_model_feature_mean, dtype=float)
        scale = np.asarray(self.score_model_feature_scale, dtype=float)
        if mean.shape != (2,) or scale.shape != (2,):
            return None
        z = (feature - mean) / np.maximum(scale, 1e-9)
        beta = np.asarray(self.score_model_coefficients, dtype=float)
        logits = beta[0] + z @ beta[1:]
        logits = np.clip(logits, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        return probability.reshape(np.broadcast_shapes(np.asarray(margin_m).shape, np.asarray(entry_angle_deg).shape))


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
    full_path_m: np.ndarray | None = None
    full_path_t_s: np.ndarray | None = None


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


def _fit_fraction_scatter_curve(
    x: np.ndarray,
    y: np.ndarray,
    mean_curve: PiecewiseCubicCalibration | None,
    source_name: str,
) -> PiecewiseCubicCalibration | None:
    """Fit an RPM-conditioned fractional 1-sigma scatter curve.

    Repeated samples at the same commanded RPM are preferred. Groups with fewer
    than 3 observations are ignored because two-shot ranges are not stable sigma
    estimates. The curve is intentionally smoothed by medians and PCHIP.
    """
    if mean_curve is None:
        return None
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    sigma_x: list[float] = []
    sigma_y: list[float] = []
    for rpm_value in np.unique(x):
        mask = np.isclose(x, rpm_value, atol=1e-9)
        values = y[mask]
        if len(values) < 3:
            continue
        center = float(mean_curve.evaluate(float(rpm_value)))
        if abs(center) < 1e-9:
            continue
        # MAD is robust to an occasional bad video/chronograph measurement.
        residual = values - center
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigma = max(1e-4, 1.4826 * mad / abs(center))
        sigma_x.append(float(rpm_value))
        sigma_y.append(sigma)
    if len(sigma_x) < 2:
        return None
    return PiecewiseCubicCalibration.from_points(
        sigma_x,
        sigma_y,
        source=f"RPM-conditioned fractional scatter from {source_name}",
        extrapolate=False,
        enforce_increasing=False,
    )


def fit_measured_calibration(
    rows: CalibrationRows,
) -> tuple[
    PiecewiseCubicCalibration | None,
    PiecewiseCubicCalibration | None,
    PiecewiseCubicCalibration | None,
    PiecewiseCubicCalibration | None,
]:
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

    speed_sigma_curve = _fit_fraction_scatter_curve(speed_rpm, speed_values, speed_curve, "exit-speed repeats")
    spin_sigma_curve = _fit_fraction_scatter_curve(spin_rpm, spin_values, spin_curve, "spin repeats")
    return speed_curve, spin_curve, speed_sigma_curve, spin_sigma_curve


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


def _apply_sphere_contact_impulse(
    velocity: np.ndarray,
    spin_rad_s_xyz: np.ndarray,
    normal_into_free_space: np.ndarray,
    mass_kg: float,
    radius_m: float,
    restitution: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Impulse response for a solid sphere, including spin/friction coupling."""
    n = np.asarray(normal_into_free_space, dtype=float)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    v = np.asarray(velocity, dtype=float).copy()
    w = np.asarray(spin_rad_s_xyz, dtype=float).copy()
    r_contact = -radius_m * n
    v_contact = v + np.cross(w, r_contact)
    vn = float(np.dot(v_contact, n))
    if vn >= 0.0:
        return v, w

    jn_mag = -(1.0 + restitution) * mass_kg * vn
    jn = jn_mag * n
    vt = v_contact - vn * n
    vt_mag = float(np.linalg.norm(vt))
    jt = np.zeros(3, dtype=float)
    if vt_mag > 1e-12 and friction > 0.0:
        # Solid sphere: I = 2/5 mr^2, so tangential effective mass = 2m/7.
        tangential_effective_mass = (2.0 / 7.0) * mass_kg
        desired = -tangential_effective_mass * vt
        desired_mag = float(np.linalg.norm(desired))
        limit = max(0.0, friction * jn_mag)
        if desired_mag > limit and desired_mag > 1e-12:
            desired *= limit / desired_mag
        jt = desired

    impulse = jn + jt
    v += impulse / mass_kg
    inertia = (2.0 / 5.0) * mass_kg * radius_m * radius_m
    if inertia > 1e-12:
        w += np.cross(r_contact, jt) / inertia
    return v, w


def _rk4_free_flight_step(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    dt_s: float,
    ball: BallModel,
    aero: AeroModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.concatenate((position, velocity, spin))

    def deriv(st: np.ndarray) -> np.ndarray:
        acc, spin_dot = acceleration_and_spin_derivative(st[3:6], st[6:9], ball, aero)
        return np.concatenate((st[3:6], acc, spin_dot))

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt_s * k1)
    k3 = deriv(state + 0.5 * dt_s * k2)
    k4 = deriv(state + dt_s * k3)
    nxt = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return nxt[:3], nxt[3:6], nxt[6:9]


def _resolve_funnel_contact(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    ball: BallModel,
    hub: HubGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    z = float(position[2])
    if z > hub.lip_height_m or z < hub.funnel_bottom_z_m:
        return position, velocity, spin, False

    allowed = _funnel_apothem_at_z(z, hub) - ball.radius_m
    projection, normal_xy = _max_hex_projection(float(position[0]), float(position[1]))
    penetration = projection - allowed
    if penetration <= 0.0:
        return position, velocity, spin, False

    da_dz = (hub.opening_apothem_m - hub.throat_apothem_m) / max(hub.funnel_depth_m, 1e-9)
    outward = np.array([normal_xy[0], normal_xy[1], -da_dz], dtype=float)
    outward /= max(float(np.linalg.norm(outward)), 1e-12)
    normal_free = -outward

    # Conservative projection back toward the valid volume.
    position = position - penetration * np.array([normal_xy[0], normal_xy[1], 0.0])
    velocity, spin = _apply_sphere_contact_impulse(
        velocity,
        spin,
        normal_free,
        ball.mass_kg,
        ball.radius_m,
        hub.restitution,
        hub.friction,
    )
    return position, velocity, spin, True


def simulate_hub_contact_after_lip(
    position_at_lip: np.ndarray,
    velocity_at_lip: np.ndarray,
    spin_rad_s_xyz: np.ndarray,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    dt_s: float = 0.0005,
    max_time_s: float = 0.65,
    save_trace: bool = False,
) -> tuple[bool, str, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Continue a descending shot through the HUB with spin-aware contacts.

    The top opening dimensions are official; the internal funnel remains a
    parameterized approximation unless the team replaces those dimensions with CAD
    measurements. Contact response uses a rigid solid-sphere impulse with Coulomb
    friction and angular impulse rather than the old velocity-damping heuristic.
    """
    position = np.asarray(position_at_lip, dtype=float).copy()
    velocity = np.asarray(velocity_at_lip, dtype=float).copy()
    spin = np.asarray(spin_rad_s_xyz, dtype=float).copy()
    rim_contacts = 0
    funnel_contacts = 0

    clean_margin = float(hub.signed_opening_margin_m(position[0], position[1], ball.radius_m))
    initially_clean = clean_margin >= 0.0
    trace_p: list[np.ndarray] = [position.copy()] if save_trace else []
    trace_t: list[float] = [0.0] if save_trace else []

    t = 0.0
    while t < max_time_s:
        step = min(dt_s, max_time_s - t)
        position, velocity, spin = _rk4_free_flight_step(position, velocity, spin, step, ball, aero)
        t += step

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
            position[2] = rim_top + ball.radius_m + 1e-6
            velocity, spin = _apply_sphere_contact_impulse(
                velocity,
                spin,
                np.array([0.0, 0.0, 1.0]),
                ball.mass_kg,
                ball.radius_m,
                hub.restitution,
                hub.friction,
            )
            # Near the inner edge, the contact normal is not perfectly horizontal;
            # add a small geometric inward component rather than a magic velocity
            # damping term. This remains an approximation until exact rim CAD is used.
            if projection < hub.opening_apothem_m + 0.35 * hub.rim_radial_width_m:
                side_normal = np.array([-normal_xy[0], -normal_xy[1], 0.35], dtype=float)
                velocity, spin = _apply_sphere_contact_impulse(
                    velocity, spin, side_normal, ball.mass_kg, ball.radius_m,
                    0.5 * hub.restitution, 0.7 * hub.friction,
                )

        position, velocity, spin, contacted = _resolve_funnel_contact(position, velocity, spin, ball, hub)
        if contacted:
            funnel_contacts += 1

        if save_trace:
            trace_p.append(position.copy())
            trace_t.append(t)

        if position[2] <= hub.funnel_bottom_z_m - ball.radius_m:
            projection, _ = _max_hex_projection(float(position[0]), float(position[1]))
            if projection <= hub.throat_apothem_m - 0.10 * ball.radius_m and velocity[2] < 0.0:
                if rim_contacts == 0 and funnel_contacts == 0 and initially_clean:
                    classification = "clean_score"
                elif rim_contacts > 0 and funnel_contacts > 0:
                    classification = "rim_funnel_score"
                elif rim_contacts > 0:
                    classification = "rim_score"
                else:
                    classification = "funnel_score"
                return (
                    True, classification, rim_contacts, funnel_contacts, position, velocity, spin,
                    np.asarray(trace_t) if save_trace else None,
                    np.asarray(trace_p) if save_trace else None,
                )
            return (
                False, "miss", rim_contacts, funnel_contacts, position, velocity, spin,
                np.asarray(trace_t) if save_trace else None,
                np.asarray(trace_p) if save_trace else None,
            )

        max_projection, _ = _max_hex_projection(float(position[0]), float(position[1]))
        if position[2] > hub.lip_height_m + 0.35 and velocity[2] > 0.0:
            break
        if max_projection > hub.opening_apothem_m + hub.rim_radial_width_m + 0.50:
            break

    return (
        False, "miss", rim_contacts, funnel_contacts, position, velocity, spin,
        np.asarray(trace_t) if save_trace else None,
        np.asarray(trace_p) if save_trace else None,
    )


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
        path = states[:, :3].copy() if save_trajectory and states is not None else None
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
            full_path_m=path,
            full_path_t_s=t_s.copy() if save_trajectory and t_s is not None else None,
        )

    margin = float(hub.signed_opening_margin_m(position[0], position[1], ball.radius_m))
    horizontal = float(np.hypot(velocity[0], velocity[1]))
    entry = math.degrees(math.atan2(-velocity[2], max(horizontal, 1e-12)))
    impact = float(np.linalg.norm(velocity))

    (
        scored,
        classification,
        rim_contacts,
        funnel_contacts,
        _,
        _,
        final_spin,
        contact_t,
        contact_path,
    ) = simulate_hub_contact_after_lip(
        position,
        velocity,
        spin_rad,
        ball,
        aero,
        hub,
        save_trace=save_trajectory,
    )

    full_path = None
    full_time = None
    if save_trajectory:
        pre_path = states[:, :3] if states is not None else np.asarray([shooter_exit_position(distance_m, shooter), position])
        pre_time = t_s if t_s is not None else np.linspace(0.0, flight_time, len(pre_path))
        if contact_path is not None and len(contact_path) > 1:
            full_path = np.vstack((pre_path, contact_path[1:]))
            full_time = np.concatenate((pre_time, flight_time + contact_t[1:]))
        else:
            full_path = np.asarray(pre_path, dtype=float)
            full_time = np.asarray(pre_time, dtype=float)

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
        final_spin_rpm_xyz=final_spin * 60.0 / (2.0 * math.pi),
        yaw_deg=float(yaw_deg),
        rim_contacts=rim_contacts,
        funnel_contacts=funnel_contacts,
        t_s=t_s,
        states=states,
        full_path_m=full_path,
        full_path_t_s=full_time,
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
    speed_curve, spin_curve, speed_sigma_curve, spin_sigma_curve = fit_measured_calibration(rows)
    fallback = fallback_calibration(ball, aero, hub, shooter)
    if speed_curve is None:
        return replace(
            fallback,
            spin_curve=spin_curve,
            speed_sigma_fraction_curve=speed_sigma_curve,
            spin_sigma_fraction_curve=spin_sigma_curve,
        )
    return ShooterCalibration(
        speed_curve=speed_curve,
        spin_curve=spin_curve,
        fallback_speed_per_rpm=fallback.fallback_speed_per_rpm,
        fallback_spin_per_rpm=fallback.fallback_spin_per_rpm,
        source=speed_curve.source,
        speed_sigma_fraction_curve=speed_sigma_curve,
        spin_sigma_fraction_curve=spin_sigma_curve,
    )


# ===========================================================================
# REAL-SHOT COVARIANCE LEARNING
# ===========================================================================


def estimate_shot_error_distribution(
    rows: CalibrationRows,
    calibration: ShooterCalibration,
    uncertainty: UncertaintyModel,
    hub: HubGeometry,
    ball: BallModel,
) -> ShotErrorDistribution:
    """Learn shot noise without double-counting RPM-induced exit-speed changes.

    Covariances are estimated pairwise so one missing sensor column does not throw
    away an otherwise useful shot. A data-dependent shrinkage toward the diagonal
    prevents small datasets from producing extreme/unstable correlations.
    """
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

    # IMPORTANT: when actual RPM is measured, compare measured ball speed against
    # speed_from_rpm(actual_rpm), not the commanded RPM. Otherwise an RPM sag is
    # counted once as rpm_error and a second time as exit-speed error.
    speed_reference_rpm = np.where(np.isfinite(actual_rpm), actual_rpm, commanded_rpm)
    predicted_speed = calibration.speed_from_rpm(speed_reference_rpm)
    valid_prediction = np.isfinite(speed) & np.isfinite(predicted_speed) & (predicted_speed > 1e-6)
    matrix[valid_prediction, 1] = (
        speed[valid_prediction] - predicted_speed[valid_prediction]
    ) / predicted_speed[valid_prediction]
    matrix[:, 2] = actual_angle - command_angle
    matrix[:, 3] = yaw_error
    matrix[:, 4] = distance_error
    matrix[:, 5] = lateral_error
    matrix[:, 6] = height_error
    matrix[:, 7] = robot_vx_error
    matrix[:, 8] = robot_vy_error
    matrix[:, 9] = np.deg2rad(robot_omega_error_deg)

    default_cov = default.covariance()
    means = default.mean_array().copy()
    covariance = default_cov.copy()
    finite_counts = np.zeros(matrix.shape[1], dtype=int)

    for col in range(matrix.shape[1]):
        finite = np.isfinite(matrix[:, col])
        n = int(np.count_nonzero(finite))
        finite_counts[col] = n
        if n >= 6:
            values = matrix[finite, col]
            means[col] = float(np.mean(values))
            measured_var = float(np.var(values, ddof=1))
            # Shrink small samples toward the conservative default variance.
            weight = n / (n + 12.0)
            covariance[col, col] = weight * measured_var + (1.0 - weight) * default_cov[col, col]

    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            finite = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            n = int(np.count_nonzero(finite))
            if n < 8:
                covariance[i, j] = covariance[j, i] = 0.0
                continue
            vi = matrix[finite, i]
            vj = matrix[finite, j]
            measured_cov = float(np.cov(vi, vj, ddof=1)[0, 1])
            # More overlap => less shrinkage of correlation.
            weight = n / (n + 20.0)
            value = weight * measured_cov
            covariance[i, j] = covariance[j, i] = value

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-12)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T

    # Optional empirical score/contact model. It learns only from real rows that
    # have measured lip x/y, entry angle, and an unambiguous score/miss label.
    # This is deliberately low-dimensional to avoid overfitting small FRC datasets.
    score_beta: tuple[float, ...] = ()
    score_mean: tuple[float, ...] = ()
    score_scale: tuple[float, ...] = ()
    score_rows = 0
    lip_x = rows.column("lip_x_m")
    lip_y = rows.column("lip_y_m")
    entry_angle = rows.column("entry_angle_deg")
    explicit_score = rows.column("scored")
    result_labels = rows.text_column("result")
    labels = np.full(rows.count, np.nan, dtype=float)
    for idx in range(rows.count):
        if np.isfinite(explicit_score[idx]):
            labels[idx] = 1.0 if explicit_score[idx] >= 0.5 else 0.0
        else:
            label = result_labels[idx]
            if any(token in label for token in ("miss", "fail", "no_score", "noscore")):
                labels[idx] = 0.0
            elif any(token in label for token in ("score", "center", "good", "bullseye", "made")):
                labels[idx] = 1.0

    score_mask = (
        np.isfinite(lip_x) & np.isfinite(lip_y) & np.isfinite(entry_angle) & np.isfinite(labels)
    )
    if np.count_nonzero(score_mask) >= 30:
        y_score = labels[score_mask]
        positives = int(np.count_nonzero(y_score >= 0.5))
        negatives = int(np.count_nonzero(y_score < 0.5))
        if positives >= 5 and negatives >= 5:
            margin = hub.signed_opening_margin_m(
                lip_x[score_mask], lip_y[score_mask], ball.radius_m
            )
            Xraw = np.column_stack((margin, entry_angle[score_mask]))
            mean_feature = np.mean(Xraw, axis=0)
            scale_feature = np.std(Xraw, axis=0, ddof=1)
            scale_feature = np.maximum(scale_feature, np.array([0.015, 1.0]))
            Z = (Xraw - mean_feature) / scale_feature
            X = np.column_stack((np.ones(len(Z)), Z))
            beta = np.zeros(3, dtype=float)
            ridge = np.diag([0.0, 0.35, 0.35])
            for _ in range(30):
                logits = np.clip(X @ beta, -25.0, 25.0)
                probability = 1.0 / (1.0 + np.exp(-logits))
                weights = np.maximum(probability * (1.0 - probability), 1e-5)
                gradient = X.T @ (probability - y_score) + ridge @ beta
                hessian = (X.T * weights) @ X + ridge
                try:
                    step = np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    break
                beta -= step
                if float(np.linalg.norm(step)) < 1e-7:
                    break
            score_beta = tuple(float(v) for v in beta)
            score_mean = tuple(float(v) for v in mean_feature)
            score_scale = tuple(float(v) for v in scale_feature)
            score_rows = int(len(y_score))

    return ShotErrorDistribution(
        mean=tuple(float(v) for v in means),
        covariance_flat=tuple(float(v) for v in covariance.ravel()),
        dimension=len(SHOT_ERROR_NAMES),
        source_rows=int(np.max(finite_counts)),
        score_model_coefficients=score_beta,
        score_model_feature_mean=score_mean,
        score_model_feature_scale=score_scale,
        score_model_rows=score_rows,
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
    max_rows: int = 120,
) -> AeroModel:
    """Fast robust/regularized aerodynamic fit from measured trajectory data.

    Missing lip coordinates are never interpreted as zero. The fit batches all
    calibration trajectories through the same RK4/JIT physics kernel used by robust
    optimization, making auto-calibration practical instead of running thousands of
    nested ``solve_ivp`` calls. The final delivered nominal shots still use the
    tighter adaptive ``solve_ivp`` solver.
    """
    distance_all = rows.column("distance_m")
    rpm_all = rows.column("rpm")
    angle_all = rows.column("angle_deg")
    entry_all = rows.column("entry_angle_deg")
    flight_all = rows.column("flight_time_s")
    lip_x_all = rows.column("lip_x_m")
    lip_y_all = rows.column("lip_y_m")

    informative = (
        np.isfinite(lip_x_all) | np.isfinite(lip_y_all)
        | np.isfinite(entry_all) | np.isfinite(flight_all)
    )
    mask = np.isfinite(distance_all) & np.isfinite(rpm_all) & informative
    indices = np.flatnonzero(mask)[:max_rows]
    if len(indices) < 8:
        warnings.warn(
            "--auto-calibrate: fewer than 8 rows contain measured lip position, entry angle, or flight time; "
            "skipping aerodynamic fit rather than fitting physics to categorical score labels"
        )
        return initial_aero

    distance = distance_all[indices]
    rpm = rpm_all[indices]
    angle = np.where(
        np.isfinite(angle_all[indices]), angle_all[indices], ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
    )
    lip_x = lip_x_all[indices]
    lip_y = lip_y_all[indices]
    entry = entry_all[indices]
    flight = flight_all[indices]
    n = len(indices)

    # Stationary calibration shots: yaw is geometric aim from shooter exit to HUB.
    p0 = np.column_stack(
        (
            -distance + shooter.forward_offset_m,
            np.full(n, shooter.left_offset_m),
            np.full(n, shooter.exit_height_m),
        )
    )
    target_xy = -p0[:, :2]
    yaw = np.arctan2(target_xy[:, 1], target_xy[:, 0])
    launch_speed = np.asarray(calibration.speed_from_rpm(rpm), dtype=float)
    spin_rpm = np.asarray(calibration.spin_from_rpm(rpm), dtype=float)

    measurement_counts = (
        np.isfinite(lip_x).astype(int) + np.isfinite(lip_y).astype(int)
        + np.isfinite(entry).astype(int) + np.isfinite(flight).astype(int)
    )

    def residuals(vector: np.ndarray) -> np.ndarray:
        cd_ref, cd_slope, magnus, log_tau, angle_bias = vector
        tau = math.exp(float(log_tau))
        effective_angle = np.deg2rad(angle + float(angle_bias))
        horizontal = np.cos(effective_angle)
        units = np.column_stack(
            (horizontal * np.cos(yaw), horizontal * np.sin(yaw), np.sin(effective_angle))
        )
        velocity0 = launch_speed[:, None] * units
        spin_axis = np.column_stack((np.sin(yaw), -np.cos(yaw), np.zeros(n)))
        spin0 = spin_axis * (spin_rpm * 2.0 * math.pi / 60.0)[:, None]

        result = _numba_batch_kernel(
            p0,
            velocity0,
            spin0,
            np.full(n, ball.mass_kg),
            np.full(n, ball.diameter_m),
            np.full(n, float(cd_ref)),
            np.full(n, float(cd_slope)),
            initial_aero.reference_reynolds,
            np.full(n, float(magnus)),
            initial_aero.magnus_re_slope,
            initial_aero.max_cl,
            np.full(n, tau),
            np.tile(
                np.array([initial_aero.wind_x_mps, initial_aero.wind_y_mps, initial_aero.wind_z_mps]),
                (n, 1),
            ),
            hub.opening_apothem_m,
            hub.lip_height_m,
            hub.rim_radial_width_m,
            hub.rim_thickness_m,
            hub.funnel_depth_m,
            hub.throat_apothem_m,
            hub.restitution,
            hub.friction,
            0.0025,
            3.0,
        )
        px, py = result[3], result[4]
        vx, vy, vz = result[5], result[6], result[7]
        time_s = result[8]
        reached = np.isfinite(time_s)

        residual: list[float] = []
        for local in range(n):
            if not reached[local]:
                residual.extend([6.0] * int(measurement_counts[local]))
                continue
            if np.isfinite(lip_x[local]):
                residual.append((float(px[local]) - float(lip_x[local])) / 0.04)
            if np.isfinite(lip_y[local]):
                residual.append((float(py[local]) - float(lip_y[local])) / 0.04)
            if np.isfinite(entry[local]):
                h = max(float(math.hypot(vx[local], vy[local])), 1e-9)
                predicted_entry = math.degrees(math.atan2(-float(vz[local]), h))
                residual.append((predicted_entry - float(entry[local])) / 2.0)
            if np.isfinite(flight[local]):
                residual.append((float(time_s[local]) - float(flight[local])) / 0.035)

        # Weak priors stop correlated aerodynamic parameters from becoming an
        # overfit explanation for measurement noise.
        residual.extend(
            [
                (cd_ref - initial_aero.cd_reference) / 0.12,
                (cd_slope - initial_aero.cd_log_re_slope) / 0.08,
                (magnus - initial_aero.magnus_lift_slope) / 0.10,
                (log_tau - math.log(max(initial_aero.spin_decay_tau_s, 0.2))) / 0.8,
                angle_bias / 2.0,
            ]
        )
        return np.asarray(residual, dtype=float)

    fit = least_squares(
        residuals,
        x0=np.array(
            [
                initial_aero.cd_reference,
                initial_aero.cd_log_re_slope,
                initial_aero.magnus_lift_slope,
                math.log(max(initial_aero.spin_decay_tau_s, 0.2)),
                initial_aero.launch_angle_bias_deg,
            ],
            dtype=float,
        ),
        bounds=(
            np.array([0.15, -0.15, 0.0, math.log(0.20), -6.0]),
            np.array([1.10, 0.20, 0.45, math.log(20.0), 6.0]),
        ),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=80,
        verbose=0,
    )
    cd_ref, cd_slope, magnus, log_tau, angle_bias = fit.x
    return replace(
        initial_aero,
        cd_reference=float(cd_ref),
        cd_log_re_slope=float(cd_slope),
        magnus_lift_slope=float(magnus),
        spin_decay_tau_s=float(math.exp(log_tau)),
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
    angle = rows.column("angle_deg")
    actual_angle = rows.column("actual_angle_deg")
    centered = _centered_row_mask(rows)

    observed_points: list[tuple[float, float, float]] = []
    for i in range(rows.count):
        if not np.isfinite(distance[i]):
            continue
        # Empirical command correction is indexed by the commanded hood angle.
        # Actual-angle deviations belong in shot noise, not in the lookup surface.
        shot_angle = (
            float(angle[i]) if np.isfinite(angle[i])
            else float(actual_angle[i]) if np.isfinite(actual_angle[i])
            else ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
        )
        if np.isfinite(recommended[i]):
            observed_points.append((float(distance[i]), shot_angle, float(recommended[i])))
        elif np.isfinite(rpm[i]) and centered[i]:
            observed_points.append((float(distance[i]), shot_angle, float(rpm[i])))

    if len(observed_points) < 2:
        return EmpiricalCorrection()

    robot = RobotState()
    grouped: dict[tuple[float, float], list[float]] = {}
    for d, a, observed_rpm in observed_points:
        key = (round(d, 6), round(a, 4))
        grouped.setdefault(key, []).append(observed_rpm)

    residual_d: list[float] = []
    residual_a: list[float] = []
    residual_rpm: list[float] = []
    for (d, a), observed_values in sorted(grouped.items()):
        baseline = nominal_center_rpm_for_angle(
            float(d), float(a), rpm_bounds, calibration, ball, aero, hub, shooter, robot
        )
        if baseline is None:
            continue
        residual_d.append(float(d))
        residual_a.append(float(a))
        residual_rpm.append(float(np.median(observed_values) - baseline))

    if len(residual_d) < 2:
        return EmpiricalCorrection()

    # PCHIP needs unique x in the single-angle case. Collapse duplicate distances.
    if len(np.unique(residual_a)) < 2:
        unique_d = np.unique(residual_d)
        collapsed_y = [
            float(np.median([r for dd, r in zip(residual_d, residual_rpm, strict=True) if math.isclose(dd, d, abs_tol=1e-9)]))
            for d in unique_d
        ]
        if len(unique_d) < 2:
            return EmpiricalCorrection()
        return EmpiricalCorrection(
            tuple(float(v) for v in unique_d),
            tuple(collapsed_y),
            tuple(float(residual_a[0]) for _ in unique_d),
        )

    return EmpiricalCorrection(tuple(residual_d), tuple(residual_rpm), tuple(residual_a))


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
def _numba_contact_impulse(
    vx: float, vy: float, vz: float,
    wx: float, wy: float, wz: float,
    nx: float, ny: float, nz: float,
    mass: float, radius: float, restitution: float, friction: float,
) -> tuple[float, float, float, float, float, float]:
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-12:
        return vx, vy, vz, wx, wy, wz
    nx /= norm; ny /= norm; nz /= norm
    # r_contact = -radius*n, contact velocity = v + omega x r.
    rx = -radius * nx; ry = -radius * ny; rz = -radius * nz
    cvx = vx + wy * rz - wz * ry
    cvy = vy + wz * rx - wx * rz
    cvz = vz + wx * ry - wy * rx
    vn = cvx * nx + cvy * ny + cvz * nz
    if vn >= 0.0:
        return vx, vy, vz, wx, wy, wz

    jn_mag = -(1.0 + restitution) * mass * vn
    jnx = jn_mag * nx; jny = jn_mag * ny; jnz = jn_mag * nz
    vtx = cvx - vn * nx; vty = cvy - vn * ny; vtz = cvz - vn * nz
    vtmag = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
    jtx = 0.0; jty = 0.0; jtz = 0.0
    if vtmag > 1e-12 and friction > 0.0:
        m_eff = (2.0 / 7.0) * mass
        jtx = -m_eff * vtx; jty = -m_eff * vty; jtz = -m_eff * vtz
        jtmag = math.sqrt(jtx * jtx + jty * jty + jtz * jtz)
        limit = friction * jn_mag
        if jtmag > limit and jtmag > 1e-12:
            scale = limit / jtmag
            jtx *= scale; jty *= scale; jtz *= scale

    jx = jnx + jtx; jy = jny + jty; jz = jnz + jtz
    vx += jx / mass; vy += jy / mass; vz += jz / mass
    inertia = (2.0 / 5.0) * mass * radius * radius
    if inertia > 1e-12:
        # torque impulse r x J_t
        tx = ry * jtz - rz * jty
        ty = rz * jtx - rx * jtz
        tz = rx * jty - ry * jtx
        wx += tx / inertia; wy += ty / inertia; wz += tz / inertia
    return vx, vy, vz, wx, wy, wz


@njit(cache=True, fastmath=True)
def _numba_rk4_step(
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
    wx: float, wy: float, wz: float,
    mass: float, diameter: float, cd_ref: float, cd_slope: float, ref_re: float,
    magnus_slope: float, magnus_re_slope: float, max_cl: float, spin_tau: float,
    wind_x: float, wind_y: float, wind_z: float, dt: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    a1x,a1y,a1z,dw1x,dw1y,dw1z = _numba_accel_spin(
        vx,vy,vz,wx,wy,wz,mass,diameter,cd_ref,cd_slope,ref_re,
        magnus_slope,magnus_re_slope,max_cl,spin_tau,wind_x,wind_y,wind_z)
    k1x=vx; k1y=vy; k1z=vz

    v2x=vx+0.5*dt*a1x; v2y=vy+0.5*dt*a1y; v2z=vz+0.5*dt*a1z
    w2x=wx+0.5*dt*dw1x; w2y=wy+0.5*dt*dw1y; w2z=wz+0.5*dt*dw1z
    a2x,a2y,a2z,dw2x,dw2y,dw2z = _numba_accel_spin(
        v2x,v2y,v2z,w2x,w2y,w2z,mass,diameter,cd_ref,cd_slope,ref_re,
        magnus_slope,magnus_re_slope,max_cl,spin_tau,wind_x,wind_y,wind_z)
    k2x=v2x; k2y=v2y; k2z=v2z

    v3x=vx+0.5*dt*a2x; v3y=vy+0.5*dt*a2y; v3z=vz+0.5*dt*a2z
    w3x=wx+0.5*dt*dw2x; w3y=wy+0.5*dt*dw2y; w3z=wz+0.5*dt*dw2z
    a3x,a3y,a3z,dw3x,dw3y,dw3z = _numba_accel_spin(
        v3x,v3y,v3z,w3x,w3y,w3z,mass,diameter,cd_ref,cd_slope,ref_re,
        magnus_slope,magnus_re_slope,max_cl,spin_tau,wind_x,wind_y,wind_z)
    k3x=v3x; k3y=v3y; k3z=v3z

    v4x=vx+dt*a3x; v4y=vy+dt*a3y; v4z=vz+dt*a3z
    w4x=wx+dt*dw3x; w4y=wy+dt*dw3y; w4z=wz+dt*dw3z
    a4x,a4y,a4z,dw4x,dw4y,dw4z = _numba_accel_spin(
        v4x,v4y,v4z,w4x,w4y,w4z,mass,diameter,cd_ref,cd_slope,ref_re,
        magnus_slope,magnus_re_slope,max_cl,spin_tau,wind_x,wind_y,wind_z)
    k4x=v4x; k4y=v4y; k4z=v4z

    sixth=dt/6.0
    x += sixth*(k1x+2.0*k2x+2.0*k3x+k4x)
    y += sixth*(k1y+2.0*k2y+2.0*k3y+k4y)
    z += sixth*(k1z+2.0*k2z+2.0*k3z+k4z)
    vx += sixth*(a1x+2.0*a2x+2.0*a3x+a4x)
    vy += sixth*(a1y+2.0*a2y+2.0*a3y+a4y)
    vz += sixth*(a1z+2.0*a2z+2.0*a3z+a4z)
    wx += sixth*(dw1x+2.0*dw2x+2.0*dw3x+dw4x)
    wy += sixth*(dw1y+2.0*dw2y+2.0*dw3y+dw4y)
    wz += sixth*(dw1z+2.0*dw2z+2.0*dw3z+dw4z)
    return x,y,z,vx,vy,vz,wx,wy,wz


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
    """Fast Monte Carlo trajectory using RK4 plus adaptive near-HUB substeps."""
    radius = 0.5 * diameter
    peak = z
    time_s = 0.0
    crossed_lip = False
    lip_x = math.nan; lip_y = math.nan
    lip_vx = math.nan; lip_vy = math.nan; lip_vz = math.nan
    lip_time = math.nan; lip_margin = -1e9
    rim_contacts = 0; funnel_contacts = 0; initially_clean = False
    funnel_bottom = lip_height - rim_thickness - funnel_depth
    rim_top = lip_height + 0.5 * rim_thickness
    rim_bottom = lip_height - 0.5 * rim_thickness

    while time_s < max_time:
        # Smaller steps near the score boundary/contact region. This keeps the fast
        # optimizer numerically consistent with the accurate nominal solver.
        step = dt
        dz = abs(z - lip_height)
        if crossed_lip or dz < 0.25:
            step = min(step, 0.0005)
        elif dz < 0.55:
            step = min(step, 0.0015)
        if time_s + step > max_time:
            step = max_time - time_s

        old_x=x; old_y=y; old_z=z
        old_vx=vx; old_vy=vy; old_vz=vz
        x,y,z,vx,vy,vz,wx,wy,wz = _numba_rk4_step(
            x,y,z,vx,vy,vz,wx,wy,wz,mass,diameter,cd_ref,cd_slope,ref_re,
            magnus_slope,magnus_re_slope,max_cl,spin_tau,wind_x,wind_y,wind_z,step)
        time_s += step
        if z > peak: peak = z

        if not crossed_lip and old_z >= lip_height and z < lip_height and vz < 0.0:
            denom = old_z - z
            frac = 0.0 if abs(denom) < 1e-12 else (old_z - lip_height) / denom
            lip_x = old_x + frac * (x - old_x)
            lip_y = old_y + frac * (y - old_y)
            lip_vx = old_vx + frac * (vx - old_vx)
            lip_vy = old_vy + frac * (vy - old_vy)
            lip_vz = old_vz + frac * (vz - old_vz)
            lip_time = time_s - step + frac * step
            proj, _ = _numba_hex_max_projection(lip_x, lip_y)
            lip_margin = opening_apothem - proj - radius
            initially_clean = lip_margin >= 0.0
            crossed_lip = True

        if crossed_lip:
            proj, normal_index = _numba_hex_max_projection(x, y)
            angle = normal_index * math.pi / 3.0
            nx = math.cos(angle); ny = math.sin(angle)
            inner = opening_apothem - radius
            outer = opening_apothem + rim_width + radius
            overlaps_z = (z - radius <= rim_top) and (z + radius >= rim_bottom)
            in_annulus = proj > inner and proj < outer
            if overlaps_z and in_annulus and vz < 0.0:
                rim_contacts += 1
                z = rim_top + radius + 1e-6
                vx,vy,vz,wx,wy,wz = _numba_contact_impulse(
                    vx,vy,vz,wx,wy,wz,0.0,0.0,1.0,mass,radius,restitution,friction)
                if proj < opening_apothem + 0.35 * rim_width:
                    vx,vy,vz,wx,wy,wz = _numba_contact_impulse(
                        vx,vy,vz,wx,wy,wz,-nx,-ny,0.35,mass,radius,
                        0.5*restitution,0.7*friction)

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
                    nx = math.cos(angle); ny = math.sin(angle)
                    penetration = proj - allowed
                    x -= penetration * nx; y -= penetration * ny
                    da_dz = (opening_apothem - throat_apothem) / max(funnel_depth, 1e-9)
                    # gradient is outward/invalid; negate it to point into free space.
                    vx,vy,vz,wx,wy,wz = _numba_contact_impulse(
                        vx,vy,vz,wx,wy,wz,-nx,-ny,da_dz,mass,radius,restitution,friction)

            if z <= funnel_bottom - radius:
                proj, _ = _numba_hex_max_projection(x, y)
                if proj <= throat_apothem - 0.10 * radius and vz < 0.0:
                    if initially_clean and rim_contacts == 0 and funnel_contacts == 0:
                        class_code = 1
                    elif rim_contacts > 0 and funnel_contacts > 0:
                        class_code = 4
                    elif rim_contacts > 0:
                        class_code = 2
                    else:
                        class_code = 3
                    return 1,class_code,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts
                return 0,0,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts

            if z > lip_height + 0.35 and vz > 0.0:
                return 0,0,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts
            proj, _ = _numba_hex_max_projection(x, y)
            if proj > opening_apothem + rim_width + 0.50:
                return 0,0,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts

        if z <= radius and vz < 0.0:
            return 0,0,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts

    return 0,0,lip_margin,lip_x,lip_y,lip_vx,lip_vy,lip_vz,lip_time,peak,rim_contacts,funnel_contacts


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

    # Common random numbers across candidates reduce ranking noise. Eleven model
    # dimensions include real FUEL size/mass variation and spin magnitude/axis.
    model_z = sobol_standard_normals(n, 11, seed + 11)
    shot_error = correlated_shot_errors(n, shot_distribution, seed + 29)
    if not include_shot:
        shot_error[:] = 0.0

    total = candidate_count * n
    rpm_command = np.repeat(rpms, n)
    angle_command = np.repeat(angles, n)

    def tile_col(array: np.ndarray, col: int) -> np.ndarray:
        return np.tile(array[:, col], candidate_count)

    mass = np.full(total, ball.mass_kg, dtype=float)
    diameter = np.full(total, ball.diameter_m, dtype=float)
    cd_ref = np.full(total, aero.cd_reference, dtype=float)
    cd_slope = np.full(total, aero.cd_log_re_slope, dtype=float)
    magnus = np.full(total, aero.magnus_lift_slope, dtype=float)
    spin_tau = np.full(total, aero.spin_decay_tau_s, dtype=float)
    speed_scale = np.ones(total, dtype=float)
    spin_model_scale = np.ones(total, dtype=float)
    tilt_horizontal_deg = np.zeros(total, dtype=float)
    tilt_vertical_deg = np.zeros(total, dtype=float)
    spin_scatter_z = np.zeros(total, dtype=float)

    if include_model:
        mass += uncertainty.mass_sigma_kg * tile_col(model_z, 0)
        mass = np.clip(mass, FUEL_MASS_MIN_KG, FUEL_MASS_MAX_KG)
        diameter += uncertainty.diameter_sigma_m * tile_col(model_z, 1)
        diameter = np.clip(diameter, 0.94 * ball.diameter_m, 1.06 * ball.diameter_m)
        cd_ref += uncertainty.cd_reference_sigma * tile_col(model_z, 2)
        cd_ref = np.clip(cd_ref, 0.15, 1.1)
        cd_slope += uncertainty.cd_log_re_slope_sigma * tile_col(model_z, 3)
        magnus += uncertainty.magnus_slope_sigma * tile_col(model_z, 4)
        magnus = np.clip(magnus, 0.0, 0.45)
        spin_tau *= np.maximum(0.2, 1.0 + uncertainty.spin_decay_fraction_sigma * tile_col(model_z, 5))
        speed_scale *= np.maximum(0.5, 1.0 + uncertainty.calibration_speed_scale_sigma * tile_col(model_z, 6))
        spin_model_scale *= np.maximum(0.3, 1.0 + uncertainty.spin_scale_sigma * tile_col(model_z, 7))
        tilt_horizontal_deg = uncertainty.spin_axis_tilt_sigma_deg * tile_col(model_z, 8)
        tilt_vertical_deg = uncertainty.spin_axis_tilt_sigma_deg * tile_col(model_z, 9)
        spin_scatter_z = tile_col(model_z, 10)

    error = np.tile(shot_error, (candidate_count, 1))
    actual_rpm = rpm_command + error[:, 0]

    # Preserve correlations learned for launch-speed residuals while making their
    # marginal sigma depend on RPM when repeated speed measurements are available.
    if include_shot:
        global_speed_sigma = math.sqrt(max(float(shot_distribution.covariance()[1, 1]), 1e-12))
        conditional_speed_sigma = calibration.speed_sigma_fraction_from_rpm(actual_rpm)
        error[:, 1] *= conditional_speed_sigma / global_speed_sigma
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
    nominal_spin_rpm = calibration.spin_from_rpm(actual_rpm)
    conditional_spin_sigma = calibration.spin_sigma_fraction_from_rpm(actual_rpm)
    spin_rpm = nominal_spin_rpm * spin_model_scale
    if include_model:
        spin_rpm *= np.maximum(0.2, 1.0 + conditional_spin_sigma * spin_scatter_z)

    position0 = np.column_stack(
        (
            -actual_distance + shooter.forward_offset_m,
            shooter.left_offset_m + lateral,
            exit_height,
        )
    )

    rotational_vx = -robot_omega * shooter.left_offset_m
    rotational_vy = robot_omega * shooter.forward_offset_m
    robot_exit_v = np.column_stack(
        (robot_vx + rotational_vx, robot_vy + rotational_vy, np.zeros(total))
    )

    # Per-sample moving-shot lead. Unlike the previous implementation this uses
    # each sample's perturbed distance, RPM/launch speed and robot velocity.
    target_xy = -position0[:, :2]
    target_len = np.maximum(np.linalg.norm(target_xy, axis=1), 1e-9)
    target_unit = target_xy / target_len[:, None]
    target_yaw = np.arctan2(target_unit[:, 1], target_unit[:, 0])
    perp = np.column_stack((-target_unit[:, 1], target_unit[:, 0]))
    v_perp = np.sum(robot_exit_v[:, :2] * perp, axis=1)
    horizontal_speed = np.maximum(
        launch_speed * np.cos(np.deg2rad(actual_angle)), 1e-6
    )
    lead_delta = np.arcsin(np.clip(-v_perp / horizontal_speed, -0.98, 0.98))
    yaw_rad = target_yaw + lead_delta + np.deg2rad(yaw_error)

    angle_rad = np.deg2rad(actual_angle)
    horizontal = np.cos(angle_rad)
    units = np.column_stack(
        (horizontal * np.cos(yaw_rad), horizontal * np.sin(yaw_rad), np.sin(angle_rad))
    )
    velocity0 = launch_speed[:, None] * units + robot_exit_v

    # Full 3-D spin vector with shot/model axis tilt rather than ideal backspin only.
    ideal_spin_axis = np.column_stack((np.sin(yaw_rad), -np.cos(yaw_rad), np.zeros(total)))
    launch_horizontal_axis = np.column_stack((np.cos(yaw_rad), np.sin(yaw_rad), np.zeros(total)))
    spin_axis = (
        ideal_spin_axis
        + np.tan(np.deg2rad(tilt_horizontal_deg))[:, None] * launch_horizontal_axis
        + np.tan(np.deg2rad(tilt_vertical_deg))[:, None] * np.array([0.0, 0.0, 1.0])
    )
    spin_axis /= np.maximum(np.linalg.norm(spin_axis, axis=1), 1e-12)[:, None]
    spin0 = spin_axis * (spin_rpm * 2.0 * math.pi / 60.0)[:, None]

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
    lip_vx, lip_vy, lip_vz = result[5], result[6], result[7]
    score = score.reshape(candidate_count, n)
    class_code = class_code.reshape(candidate_count, n)
    margin = margin.reshape(candidate_count, n)
    lip_vx = lip_vx.reshape(candidate_count, n)
    lip_vy = lip_vy.reshape(candidate_count, n)
    lip_vz = lip_vz.reshape(candidate_count, n)
    finite_margin = np.where(np.isfinite(margin), margin, -1.0)

    # When enough real rim/score data exists, blend the rigid contact result with
    # an empirical P(score | lip margin, entry angle) model. This captures foam/rim
    # behavior that a rigid-body approximation cannot fully reproduce.
    horizontal_at_lip = np.maximum(np.hypot(lip_vx, lip_vy), 1e-9)
    entry_angle_mc = np.degrees(np.arctan2(-lip_vz, horizontal_at_lip))
    empirical_probability = shot_distribution.empirical_score_probability(margin, entry_angle_mc)
    if empirical_probability is not None:
        blend = min(0.85, max(0.50, shot_distribution.score_model_rows / 200.0))
        sample_probability = (1.0 - blend) * score.astype(float) + blend * empirical_probability
        probability = np.mean(sample_probability, axis=1)
    else:
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
    correction_rpm = float(correction.correction_rpm(distance_m, ROBOT_NOMINAL_LAUNCH_ANGLE_DEG))

    # Preserve the original trusted fallback anchor exactly until real measured
    # RPM->exit-speed calibration replaces it.
    if (
        calibration.uses_fallback_anchor
        and abs(distance_m - CALIBRATION_DISTANCE_M) < 1e-9
        and np.any(np.isclose(angle_candidates_deg, ROBOT_NOMINAL_LAUNCH_ANGLE_DEG))
    ):
        best_angle = ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
        best_rpm = CALIBRATION_RPM
        final_stats = monte_carlo_candidates(
            distance_m, np.array([best_rpm]), np.array([best_angle]),
            max(final_samples, adaptive_max_samples), calibration, ball, aero, hub,
            shooter, robot, uncertainty, shot_distribution, "combined", seed,
        )
        model_stats = monte_carlo_candidates(
            distance_m, np.array([best_rpm]), np.array([best_angle]), final_samples,
            calibration, ball, aero, hub, shooter, robot, uncertainty,
            shot_distribution, "model", seed + 2,
        )
        shot_stats = monte_carlo_candidates(
            distance_m, np.array([best_rpm]), np.array([best_angle]), final_samples,
            calibration, ball, aero, hub, shooter, robot, uncertainty,
            shot_distribution, "shot", seed + 3,
        )
        nominal = simulate_shot(
            distance_m, best_rpm, best_angle, calibration, ball, aero, hub, shooter,
            robot, exact_lead=True, save_trajectory=True,
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

    # ------------------------------------------------------------------
    # Stage 1: genuine 2-D coarse search. For every angle, search a band of RPMs
    # around the deterministic center solution rather than evaluating only one RPM.
    # ------------------------------------------------------------------
    coarse_rpm: list[float] = []
    coarse_angle: list[float] = []
    offsets = np.arange(-200.0, 200.1, 50.0)
    for angle in angle_candidates_deg:
        center = nominal_center_rpm_for_angle(
            distance_m, float(angle), rpm_bounds, calibration, ball, aero, hub, shooter, robot
        )
        if center is None or not np.isfinite(center):
            continue
        angle_correction = float(correction.correction_rpm(distance_m, float(angle)))
        center = float(center + angle_correction)
        for offset in offsets:
            rpm = float(np.clip(center + offset, *rpm_bounds))
            rpm = float(_round_rpm(rpm))
            rpm = float(np.clip(rpm, *rpm_bounds))
            coarse_rpm.append(rpm)
            coarse_angle.append(float(angle))

    if not coarse_rpm:
        raise ValueError(f"No valid descending shot at {distance_m:.2f} m")

    coarse_pairs = np.unique(np.column_stack((coarse_rpm, coarse_angle)), axis=0)
    coarse_stats = adaptive_candidate_statistics(
        distance_m,
        coarse_pairs[:, 0],
        coarse_pairs[:, 1],
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
    coarse_index = choose_best_candidate(
        coarse_stats, distance_m, calibration, ball, aero, hub, shooter, robot
    )
    coarse_best_rpm = float(coarse_stats.rpm[coarse_index])
    coarse_best_angle = float(coarse_stats.angle_deg[coarse_index])

    # ------------------------------------------------------------------
    # Stage 2: local 2-D refinement. The previous implementation locked angle
    # before refining RPM, which could miss a more robust angle/RPM combination.
    # ------------------------------------------------------------------
    if len(angle_candidates_deg) > 1:
        unique_angles = np.sort(np.unique(angle_candidates_deg))
        coarse_angle_step = float(np.min(np.diff(unique_angles)))
        fine_angle_step = max(0.25, 0.5 * coarse_angle_step)
        angle_low = max(float(np.min(unique_angles)), coarse_best_angle - coarse_angle_step)
        angle_high = min(float(np.max(unique_angles)), coarse_best_angle + coarse_angle_step)
        fine_angles = np.arange(angle_low, angle_high + 0.5 * fine_angle_step, fine_angle_step)
    else:
        fine_angles = np.array([coarse_best_angle])

    fine_rpms = np.arange(
        max(rpm_bounds[0], coarse_best_rpm - 125.0),
        min(rpm_bounds[1], coarse_best_rpm + 125.0) + 0.1,
        ROUND_TO_RPM,
    )
    mesh_rpm, mesh_angle = np.meshgrid(fine_rpms, fine_angles, indexing="xy")
    refine_rpm = mesh_rpm.ravel()
    refine_angle = mesh_angle.ravel()

    refine_stats = adaptive_candidate_statistics(
        distance_m,
        refine_rpm,
        refine_angle,
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
    refine_index = choose_best_candidate(
        refine_stats,
        distance_m,
        calibration,
        ball,
        aero,
        hub,
        shooter,
        robot,
        preferred_rpm=coarse_best_rpm,
    )
    best_rpm = float(refine_stats.rpm[refine_index])
    best_angle = float(refine_stats.angle_deg[refine_index])

    # Final one-dimensional RPM sweep at the selected angle gives a meaningful
    # probability band and permits the final RPM to move if the high-sample sweep
    # discovers a better nearby quantized command.
    band_rpms = np.arange(
        max(rpm_bounds[0], best_rpm - 250.0),
        min(rpm_bounds[1], best_rpm + 250.0) + 0.1,
        ROUND_TO_RPM,
    )
    band_angles = np.full_like(band_rpms, best_angle)
    band_stats = adaptive_candidate_statistics(
        distance_m,
        band_rpms,
        band_angles,
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
        seed + 151,
    )
    band_best = choose_best_candidate(
        band_stats, distance_m, calibration, ball, aero, hub, shooter, robot,
        preferred_rpm=best_rpm,
    )
    best_rpm = float(band_stats.rpm[band_best])
    final_combined = float(band_stats.probability[band_best])
    final_clean = float(band_stats.clean_probability[band_best])
    final_contact = float(band_stats.rim_probability[band_best] + band_stats.funnel_probability[band_best])

    acceptable = band_stats.probability >= probability_threshold
    if np.any(acceptable):
        band_low = float(np.min(band_stats.rpm[acceptable]))
        band_high = float(np.max(band_stats.rpm[acceptable]))
    else:
        band_low = band_high = float("nan")

    correction_rpm = float(correction.correction_rpm(distance_m, best_angle))

    model_stats = monte_carlo_candidates(
        distance_m, np.array([best_rpm]), np.array([best_angle]), final_samples,
        calibration, ball, aero, hub, shooter, robot, uncertainty,
        shot_distribution, "model", seed + 201,
    )
    shot_stats = monte_carlo_candidates(
        distance_m, np.array([best_rpm]), np.array([best_angle]), final_samples,
        calibration, ball, aero, hub, shooter, robot, uncertainty,
        shot_distribution, "shot", seed + 301,
    )
    nominal = simulate_shot(
        distance_m, best_rpm, best_angle, calibration, ball, aero, hub, shooter,
        robot, exact_lead=True, save_trajectory=True,
    )
    return OptimizedShot(
        distance_m=distance_m,
        rpm=best_rpm,
        angle_deg=best_angle,
        learned_correction_rpm=correction_rpm,
        combined_probability=final_combined,
        model_only_probability=float(model_stats.probability[0]),
        shot_only_probability=float(shot_stats.probability[0]),
        clean_probability=final_clean,
        contact_probability=final_contact,
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
    on_result: Callable[[OptimizedShot], None] | None = None,
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
        results: list[OptimizedShot] = []
        for payload in iterator:
            shot = _optimize_worker(payload)
            results.append(shot)
            if on_result is not None:
                on_result(shot)
        return sorted(results, key=lambda shot: shot.distance_m)

    executor_cls = _executor_class(executor_kind)
    results: list[OptimizedShot] = []
    with executor_cls(max_workers=workers) as executor:
        future_map = {executor.submit(_optimize_worker, payload): payload.distance_m for payload in payloads}
        iterator = as_completed(future_map)
        if show_progress:
            iterator = tqdm(iterator, total=len(payloads), desc="Optimizing shots", unit="distance")
        for future in iterator:
            shot = future.result()
            results.append(shot)
            if on_result is not None:
                on_result(shot)
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
    validation_samples: int,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    robot: RobotState,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    seed: int,
    show_progress: bool = True,
) -> Path:
    """Write a dense lookup validated at the exact command the robot will use.

    RPM/angle are interpolated as control proposals, then RPM is quantized and every
    dense row is re-simulated. Probability is never interpolated from sparse points.
    """
    distances = np.asarray([shot.distance_m for shot in shots], dtype=float)
    rpms = np.asarray([shot.rpm for shot in shots], dtype=float)
    angles = np.asarray([shot.angle_deg for shot in shots], dtype=float)
    dense = np.arange(distances[0], distances[-1] + 0.5 * step_m, step_m)
    proposed_rpm = _monotonic_dense_curve(distances, rpms, dense)
    proposed_angle = _monotonic_dense_curve(distances, angles, dense)
    commanded_rpm = _round_rpm(proposed_rpm)

    # Vectorized candidate evaluation: each dense distance is different, so evaluate
    # by distance in a short loop. The expensive trajectories remain JIT compiled.
    probability = np.zeros(len(dense), dtype=float)
    clean_probability = np.zeros(len(dense), dtype=float)
    iterator: Iterable[int] = range(len(dense))
    if show_progress and len(dense) > 10:
        iterator = tqdm(iterator, total=len(dense), desc="Validating dense lookup", unit="row")
    for idx in iterator:
        stats = monte_carlo_candidates(
            float(dense[idx]),
            np.array([float(commanded_rpm[idx])]),
            np.array([float(proposed_angle[idx])]),
            validation_samples,
            calibration,
            ball,
            aero,
            hub,
            shooter,
            robot,
            uncertainty,
            shot_distribution,
            "combined",
            seed + 4099 * idx,
        )
        probability[idx] = float(stats.probability[0])
        clean_probability[idx] = float(stats.clean_probability[0])

    path = output_directory / "shooter_lookup_dense.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "distance_m",
                "interpolated_flywheel_rpm",
                "commanded_rounded_rpm",
                "hood_angle_deg",
                "validated_score_probability",
                "validated_clean_probability",
                "validation_samples",
            ]
        )
        for d, proposed, command, angle, p_score, p_clean in zip(
            dense, proposed_rpm, commanded_rpm, proposed_angle, probability, clean_probability, strict=True
        ):
            writer.writerow(
                [
                    f"{d:.3f}",
                    f"{proposed:.3f}",
                    f"{float(command):.1f}",
                    f"{angle:.4f}",
                    f"{p_score:.6f}",
                    f"{p_clean:.6f}",
                    int(validation_samples),
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
        "timestamp",
        "session_id",
        "ball_id",
        "distance_m",
        "rpm",
        "angle_deg",
        "exit_speed_mps",
        "spin_rpm",
        "spin_x_rpm",
        "spin_y_rpm",
        "spin_z_rpm",
        "result",
        "scored",
        "recommended_rpm",
        "actual_rpm",
        "rpm_before_shot",
        "rpm_min_during_shot",
        "rpm_after_shot",
        "time_since_previous_shot_s",
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
        "battery_voltage",
        "notes",
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
        output_file.write(f"Speed scatter source: {calibration.speed_sigma_fraction_curve.source if calibration.speed_sigma_fraction_curve else 'fallback constant'}\n")
        output_file.write(f"Spin scatter source: {calibration.spin_sigma_fraction_curve.source if calibration.spin_sigma_fraction_curve else 'fallback constant'}\n")
        output_file.write(f"Cd reference: {aero.cd_reference:.6f}\n")
        output_file.write(f"Cd log-Re slope: {aero.cd_log_re_slope:.6f}\n")
        output_file.write(f"Magnus slope: {aero.magnus_lift_slope:.6f}\n")
        output_file.write(f"Spin decay tau: {aero.spin_decay_tau_s:.6f} s\n")
        output_file.write(f"Launch-angle bias: {aero.launch_angle_bias_deg:+.6f} deg\n")
        output_file.write(f"Shot covariance learned rows: {shot_distribution.source_rows}\n")
        output_file.write(f"Empirical score/contact model rows: {shot_distribution.score_model_rows}\n")
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
# HELD-OUT VALIDATION + 3-D VISUALIZATION
# ===========================================================================


def _result_to_binary_score(text: str, explicit: float) -> float:
    if np.isfinite(explicit):
        return 1.0 if explicit >= 0.5 else 0.0
    label = text.strip().lower()
    if not label:
        return float("nan")
    if any(token in label for token in ("miss", "fail", "no_score", "noscore")):
        return 0.0
    if any(token in label for token in ("score", "center", "good", "bullseye", "made")):
        return 1.0
    return float("nan")


def write_validation_report(
    output_directory: Path,
    rows: CalibrationRows,
    calibration: ShooterCalibration,
    ball: BallModel,
    aero: AeroModel,
    hub: HubGeometry,
    shooter: ShooterGeometry,
    uncertainty: UncertaintyModel,
    shot_distribution: ShotErrorDistribution,
    samples: int,
    seed: int,
) -> Path | None:
    if rows.count == 0:
        return None
    distance = rows.column("distance_m")
    rpm = rows.column("rpm")
    actual_rpm = rows.column("actual_rpm")
    angle = rows.column("actual_angle_deg")
    command_angle = rows.column("angle_deg")
    lip_x = rows.column("lip_x_m")
    lip_y = rows.column("lip_y_m")
    entry = rows.column("entry_angle_deg")
    flight = rows.column("flight_time_s")
    explicit_scored = rows.column("scored")
    labels = rows.text_column("result")

    errors_x: list[float] = []
    errors_y: list[float] = []
    errors_entry: list[float] = []
    errors_flight: list[float] = []
    predicted_p: list[float] = []
    observed_y: list[float] = []
    prediction_rows: list[list[object]] = []
    robot = RobotState()

    for idx in range(rows.count):
        if not (np.isfinite(distance[idx]) and np.isfinite(rpm[idx])):
            continue
        use_rpm = float(actual_rpm[idx]) if np.isfinite(actual_rpm[idx]) else float(rpm[idx])
        use_angle = (
            float(angle[idx]) if np.isfinite(angle[idx])
            else float(command_angle[idx]) if np.isfinite(command_angle[idx])
            else ROBOT_NOMINAL_LAUNCH_ANGLE_DEG
        )
        try:
            nominal = simulate_shot(
                float(distance[idx]), use_rpm, use_angle, calibration, ball, aero,
                hub, shooter, robot, exact_lead=True, save_trajectory=False,
            )
        except (ValueError, RuntimeError):
            continue
        if nominal.reached_hub:
            if np.isfinite(lip_x[idx]): errors_x.append(float(nominal.position_at_lip_m[0] - lip_x[idx]))
            if np.isfinite(lip_y[idx]): errors_y.append(float(nominal.position_at_lip_m[1] - lip_y[idx]))
            if np.isfinite(entry[idx]): errors_entry.append(float(nominal.entry_angle_deg - entry[idx]))
            if np.isfinite(flight[idx]): errors_flight.append(float(nominal.flight_time_s - flight[idx]))

        stats = monte_carlo_candidates(
            float(distance[idx]), np.array([float(rpm[idx])]), np.array([use_angle]), samples,
            calibration, ball, aero, hub, shooter, robot, uncertainty, shot_distribution,
            "combined", seed + idx * 7919,
        )
        p_score = float(stats.probability[0])
        y_score = _result_to_binary_score(labels[idx], explicit_scored[idx])
        if np.isfinite(y_score):
            predicted_p.append(p_score)
            observed_y.append(y_score)
        prediction_rows.append([
            idx, float(distance[idx]), float(rpm[idx]), use_rpm, use_angle,
            p_score, y_score, nominal.classification,
        ])

    def mae(values: list[float]) -> float:
        return float(np.mean(np.abs(values))) if values else float("nan")

    brier = float("nan")
    log_loss = float("nan")
    if predicted_p:
        pp = np.clip(np.asarray(predicted_p), 1e-6, 1.0 - 1e-6)
        yy = np.asarray(observed_y)
        brier = float(np.mean((pp - yy) ** 2))
        log_loss = float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))

    reliability_rows: list[tuple[float, float, int]] = []
    if predicted_p:
        pp = np.asarray(predicted_p, dtype=float)
        yy = np.asarray(observed_y, dtype=float)
        edges = np.linspace(0.0, 1.0, 11)
        for b in range(10):
            if b == 9:
                mask = (pp >= edges[b]) & (pp <= edges[b + 1])
            else:
                mask = (pp >= edges[b]) & (pp < edges[b + 1])
            if np.any(mask):
                reliability_rows.append((float(np.mean(pp[mask])), float(np.mean(yy[mask])), int(np.count_nonzero(mask))))
        if reliability_rows:
            figure, axis = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
            axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Perfect calibration")
            axis.plot(
                [r[0] for r in reliability_rows],
                [r[1] for r in reliability_rows],
                "o-",
                label="Held-out shots",
            )
            axis.set(
                title="Held-out score-probability reliability",
                xlabel="Predicted P(score)",
                ylabel="Observed score fraction",
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            )
            axis.grid(alpha=0.3)
            axis.legend()
            figure.savefig(output_directory / "validation_reliability.png", dpi=180)
            plt.close(figure)

    report = output_directory / "held_out_validation.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("HELD-OUT SHOOTER MODEL VALIDATION\n")
        f.write("=================================\n")
        f.write(f"Rows supplied: {rows.count}\n")
        f.write(f"Lip X MAE: {mae(errors_x):.5f} m (n={len(errors_x)})\n")
        f.write(f"Lip Y MAE: {mae(errors_y):.5f} m (n={len(errors_y)})\n")
        f.write(f"Entry-angle MAE: {mae(errors_entry):.3f} deg (n={len(errors_entry)})\n")
        f.write(f"Flight-time MAE: {mae(errors_flight):.5f} s (n={len(errors_flight)})\n")
        f.write(f"Brier score: {brier:.6f} (n={len(predicted_p)})\n")
        f.write(f"Log loss: {log_loss:.6f} (n={len(predicted_p)})\n")
        if reliability_rows:
            f.write("\nProbability reliability bins (mean predicted -> actual, n):\n")
            for mean_pred, actual, count in reliability_rows:
                f.write(f"  {mean_pred:.3f} -> {actual:.3f}, n={count}\n")
        f.write("Validation data is never used to fit the calibration/physics model.\n")

    csv_path = output_directory / "held_out_validation_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row", "distance_m", "command_rpm", "actual_or_command_rpm", "angle_deg",
            "predicted_score_probability", "observed_score", "nominal_classification",
        ])
        writer.writerows(prediction_rows)
    return report


def write_3d_trajectory_archive(output_directory: Path, shots: Sequence[OptimizedShot]) -> Path:
    payload: dict[str, np.ndarray] = {
        "distance_m": np.asarray([s.distance_m for s in shots], dtype=float),
        "rpm": np.asarray([s.rpm for s in shots], dtype=float),
        "angle_deg": np.asarray([s.angle_deg for s in shots], dtype=float),
    }
    for i, shot in enumerate(shots):
        payload[f"path_{i}"] = (
            np.asarray(shot.nominal.full_path_m, dtype=float)
            if shot.nominal.full_path_m is not None else np.empty((0, 3), dtype=float)
        )
        payload[f"time_{i}"] = (
            np.asarray(shot.nominal.full_path_t_s, dtype=float)
            if shot.nominal.full_path_t_s is not None else np.empty((0,), dtype=float)
        )
    path = output_directory / "shooter_trajectories_3d.npz"
    np.savez_compressed(path, **payload)
    return path


class Live3DShotViewer:
    """Matplotlib 3-D viewer using the exact nominal path including HUB contacts."""

    def __init__(self, hub: HubGeometry, ball: BallModel, fps: float = 60.0, time_scale: float = 1.0):
        self.hub = hub
        self.ball = ball
        self.fps = max(float(fps), 5.0)
        self.time_scale = max(float(time_scale), 0.05)
        self.enabled = True
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            self._Poly3DCollection = Poly3DCollection
            plt.ion()
            self.figure = plt.figure(figsize=(10.5, 7.2), layout="constrained")
            self.axis = self.figure.add_subplot(111, projection="3d")
            self.axis.set_xlabel("x toward HUB (m)")
            self.axis.set_ylabel("y left (m)")
            self.axis.set_zlabel("height (m)")
            self.axis.set_zlim(0.0, max(3.0, hub.lip_height_m + 1.0))
            self.axis.set_ylim(-1.2, 1.2)
            self._draw_hub()
            self.path_line, = self.axis.plot([], [], [], linewidth=2.0, label="trajectory")
            self.ball_marker = self.axis.scatter([], [], [], s=130, depthshade=True, label="FUEL")
            self.axis.legend(loc="upper left")
            self.figure.canvas.draw_idle()
            plt.pause(0.001)
        except Exception as exc:
            self.enabled = False
            warnings.warn(f"Could not start interactive 3-D viewer: {exc}")

    def _hex_vertices(self, apothem: float, z: float) -> np.ndarray:
        radius = apothem / math.cos(math.pi / 6.0)
        angles = np.deg2rad(np.arange(30.0, 390.0, 60.0))
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(6, z)))

    def _draw_hub(self) -> None:
        top = self._hex_vertices(self.hub.opening_apothem_m, self.hub.lip_height_m)
        bottom = self._hex_vertices(self.hub.throat_apothem_m, self.hub.funnel_bottom_z_m)
        loop_top = np.vstack((top, top[0]))
        loop_bottom = np.vstack((bottom, bottom[0]))
        self.axis.plot(loop_top[:,0], loop_top[:,1], loop_top[:,2], linewidth=2.5, label="HUB opening")
        self.axis.plot(loop_bottom[:,0], loop_bottom[:,1], loop_bottom[:,2], linestyle="--", linewidth=1.5)
        faces = []
        for i in range(6):
            j = (i + 1) % 6
            faces.append([top[i], top[j], bottom[j], bottom[i]])
        poly = self._Poly3DCollection(faces, alpha=0.12, linewidth=0.7)
        self.axis.add_collection3d(poly)
        self.axis.plot([-0.65,0.65,0.65,-0.65,-0.65],[-0.65,-0.65,0.65,0.65,-0.65],[0,0,0,0,0], alpha=0.35)

    def play_shot(self, shot: OptimizedShot) -> None:
        if not self.enabled:
            return
        path = shot.nominal.full_path_m
        times = shot.nominal.full_path_t_s
        if path is None or len(path) < 2:
            return
        path = np.asarray(path, dtype=float)
        times = np.asarray(times, dtype=float) if times is not None else np.linspace(0, 1, len(path))
        self.axis.set_xlim(min(-0.5, float(np.min(path[:,0])) - 0.3), 0.8)
        self.axis.set_title(
            f"{shot.distance_m:.2f} m  |  {shot.rpm:.0f} RPM @ {shot.angle_deg:.2f}°  |  {shot.nominal.classification}"
        )
        self.path_line.set_data([], [])
        self.path_line.set_3d_properties([])
        max_frames = max(2, int(self.fps * max(times[-1] - times[0], 0.1) / self.time_scale))
        indices = np.unique(np.linspace(0, len(path)-1, max_frames).astype(int))
        for idx in indices:
            segment = path[:idx+1]
            self.path_line.set_data(segment[:,0], segment[:,1])
            self.path_line.set_3d_properties(segment[:,2])
            self.ball_marker._offsets3d = ([path[idx,0]], [path[idx,1]], [path[idx,2]])
            self.figure.canvas.draw_idle()
            plt.pause(1.0 / self.fps)
        plt.pause(0.10)

    def hold(self) -> None:
        if not self.enabled:
            return
        plt.ioff()
        plt.show()


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
    parser.add_argument("--samples", type=int, default=256, help="Base final MC samples per candidate")
    parser.add_argument("--optimizer-samples", type=int, default=96, help="Base MC samples in coarse 2-D search")
    parser.add_argument("--adaptive-max-samples", type=int, default=1024, help="Sobol samples for competitive candidates")
    parser.add_argument("--heatmap-samples", type=int, default=96, help="Samples per heatmap cell")
    parser.add_argument("--dense-validation-samples", type=int, default=128, help="MC samples used to validate each dense rounded lookup row")
    parser.add_argument("--validation-samples", type=int, default=256, help="MC samples per held-out validation row")
    parser.add_argument("--probability-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--executor", choices=("process", "thread"), default="thread", help="thread is usually fastest with the Numba kernel; process is available for isolation")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--calibration-csv", type=Path, default=None, help="Training/calibration shot log")
    parser.add_argument("--validation-csv", type=Path, default=None, help="Held-out shot log; never used for fitting")
    parser.add_argument("--auto-calibrate", action="store_true", help="Fit regularized physics and shot covariance from training logs")
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

    # Internal HUB/contact defaults are approximations unless replaced from team
    # CAD/measurements. The official top opening and 72in height remain fixed above.
    parser.add_argument("--hub-rim-width", type=float, default=HUB_RIM_RADIAL_WIDTH_M)
    parser.add_argument("--hub-rim-thickness", type=float, default=HUB_RIM_THICKNESS_M)
    parser.add_argument("--hub-funnel-depth", type=float, default=HUB_FUNNEL_DEPTH_M)
    parser.add_argument("--hub-throat-flat-to-flat", type=float, default=HUB_FUNNEL_THROAT_FLAT_TO_FLAT_M)
    parser.add_argument("--hub-restitution", type=float, default=HUB_CONTACT_RESTITUTION)
    parser.add_argument("--hub-friction", type=float, default=HUB_CONTACT_FRICTION)

    parser.add_argument("--live-3d", action="store_true", help="Animate each optimized shot as soon as its calculation completes")
    parser.add_argument("--show-3d", action="store_true", help="Replay all optimized shots in the 3-D viewer after calculation", default=True)
    parser.add_argument("--3d-fps", type=float, default=60.0, dest="viewer_fps")
    parser.add_argument("--3d-time-scale", type=float, default=1.0, dest="viewer_time_scale", help="1=real time, 0.5=half-speed, 2=double-speed")
    parser.add_argument("--no-3d-hold", action="store_true", help="Do not block on the 3-D window after the run")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples < 16 or args.optimizer_samples < 16 or args.heatmap_samples < 8:
        parser.error("Monte Carlo sample counts are too small")
    if args.dense_validation_samples < 16 or args.validation_samples < 16:
        parser.error("validation sample counts must be at least 16")
    if args.adaptive_max_samples < max(args.samples, args.optimizer_samples):
        parser.error("--adaptive-max-samples must be >= both --samples and --optimizer-samples")
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
    if args.viewer_fps <= 0.0 or args.viewer_time_scale <= 0.0:
        parser.error("3-D FPS and time scale must be positive")
    if min(args.hub_rim_width, args.hub_rim_thickness, args.hub_funnel_depth, args.hub_throat_flat_to_flat) <= 0.0:
        parser.error("HUB geometry dimensions must be positive")
    if not (0.0 <= args.hub_restitution <= 1.0):
        parser.error("--hub-restitution must be in [0,1]")
    if not (0.0 <= args.hub_friction <= 2.0):
        parser.error("--hub-friction must be in [0,2]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if not NUMBA_AVAILABLE:
        warnings.warn("Numba is not installed; batch Monte Carlo will be much slower. pip install numba")

    output_directory: Path = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    ball = BallModel()
    hub = HubGeometry(
        rim_radial_width_m=float(args.hub_rim_width),
        rim_thickness_m=float(args.hub_rim_thickness),
        funnel_depth_m=float(args.hub_funnel_depth),
        throat_flat_to_flat_m=float(args.hub_throat_flat_to_flat),
        restitution=float(args.hub_restitution),
        friction=float(args.hub_friction),
    )
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

    # Training and validation are deliberately separate. The validation CSV is
    # never passed to any fitter, preventing optimistic in-sample accuracy reports.
    rows = load_calibration_rows(args.calibration_csv)
    validation_rows = load_calibration_rows(args.validation_csv)
    calibration = build_calibration(rows, ball, aero, hub, shooter)

    if args.auto_calibrate and rows.count > 0:
        aero = fit_physics_from_logs(rows, calibration, ball, aero, hub, shooter)
        if calibration.speed_curve is None:
            calibration = build_calibration(rows, ball, aero, hub, shooter)

    shot_distribution = estimate_shot_error_distribution(rows, calibration, uncertainty, hub, ball)
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

    # Warm the Numba cache in the parent so worker startup is predictable.
    if NUMBA_AVAILABLE:
        dummy = np.array([[0.0, 0.0, shooter.exit_height_m]])
        dummy_v = np.array([[1.0, 0.0, 3.0]])
        dummy_s = np.array([[0.0, -10.0, 0.0]])
        _numba_batch_kernel(
            dummy, dummy_v, dummy_s,
            np.array([ball.mass_kg]), np.array([ball.diameter_m]),
            np.array([aero.cd_reference]), np.array([aero.cd_log_re_slope]),
            aero.reference_reynolds, np.array([aero.magnus_lift_slope]),
            aero.magnus_re_slope, aero.max_cl, np.array([aero.spin_decay_tau_s]),
            np.zeros((1, 3)), hub.opening_apothem_m, hub.lip_height_m,
            hub.rim_radial_width_m, hub.rim_thickness_m, hub.funnel_depth_m,
            hub.throat_apothem_m, hub.restitution, hub.friction, 0.003, 0.01,
        )

    viewer: Live3DShotViewer | None = None
    if args.live_3d:
        viewer = Live3DShotViewer(hub, ball, fps=args.viewer_fps, time_scale=args.viewer_time_scale)

    def on_shot(shot: OptimizedShot) -> None:
        print(
            f"{shot.distance_m:4.2f} m -> {shot.rpm:6.0f} RPM @ {shot.angle_deg:5.2f} deg | "
            f"P(score)={shot.combined_probability:6.1%} | clean={shot.clean_probability:6.1%} | "
            f"entry={shot.nominal.entry_angle_deg:5.1f} deg | {shot.nominal.classification}"
        )
        if viewer is not None:
            viewer.play_shot(shot)

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
        on_result=on_shot if args.live_3d else None,
    )

    # If live mode was not requested, print in sorted distance order now.
    if not args.live_3d:
        for shot in shots:
            print(
                f"{shot.distance_m:4.2f} m -> {shot.rpm:6.0f} RPM @ {shot.angle_deg:5.2f} deg | "
                f"P(score)={shot.combined_probability:6.1%} | clean={shot.clean_probability:6.1%} | "
                f"entry={shot.nominal.entry_angle_deg:5.1f} deg | {shot.nominal.classification}"
            )

    write_lookup_csv(output_directory, shots)
    write_dense_lookup(
        output_directory, shots, float(args.dense_step), int(args.dense_validation_samples),
        calibration, ball, aero, hub, shooter, robot, uncertainty, shot_distribution,
        int(args.seed) + 500001,
        not bool(args.no_progress),
    )
    write_rpm_plot(output_directory, shots)
    write_trajectory_plot(output_directory, shots, hub)
    write_3d_trajectory_archive(output_directory, shots)
    write_shot_log_template(output_directory)
    write_calibration_plot(output_directory, rows, calibration)
    write_calibration_report(output_directory, calibration, aero, shot_distribution, correction)
    if validation_rows.count > 0:
        write_validation_report(
            output_directory, validation_rows, calibration, ball, aero, hub, shooter,
            uncertainty, shot_distribution, int(args.validation_samples), int(args.seed) + 700001,
        )

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

    if args.show_3d and not args.live_3d:
        viewer = Live3DShotViewer(hub, ball, fps=args.viewer_fps, time_scale=args.viewer_time_scale)
        for shot in shots:
            viewer.play_shot(shot)

    if viewer is not None and not args.no_3d_hold:
        viewer.hold()

    print_java_entries(shots)
    print("\nModel summary:")
    print("  RPM definition: measured flywheel / launcher-roller RPM")
    print(f"  Shooter geometry: z={shooter.exit_height_m:.3f} m, x-offset={shooter.forward_offset_m:.3f} m, y-offset={shooter.left_offset_m:.3f} m")
    print(f"  Nominal launch angle: {ROBOT_NOMINAL_LAUNCH_ANGLE_DEG:.1f} deg")
    print(f"  Calibration: {calibration.source}")
    print(f"  1400 RPM -> {float(calibration.speed_from_rpm(CALIBRATION_RPM)):.3f} m/s")
    print(f"  Official FUEL mass envelope used in MC: {FUEL_MASS_MIN_KG:.3f}-{FUEL_MASS_MAX_KG:.3f} kg")
    print(f"  Cd(reference): {aero.cd_reference:.4f}")
    print(f"  Cd log-Re slope: {aero.cd_log_re_slope:+.4f}")
    print(f"  Magnus slope: {aero.magnus_lift_slope:.4f}")
    print(f"  Spin decay tau: {aero.spin_decay_tau_s:.3f} s")
    print(f"  Launch-angle bias: {aero.launch_angle_bias_deg:+.3f} deg")
    print(f"  Learned shot-covariance rows: {shot_distribution.source_rows}")
    print(f"  Empirical score/contact rows: {shot_distribution.score_model_rows}")
    print(f"  Empirical RPM correction points: {len(correction.x)}")
    print(f"  Held-out validation rows: {validation_rows.count}")
    print(f"  Numba JIT: {'enabled' if NUMBA_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Outputs: {output_directory}")


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows.
    main()
