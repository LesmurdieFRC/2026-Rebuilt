#!/usr/bin/env python3
"""Estimate flywheel RPM versus hub distance for the 2026 FUEL.

This is a starting-point model, not a replacement for on-robot calibration. It
integrates a two-dimensional projectile with quadratic aerodynamic drag, then
calibrates the unknown ball-speed-per-flywheel-RPM factor so that the trusted
2.0 m shot is exactly 1400 RPM. A seeded Monte Carlo sweep shows how much the
curve moves when the least-certain physical inputs are varied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


G_MPS2 = 9.80665
AIR_DENSITY_KG_M3 = 1.225

# Official 2026 REBUILT presented by Haas dimensions.
FUEL_DIAMETER_M = 0.150
FUEL_MASS_KG = 0.215  # Midpoint of the official 0.448-0.500 lb range.
HUB_OPENING_HEIGHT_M = 1.8288  # 72 in above carpet.

# Robot measurements/assumptions. Measure these two values on the finished robot.
SHOOTER_EXIT_HEIGHT_M = 0.60
LAUNCH_ANGLE_DEG = 70.0

# Aim the ball center one radius plus 25 mm above the front lip. The 70 degree
# angle makes the nominal 1 m shot cross this plane while descending.
LIP_CLEARANCE_M = 0.025
SPHERE_DRAG_COEFFICIENT = 0.47

CALIBRATION_DISTANCE_M = 2.0
CALIBRATION_RPM = 1400.0
DISTANCES_M = np.arange(1.0, 6.01, 0.5)
ROUND_TO_RPM = 25.0


@dataclass(frozen=True)
class ShotParameters:
    mass_kg: float = FUEL_MASS_KG
    diameter_m: float = FUEL_DIAMETER_M
    drag_coefficient: float = SPHERE_DRAG_COEFFICIENT
    exit_height_m: float = SHOOTER_EXIT_HEIGHT_M
    launch_angle_deg: float = LAUNCH_ANGLE_DEG
    lip_clearance_m: float = LIP_CLEARANCE_M

    @property
    def target_center_height_m(self) -> float:
        return HUB_OPENING_HEIGHT_M + self.diameter_m / 2.0 + self.lip_clearance_m

    @property
    def drag_factor_per_m(self) -> float:
        area_m2 = np.pi * (self.diameter_m / 2.0) ** 2
        return 0.5 * AIR_DENSITY_KG_M3 * self.drag_coefficient * area_m2 / self.mass_kg


def state_at_distance(
    launch_speed_mps: float, distance_m: float, parameters: ShotParameters
) -> tuple[float, float, float]:
    """Return height, horizontal velocity, and vertical velocity at distance."""
    angle_rad = np.deg2rad(parameters.launch_angle_deg)
    initial_state = [
        parameters.exit_height_m,
        launch_speed_mps * np.cos(angle_rad),
        launch_speed_mps * np.sin(angle_rad),
    ]
    drag = parameters.drag_factor_per_m

    def derivative(_x: float, state: np.ndarray) -> list[float]:
        _height, velocity_x, velocity_z = state
        speed = np.hypot(velocity_x, velocity_z)
        return [
            velocity_z / velocity_x,
            -drag * speed,
            (-G_MPS2 - drag * speed * velocity_z) / velocity_x,
        ]

    solution = solve_ivp(
        derivative,
        (0.0, distance_m),
        initial_state,
        rtol=2e-7,
        atol=2e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return tuple(float(value) for value in solution.y[:, -1])


def required_launch_speed(distance_m: float, parameters: ShotParameters) -> float:
    """Solve for launch speed that puts the ball center over the hub lip."""
    target_height = parameters.target_center_height_m

    def height_error(speed_mps: float) -> float:
        try:
            height_m, _velocity_x, _velocity_z = state_at_distance(
                speed_mps, distance_m, parameters
            )
        except RuntimeError:
            # This trial has stalled horizontally before reaching the target.
            # It is unambiguously below the valid-shot root.
            return -1.0e6
        return height_m - target_height

    low_speed = 1.0
    high_speed = 35.0
    if height_error(high_speed) <= 0.0:
        raise ValueError(
            f"No shot solution at {distance_m:.2f} m for the configured geometry"
        )
    return float(brentq(height_error, low_speed, high_speed, xtol=1e-8))


def rpm_curve(parameters: ShotParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_speed = required_launch_speed(CALIBRATION_DISTANCE_M, parameters)
    launch_speeds = np.array(
        [required_launch_speed(distance, parameters) for distance in DISTANCES_M]
    )
    rpm = CALIBRATION_RPM * launch_speeds / calibration_speed
    vertical_velocities = np.array(
        [
            state_at_distance(speed, distance, parameters)[2]
            for speed, distance in zip(launch_speeds, DISTANCES_M, strict=True)
        ]
    )
    return launch_speeds, rpm, vertical_velocities


def uncertainty_bounds(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 10th/90th percentiles from plausible parameter variation."""
    random = np.random.default_rng(2026)
    curves: list[np.ndarray] = []
    for _ in range(sample_count):
        parameters = ShotParameters(
            mass_kg=random.uniform(0.203, 0.227),
            drag_coefficient=np.clip(random.normal(0.47, 0.05), 0.35, 0.60),
            exit_height_m=random.normal(SHOOTER_EXIT_HEIGHT_M, 0.05),
            launch_angle_deg=random.normal(LAUNCH_ANGLE_DEG, 1.5),
            lip_clearance_m=np.clip(random.normal(LIP_CLEARANCE_M, 0.012), 0.0, 0.05),
        )
        try:
            _speeds, rpm, vertical_velocities = rpm_curve(parameters)
        except ValueError:
            continue
        # An upward crossing cannot enter through the top of the HUB.
        if np.all(vertical_velocities < 0.0):
            curves.append(rpm)

    if len(curves) < max(20, sample_count // 10):
        raise RuntimeError("Too few physically valid uncertainty samples")
    samples = np.asarray(curves)
    return np.percentile(samples, 10, axis=0), np.percentile(samples, 90, axis=0)


def rounded_rpm(rpm: np.ndarray) -> np.ndarray:
    return np.round(rpm / ROUND_TO_RPM) * ROUND_TO_RPM


def write_outputs(
    output_directory: Path,
    launch_speeds: np.ndarray,
    rpm: np.ndarray,
    low_rpm: np.ndarray,
    high_rpm: np.ndarray,
    vertical_velocities: np.ndarray,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    rounded = rounded_rpm(rpm)

    with (output_directory / "shooter_velocity_estimates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "distance_m",
                "launch_speed_mps",
                "estimated_flywheel_rpm",
                "rounded_flywheel_rpm",
                "sensitivity_p10_rpm",
                "sensitivity_p90_rpm",
                "vertical_speed_at_hub_mps",
            ]
        )
        for values in zip(
            DISTANCES_M,
            launch_speeds,
            rpm,
            rounded,
            low_rpm,
            high_rpm,
            vertical_velocities,
            strict=True,
        ):
            writer.writerow([f"{value:.3f}" for value in values])

    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    axis.fill_between(
        DISTANCES_M,
        low_rpm,
        high_rpm,
        alpha=0.22,
        label="10th-90th percentile model sensitivity",
    )
    axis.plot(DISTANCES_M, rpm, "o-", label="Nominal drag model")
    axis.plot(
        [CALIBRATION_DISTANCE_M],
        [CALIBRATION_RPM],
        "s",
        markersize=8,
        label="Known calibration point",
    )
    axis.set(
        title="2026 Shooter Starting-Point Model",
        xlabel="Robot-center distance to HUB center (m)",
        ylabel="Flywheel motor setpoint (RPM)",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_directory / "shooter_velocity_model.png", dpi=180)
    plt.close(figure)

    print("\nJava interpolation-map entries (rounded to nearest 25 RPM):")
    for distance, setpoint in zip(DISTANCES_M, rounded, strict=True):
        print(f"FLYWHEEL_RPM_BY_DISTANCE.put({distance:.1f}, {setpoint:.1f});")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Monte Carlo samples used for sensitivity bounds (default: 300)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    arguments = parser.parse_args()

    parameters = ShotParameters()
    launch_speeds, rpm, vertical_velocities = rpm_curve(parameters)
    low_rpm, high_rpm = uncertainty_bounds(arguments.samples)
    write_outputs(
        arguments.output_directory,
        launch_speeds,
        rpm,
        low_rpm,
        high_rpm,
        vertical_velocities,
    )

    calibration_speed = required_launch_speed(CALIBRATION_DISTANCE_M, parameters)
    print(f"\nCalibrated ball launch speed at 2 m: {calibration_speed:.3f} m/s")
    print(f"Effective launch-speed conversion: {calibration_speed / CALIBRATION_RPM:.6f} m/s/RPM")
    print(f"Target ball-center height: {parameters.target_center_height_m:.3f} m")
    print(f"Outputs written to: {arguments.output_directory}")


if __name__ == "__main__":
    main()
