// Copyright (c) 2026 Curtin FRC
// Use of this source code is governed by the LICENSE file at the repository root.

package org.curtinfrc.frc2026.drive;

import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;

/** Pure field-geometry calculations used while aiming at the hub. */
final class HubAiming {
  private HubAiming() {}

  /**
   * Returns the point on a circle around {@code center} that is closest to {@code current}.
   *
   * <p>If {@code current} is exactly at the center, the positive field X direction is used as a
   * deterministic fallback.
   */
  static Translation2d nearestPointOnCircle(
      Translation2d current, Translation2d center, double radiusMeters) {
    Translation2d centerToCurrent = current.minus(center);
    double currentRadius = centerToCurrent.getNorm();
    if (currentRadius == 0.0) {
      return center.plus(new Translation2d(radiusMeters, 0.0));
    }

    return center.plus(centerToCurrent.times(radiusMeters / currentRadius));
  }

  /**
   * Returns the field-relative heading whose positive X axis points from {@code current} to target.
   */
  static Rotation2d headingToTarget(Translation2d current, Translation2d target) {
    return target.minus(current).getAngle();
  }
}
