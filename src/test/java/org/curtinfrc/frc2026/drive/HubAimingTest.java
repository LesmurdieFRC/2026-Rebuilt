// Copyright (c) 2026 Curtin FRC
// Use of this source code is governed by the LICENSE file at the repository root.

package org.curtinfrc.frc2026.drive;

import static org.junit.jupiter.api.Assertions.assertEquals;

import edu.wpi.first.math.geometry.Translation2d;
import org.junit.jupiter.api.Test;

class HubAimingTest {
  private static final double EPSILON = 1e-9;

  @Test
  void headingToTargetPreservesAllFourQuadrants() {
    Translation2d origin = Translation2d.kZero;

    assertEquals(
        45.0, HubAiming.headingToTarget(origin, new Translation2d(1.0, 1.0)).getDegrees(), EPSILON);
    assertEquals(
        135.0,
        HubAiming.headingToTarget(origin, new Translation2d(-1.0, 1.0)).getDegrees(),
        EPSILON);
    assertEquals(
        -135.0,
        HubAiming.headingToTarget(origin, new Translation2d(-1.0, -1.0)).getDegrees(),
        EPSILON);
    assertEquals(
        -45.0,
        HubAiming.headingToTarget(origin, new Translation2d(1.0, -1.0)).getDegrees(),
        EPSILON);
  }

  @Test
  void headingToTargetUsesFieldXAxisAsZero() {
    Translation2d robot = new Translation2d(2.0, 3.0);

    assertEquals(
        0.0, HubAiming.headingToTarget(robot, new Translation2d(4.0, 3.0)).getDegrees(), EPSILON);
    assertEquals(
        90.0, HubAiming.headingToTarget(robot, new Translation2d(2.0, 5.0)).getDegrees(), EPSILON);
    assertEquals(
        180.0,
        Math.abs(HubAiming.headingToTarget(robot, new Translation2d(0.0, 3.0)).getDegrees()),
        EPSILON);
    assertEquals(
        -90.0, HubAiming.headingToTarget(robot, new Translation2d(2.0, 1.0)).getDegrees(), EPSILON);
  }
}
