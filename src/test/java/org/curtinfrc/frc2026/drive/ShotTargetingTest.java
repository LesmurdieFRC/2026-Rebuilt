package org.curtinfrc.frc2026.drive;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj.DriverStation.Alliance;
import org.junit.jupiter.api.Test;

class ShotTargetingTest {
  private static final double FIELD_LENGTH = 16.541;
  private static final double FIELD_WIDTH = 8.069;
  private static final double CENTER_Y = FIELD_WIDTH / 2.0;
  private static final double BLUE_HUB_X = 4.619;
  private static final double RED_HUB_X = 11.909;
  private static final double BLUE_ZONE_X = 4.022;
  private static final double RED_ZONE_X = 12.519;
  private static final double HUB_WIDTH = 1.194;

  @Test
  void aimsAtHubOutsideMidfield() {
    var blueTarget = targetFor(Pose2d.kZero, Alliance.Blue);
    var redTarget = targetFor(new Pose2d(FIELD_LENGTH, 0.0, Rotation2d.kZero), Alliance.Red);

    assertFalse(blueTarget.returnsToAllianceSide());
    assertFalse(redTarget.returnsToAllianceSide());
  }

  @Test
  void midfieldTargetsLandInsideCorrectAllianceZone() {
    var midfieldPose = new Pose2d(FIELD_LENGTH / 2.0, CENTER_Y, Rotation2d.kZero);
    var blueTarget = targetFor(midfieldPose, Alliance.Blue);
    var redTarget = targetFor(midfieldPose, Alliance.Red);

    assertTrue(blueTarget.returnsToAllianceSide());
    assertTrue(redTarget.returnsToAllianceSide());
    assertTrue(blueTarget.position().getX() < BLUE_ZONE_X);
    assertTrue(redTarget.position().getX() > RED_ZONE_X);
  }

  private static ShotTargeting.Target targetFor(Pose2d pose, Alliance alliance) {
    var hub = new Translation2d(alliance == Alliance.Blue ? BLUE_HUB_X : RED_HUB_X, CENTER_Y);
    return ShotTargeting.targetForFieldGeometry(
        pose,
        alliance,
        hub,
        BLUE_HUB_X,
        RED_HUB_X,
        BLUE_ZONE_X,
        RED_ZONE_X,
        CENTER_Y,
        FIELD_WIDTH,
        HUB_WIDTH);
  }
}
