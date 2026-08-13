package org.curtinfrc.frc2026.drive;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj.DriverStation.Alliance;
import org.curtinfrc.frc2026.util.FieldConstants;

/** Selects either the alliance hub or a safe alliance-side floor landing target. */
final class ShotTargeting {
  private static final double ALLIANCE_ZONE_LANDING_MARGIN_METERS = 0.50;
  private static final double HUB_SIDE_CLEARANCE_METERS = 0.50;

  record Target(Translation2d position, boolean returnsToAllianceSide) {}

  private ShotTargeting() {}

  static Target targetFor(Pose2d robotPose, Alliance alliance) {
    return targetForFieldGeometry(
        robotPose,
        alliance,
        FieldConstants.Hub.getAllianceCenter(alliance),
        FieldConstants.LinesVertical.hubCenter,
        FieldConstants.LinesVertical.oppHubCenter,
        FieldConstants.LinesVertical.allianceZone,
        FieldConstants.LinesVertical.oppAllianceZone,
        FieldConstants.LinesHorizontal.center,
        FieldConstants.fieldWidth,
        FieldConstants.Hub.width);
  }

  static Target targetForFieldGeometry(
      Pose2d robotPose,
      Alliance alliance,
      Translation2d allianceHub,
      double blueHubX,
      double redHubX,
      double blueAllianceZoneX,
      double redAllianceZoneX,
      double fieldCenterY,
      double fieldWidth,
      double hubWidth) {
    if (!isBetweenHubs(robotPose.getX(), blueHubX, redHubX)) {
      return new Target(allianceHub, false);
    }

    double targetX =
        alliance == Alliance.Red
            ? redAllianceZoneX + ALLIANCE_ZONE_LANDING_MARGIN_METERS
            : blueAllianceZoneX - ALLIANCE_ZONE_LANDING_MARGIN_METERS;

    // Keep the return trajectory beside the hub instead of sending it through the structure.
    double sideOffset = hubWidth / 2.0 + HUB_SIDE_CLEARANCE_METERS;
    double targetY =
        robotPose.getY() >= fieldCenterY ? fieldCenterY + sideOffset : fieldCenterY - sideOffset;
    targetY = MathUtil.clamp(targetY, 0.0, fieldWidth);

    return new Target(new Translation2d(targetX, targetY), true);
  }

  static boolean isBetweenHubs(double fieldX) {
    return isBetweenHubs(
        fieldX, FieldConstants.LinesVertical.hubCenter, FieldConstants.LinesVertical.oppHubCenter);
  }

  private static boolean isBetweenHubs(double fieldX, double blueHubX, double redHubX) {
    double firstHubX = Math.min(blueHubX, redHubX);
    double secondHubX = Math.max(blueHubX, redHubX);
    return fieldX > firstHubX && fieldX < secondHubX;
  }
}
