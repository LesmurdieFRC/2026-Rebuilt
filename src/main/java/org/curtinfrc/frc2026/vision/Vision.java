package org.curtinfrc.frc2026.vision;

import edu.wpi.first.apriltag.AprilTagFieldLayout;
import edu.wpi.first.apriltag.AprilTagFields;
import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.Matrix;
import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;
import edu.wpi.first.math.interpolation.TimeInterpolatableBuffer;
import edu.wpi.first.math.numbers.N1;
import edu.wpi.first.math.numbers.N3;
import edu.wpi.first.wpilibj.Timer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Supplier;
import org.curtinfrc.frc2026.util.VirtualSubsystem;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.networktables.LoggedNetworkNumber;
import org.photonvision.common.hardware.VisionLEDMode;

public class Vision extends VirtualSubsystem {
  public static AprilTagFieldLayout aprilTagLayout =
      AprilTagFieldLayout.loadField(AprilTagFields.kDefaultField);

  public static record CameraConfig(
      String name,
      Transform3d robotToCamera,
      double linearStdDevMultiplier,
      double angularStdDevMultiplier) {}

  // Transform3d uses robot-to-camera coordinates and roll, pitch, yaw rotations. Keep these
  // explicit because an incorrect camera roll/yaw creates a consistent heading residual in the
  // logs below. Positive X is forward, positive Y is left, and positive Z is up.
  private static final Rotation3d REAR_HORIZONTAL_CAMERA_ROTATION =
      new Rotation3d(0.0, 0.0, Math.toRadians(180.0));
  private static final Rotation3d FRONT_VERTICAL_LEFT_IS_DOWN_ROTATION =
      new Rotation3d(Math.toRadians(-90.0), 0.0, 0.0);
  private static final Rotation3d FRONT_VERTICAL_LEFT_IS_UP_ROTATION =
      new Rotation3d(Math.toRadians(90.0), 0.0, 0.0);

  public static CameraConfig[] cameraConfigs =
      new CameraConfig[] {
        new CameraConfig(
            "Camera0",
            new Transform3d(new Translation3d(-0.28, 0.25, 0.53), REAR_HORIZONTAL_CAMERA_ROTATION),
            2.0,
            3.0),
        new CameraConfig(
            "Camera2",
            new Transform3d(new Translation3d(-0.28, -0.25, 0.53), REAR_HORIZONTAL_CAMERA_ROTATION),
            1.0,
            1.0),
        new CameraConfig(
            "Camera3",
            new Transform3d(
                new Translation3d(-0.32, 0.23, 0.51), FRONT_VERTICAL_LEFT_IS_DOWN_ROTATION),
            1.0,
            1.0),
        new CameraConfig(
            "Camera1",
            new Transform3d(
                new Translation3d(0.32, -0.23, 0.51), FRONT_VERTICAL_LEFT_IS_UP_ROTATION),
            1.0,
            1.0),
      };

  private static final LoggedNetworkNumber maxSingleTagAmbiguity =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Max Single Tag Ambiguity", 0.30);
  private static final LoggedNetworkNumber maxZErrorMeters =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Max Z Error Meters", 0.4);
  private static final LoggedNetworkNumber linearStdDevBaselineMeters =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Linear Std Dev Baseline M", 0.04);
  private static final LoggedNetworkNumber angularStdDevBaselineRadians =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Angular Std Dev Baseline Rad", 0.10);
  private static final LoggedNetworkNumber ambiguityStdDevScale =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Ambiguity Std Dev Scale", 3.0);
  private static final LoggedNetworkNumber maxCameraConsensusErrorMeters =
      new LoggedNetworkNumber("/SmartDashboard/Vision Tuning/Max Camera Consensus Error M", 0.75);
  private static final LoggedNetworkNumber[] cameraLinearStdDevMultipliers =
      createCameraStdDevTunables("Linear Multiplier", true);
  private static final LoggedNetworkNumber[] cameraAngularStdDevMultipliers =
      createCameraStdDevTunables("Angular Multiplier", false);

  private static final double MIN_LINEAR_STD_DEV_METERS = 0.08;
  private static final double MAX_LINEAR_STD_DEV_METERS = 1.5;
  private static final double MIN_ANGULAR_STD_DEV_RADIANS = 0.10;
  private static final double MAX_ANGULAR_STD_DEV_RADIANS = 1.0;
  private static final double ACTIVE_CAMERA_LOSS_TIMEOUT_SECONDS = 0.25;
  // A single AprilTag does not provide a heading measurement reliable enough to beat the gyro.
  private static final double SINGLE_TAG_ANGULAR_STD_DEV_RADIANS = 1.0e6;

  private final PoseEstimateConsumer consumer;
  private final Supplier<Rotation2d> gyro;
  private final TimeInterpolatableBuffer<Rotation2d> headingBuffer =
      TimeInterpolatableBuffer.createBuffer(1.0);
  private final VisionIO[] io;
  private final VisionIOInputsAutoLogged[] inputs;
  private boolean fusionEnabled = true;
  private int activeFusionCameraIndex = -1;
  private double activeFusionCameraLastSeenSeconds = Double.NEGATIVE_INFINITY;

  private static LoggedNetworkNumber[] createCameraStdDevTunables(
      String tuningName, boolean linear) {
    LoggedNetworkNumber[] tunables = new LoggedNetworkNumber[cameraConfigs.length];
    for (int i = 0; i < cameraConfigs.length; i++) {
      CameraConfig config = cameraConfigs[i];
      tunables[i] =
          new LoggedNetworkNumber(
              "/SmartDashboard/Vision Tuning/Camera " + config.name() + "/" + tuningName,
              linear ? config.linearStdDevMultiplier() : config.angularStdDevMultiplier());
    }
    return tunables;
  }

  private static boolean passesBasicFilters(VisionIO.PoseObservation observation) {
    Pose3d pose = observation.pose();
    return observation.tagCount() > 0
        && Math.abs(pose.getZ()) <= maxZErrorMeters.get()
        && pose.getX() >= 0.0
        && pose.getX() <= aprilTagLayout.getFieldLength()
        && pose.getY() >= 0.0
        && pose.getY() <= aprilTagLayout.getFieldWidth()
        && (observation.tagCount() != 1 || observation.ambiguity() <= maxSingleTagAmbiguity.get());
  }

  private static double median(List<Double> values) {
    double[] sorted = values.stream().mapToDouble(Double::doubleValue).toArray();
    Arrays.sort(sorted);
    int middle = sorted.length / 2;
    return sorted.length % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2.0 : sorted[middle];
  }

  public Vision(PoseEstimateConsumer consumer, Supplier<Rotation2d> gyro, VisionIO... io) {
    this.consumer = consumer;
    this.gyro = gyro;
    this.io = io;

    // Initialize inputs
    this.inputs = new VisionIOInputsAutoLogged[io.length];
    for (int i = 0; i < inputs.length; i++) {
      inputs[i] = new VisionIOInputsAutoLogged();
    }
  }

  public void setLEDMode(VisionLEDMode mode) {
    for (var i : io) {
      i.setLEDMode(mode);
    }
  }

  /** Enables or disables adding camera pose observations to drivetrain odometry. */
  public void setFusionEnabled(boolean enabled) {
    fusionEnabled = enabled;
  }

  public boolean isFusionEnabled() {
    return fusionEnabled;
  }

  @Override
  public void periodic() {
    Logger.recordOutput("Vision/FusionEnabled", fusionEnabled);

    // Update heading data
    headingBuffer.addSample(Timer.getFPGATimestamp(), gyro.get());

    for (int i = 0; i < io.length; i++) {
      io[i].updateInputs(inputs[i]);
      Logger.processInputs("Vision/Camera" + Integer.toString(i), inputs[i]);
    }

    // Use each camera's newest valid sample to measure disagreement between camera transforms.
    // The median is robust to one badly calibrated camera when three or more cameras are visible.
    VisionIO.PoseObservation[] latestCameraObservations = new VisionIO.PoseObservation[io.length];
    Pose3d[] latestCameraPoses = new Pose3d[io.length];
    List<Double> latestCameraXs = new ArrayList<>();
    List<Double> latestCameraYs = new ArrayList<>();
    for (int cameraIndex = 0; cameraIndex < io.length; cameraIndex++) {
      double newestTimestamp = Double.NEGATIVE_INFINITY;
      for (var observation : inputs[cameraIndex].poseObservations) {
        if (passesBasicFilters(observation) && observation.timestamp() > newestTimestamp) {
          newestTimestamp = observation.timestamp();
          latestCameraObservations[cameraIndex] = observation;
          latestCameraPoses[cameraIndex] = observation.pose();
        }
      }
      if (latestCameraPoses[cameraIndex] != null) {
        latestCameraXs.add(latestCameraPoses[cameraIndex].getX());
        latestCameraYs.add(latestCameraPoses[cameraIndex].getY());
      }
    }

    int visibleCameraCount = latestCameraXs.size();
    double consensusX = visibleCameraCount > 0 ? median(latestCameraXs) : Double.NaN;
    double consensusY = visibleCameraCount > 0 ? median(latestCameraYs) : Double.NaN;
    double[] cameraConsensusErrors = new double[io.length];
    Arrays.fill(cameraConsensusErrors, Double.NaN);
    for (int cameraIndex = 0; cameraIndex < io.length; cameraIndex++) {
      if (latestCameraPoses[cameraIndex] != null) {
        cameraConsensusErrors[cameraIndex] =
            Math.hypot(
                latestCameraPoses[cameraIndex].getX() - consensusX,
                latestCameraPoses[cameraIndex].getY() - consensusY);
      }
    }
    Logger.recordOutput("Vision/Summary/VisibleCameraCount", visibleCameraCount);
    Logger.recordOutput("Vision/Summary/ConsensusX", consensusX);
    Logger.recordOutput("Vision/Summary/ConsensusY", consensusY);

    // Keep one camera as the fusion source instead of allowing imperfect transforms from several
    // cameras to pull odometry back and forth. Retain the current source through short frame gaps,
    // and switch only when it is stale or is a clear consensus outlier.
    double nowSeconds = Timer.getFPGATimestamp();
    boolean activeCameraHasObservation =
        activeFusionCameraIndex >= 0
            && activeFusionCameraIndex < latestCameraObservations.length
            && latestCameraObservations[activeFusionCameraIndex] != null;
    boolean activeCameraIsOutlier =
        activeCameraHasObservation
            && visibleCameraCount >= 3
            && cameraConsensusErrors[activeFusionCameraIndex] > maxCameraConsensusErrorMeters.get();
    if (activeCameraHasObservation && !activeCameraIsOutlier) {
      activeFusionCameraLastSeenSeconds = nowSeconds;
    } else if (activeCameraIsOutlier
        || nowSeconds - activeFusionCameraLastSeenSeconds > ACTIVE_CAMERA_LOSS_TIMEOUT_SECONDS) {
      activeFusionCameraIndex = -1;
      double bestScore = Double.POSITIVE_INFINITY;
      for (int cameraIndex = 0; cameraIndex < latestCameraObservations.length; cameraIndex++) {
        var observation = latestCameraObservations[cameraIndex];
        if (observation == null
            || (visibleCameraCount >= 3
                && cameraConsensusErrors[cameraIndex] > maxCameraConsensusErrorMeters.get())) {
          continue;
        }
        double configuredMultiplier =
            cameraIndex < cameraLinearStdDevMultipliers.length
                ? Math.max(0.0, cameraLinearStdDevMultipliers[cameraIndex].get())
                : 1.0;
        double score =
            cameraConsensusErrors[cameraIndex]
                + 0.05 * configuredMultiplier
                + 0.02 * Math.pow(observation.averageTagDistance(), 2.0) / observation.tagCount();
        if (score < bestScore) {
          bestScore = score;
          activeFusionCameraIndex = cameraIndex;
        }
      }
      if (activeFusionCameraIndex >= 0) {
        activeFusionCameraLastSeenSeconds = nowSeconds;
      }
    }
    Logger.recordOutput("Vision/Summary/ActiveFusionCameraIndex", activeFusionCameraIndex);
    Logger.recordOutput(
        "Vision/Summary/ActiveFusionCameraName",
        activeFusionCameraIndex >= 0 && activeFusionCameraIndex < cameraConfigs.length
            ? cameraConfigs[activeFusionCameraIndex].name()
            : "None");

    // Initialize logging values
    List<Pose3d> allTagPoses = new LinkedList<>();
    List<Pose3d> allRobotPoses = new LinkedList<>();
    List<Pose3d> allRobotPosesAccepted = new LinkedList<>();
    List<Pose3d> allRobotPosesRejected = new LinkedList<>();
    // Loop over cameras
    for (int cameraIndex = 0; cameraIndex < io.length; cameraIndex++) {
      // Initialize logging values
      List<Pose3d> tagPoses = new LinkedList<>();
      List<Pose3d> robotPoses = new LinkedList<>();
      List<Pose3d> robotPosesAccepted = new LinkedList<>();
      List<Pose3d> robotPosesRejected = new LinkedList<>();
      List<Double> headingErrorsDegrees = new LinkedList<>();
      List<Double> linearStdDevs = new LinkedList<>();
      List<Double> angularStdDevs = new LinkedList<>();

      // Add tag poses
      for (int tagId : inputs[cameraIndex].tagIds) {
        var tagPose = aprilTagLayout.getTagPose(tagId);
        if (tagPose.isPresent()) {
          tagPoses.add(tagPose.get());
        }
      }

      // Loop over pose observations
      for (var observation : inputs[cameraIndex].poseObservations) {
        var rotation = headingBuffer.getSample(observation.timestamp());
        var pose = observation.pose();
        double headingErrorDegrees =
            rotation
                .map(value -> Math.abs(pose.getRotation().toRotation2d().minus(value).getDegrees()))
                .orElse(Double.NaN);
        headingErrorsDegrees.add(headingErrorDegrees);

        // Check whether to reject pose
        boolean rejectPose =
            !passesBasicFilters(observation)
                // With at least three cameras, reject a camera that is far from the robust median.
                || (visibleCameraCount >= 3
                    && cameraConsensusErrors[cameraIndex] > maxCameraConsensusErrorMeters.get());

        // Add pose to log
        robotPoses.add(observation.pose());
        if (rejectPose) {
          robotPosesRejected.add(observation.pose());
        } else {
          robotPosesAccepted.add(observation.pose());
        }

        // Skip if rejected
        if (rejectPose) {
          continue;
        }

        // Calculate standard deviations
        double distanceFactor =
            Math.max(1.0, Math.pow(observation.averageTagDistance(), 2.0)) / observation.tagCount();
        double ambiguityFactor =
            observation.tagCount() == 1 && observation.ambiguity() > 0.0
                ? 1.0 + ambiguityStdDevScale.get() * observation.ambiguity()
                : 1.0;
        double linearStdDev = linearStdDevBaselineMeters.get() * distanceFactor * ambiguityFactor;
        double angularStdDev =
            observation.tagCount() == 1
                ? SINGLE_TAG_ANGULAR_STD_DEV_RADIANS
                : angularStdDevBaselineRadians.get() * distanceFactor;
        if (cameraIndex < cameraConfigs.length) {
          linearStdDev *= Math.max(0.0, cameraLinearStdDevMultipliers[cameraIndex].get());
          angularStdDev *= Math.max(0.0, cameraAngularStdDevMultipliers[cameraIndex].get());
        }
        // Camera errors are correlated, so N simultaneous cameras must not provide N times the
        // estimator weight. Also incorporate measured disagreement as systematic uncertainty.
        double correlatedCameraScale = Math.sqrt(Math.max(1, visibleCameraCount));
        linearStdDev *= correlatedCameraScale;
        angularStdDev *= correlatedCameraScale;
        if (!Double.isNaN(cameraConsensusErrors[cameraIndex])) {
          linearStdDev = Math.hypot(linearStdDev, cameraConsensusErrors[cameraIndex]);
        }
        linearStdDev =
            MathUtil.clamp(linearStdDev, MIN_LINEAR_STD_DEV_METERS, MAX_LINEAR_STD_DEV_METERS);
        if (observation.tagCount() > 1) {
          angularStdDev =
              MathUtil.clamp(
                  angularStdDev, MIN_ANGULAR_STD_DEV_RADIANS, MAX_ANGULAR_STD_DEV_RADIANS);
        }
        linearStdDevs.add(linearStdDev);
        angularStdDevs.add(angularStdDev);

        // Send vision observation
        if (fusionEnabled && cameraIndex == activeFusionCameraIndex) {
          consumer.accept(
              observation.pose().toPose2d(),
              observation.timestamp(),
              VecBuilder.fill(linearStdDev, linearStdDev, angularStdDev));
        }
      }

      // Log camera datadata
      Logger.recordOutput(
          "Vision/Camera" + Integer.toString(cameraIndex) + "/TagPoses",
          tagPoses.toArray(new Pose3d[tagPoses.size()]));
      Logger.recordOutput(
          "Vision/Camera" + Integer.toString(cameraIndex) + "/RobotPoses",
          robotPoses.toArray(new Pose3d[robotPoses.size()]));
      Logger.recordOutput(
          "Vision/Camera" + Integer.toString(cameraIndex) + "/RobotPosesAccepted",
          robotPosesAccepted.toArray(new Pose3d[robotPosesAccepted.size()]));
      Logger.recordOutput(
          "Vision/Camera" + Integer.toString(cameraIndex) + "/RobotPosesRejected",
          robotPosesRejected.toArray(new Pose3d[robotPosesRejected.size()]));
      Logger.recordOutput(
          "Vision/Camera" + cameraIndex + "/HeadingErrorsDegrees",
          headingErrorsDegrees.stream().mapToDouble(Double::doubleValue).toArray());
      Logger.recordOutput(
          "Vision/Camera" + cameraIndex + "/AcceptedLinearStdDevsMeters",
          linearStdDevs.stream().mapToDouble(Double::doubleValue).toArray());
      Logger.recordOutput(
          "Vision/Camera" + cameraIndex + "/AcceptedAngularStdDevsRadians",
          angularStdDevs.stream().mapToDouble(Double::doubleValue).toArray());
      Logger.recordOutput(
          "Vision/Camera" + cameraIndex + "/ConsensusErrorMeters",
          cameraConsensusErrors[cameraIndex]);
      Logger.recordOutput(
          "Vision/Camera" + cameraIndex + "/SelectedForFusion",
          cameraIndex == activeFusionCameraIndex);
      if (cameraIndex < cameraConfigs.length) {
        CameraConfig config = cameraConfigs[cameraIndex];
        Transform3d transform = config.robotToCamera();
        Logger.recordOutput("Vision/Camera" + cameraIndex + "/Config/Name", config.name());
        Logger.recordOutput(
            "Vision/Camera" + cameraIndex + "/Config/RobotToCamera",
            new Pose3d().transformBy(transform));
        Logger.recordOutput(
            "Vision/Camera" + cameraIndex + "/Config/RollDegrees",
            Math.toDegrees(transform.getRotation().getX()));
        Logger.recordOutput(
            "Vision/Camera" + cameraIndex + "/Config/PitchDegrees",
            Math.toDegrees(transform.getRotation().getY()));
        Logger.recordOutput(
            "Vision/Camera" + cameraIndex + "/Config/YawDegrees",
            Math.toDegrees(transform.getRotation().getZ()));
      }
      allTagPoses.addAll(tagPoses);
      allRobotPoses.addAll(robotPoses);
      allRobotPosesAccepted.addAll(robotPosesAccepted);
      allRobotPosesRejected.addAll(robotPosesRejected);
    }

    // Log summary data
    Logger.recordOutput(
        "Vision/Summary/TagPoses", allTagPoses.toArray(new Pose3d[allTagPoses.size()]));
    Logger.recordOutput(
        "Vision/Summary/RobotPoses", allRobotPoses.toArray(new Pose3d[allRobotPoses.size()]));
    Logger.recordOutput(
        "Vision/Summary/RobotPosesAccepted",
        allRobotPosesAccepted.toArray(new Pose3d[allRobotPosesAccepted.size()]));
    Logger.recordOutput(
        "Vision/Summary/RobotPosesRejected",
        allRobotPosesRejected.toArray(new Pose3d[allRobotPosesRejected.size()]));
  }

  @FunctionalInterface
  public static interface PoseEstimateConsumer {
    public void accept(
        Pose2d visionRobotPoseMeters,
        double timestampSeconds,
        Matrix<N3, N1> visionMeasurementStdDevs);
  }
}
