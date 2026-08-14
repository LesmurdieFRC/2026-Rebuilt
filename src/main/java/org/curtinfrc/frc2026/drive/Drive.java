package org.curtinfrc.frc2026.drive;

import static edu.wpi.first.units.Units.*;

import choreo.trajectory.SwerveSample;
import com.ctre.phoenix6.CANBus;
import com.ctre.phoenix6.configs.CANcoderConfiguration;
import com.ctre.phoenix6.configs.TalonFXConfiguration;
import com.ctre.phoenix6.swerve.SwerveModuleConstants;
import edu.wpi.first.hal.FRCNetComm.tInstances;
import edu.wpi.first.hal.FRCNetComm.tResourceType;
import edu.wpi.first.hal.HAL;
import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.Matrix;
import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.estimator.SwerveDrivePoseEstimator;
import edu.wpi.first.math.filter.SlewRateLimiter;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Transform2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.geometry.Twist2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.math.kinematics.SwerveDriveKinematics;
import edu.wpi.first.math.kinematics.SwerveModulePosition;
import edu.wpi.first.math.kinematics.SwerveModuleState;
import edu.wpi.first.math.numbers.N1;
import edu.wpi.first.math.numbers.N3;
import edu.wpi.first.units.measure.LinearVelocity;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj.DriverStation;
import edu.wpi.first.wpilibj.DriverStation.Alliance;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.DoubleSupplier;
import org.curtinfrc.frc2026.Constants;
import org.curtinfrc.frc2026.Constants.Mode;
import org.curtinfrc.frc2026.util.FieldConstants;
import org.littletonrobotics.junction.AutoLogOutput;
import org.littletonrobotics.junction.Logger;

public class Drive extends SubsystemBase {
  // TunerConstants doesn't include these constants, so they are declared locally
  static double ODOMETRY_FREQUENCY =
      new CANBus(TunerConstants.DrivetrainConstants.CANBusName).isNetworkFD() ? 250.0 : 100.0;
  private static final double ANGLE_MAX_ACCELERATION = 20.0;
  private static final double WHEEL_RADIUS_MAX_VELOCITY = 0.25; // Rad/Sec
  private static final double WHEEL_RADIUS_RAMP_RATE = 0.05; // Rad/Sec^2
  private static final double SHOT_ALIGNMENT_RADIUS_METERS = 2.0;
  private static final double JOYSTICK_TRANSLATION_SLEW_RATE_PER_SECOND = 2.5;
  private static final double JOYSTICK_ROTATION_SLEW_RATE_PER_SECOND = 3.0;
  private static final double AIM_MAX_LINEAR_SPEED_METERS_PER_SECOND = 2.0;
  private static final double AIM_MAX_ANGULAR_SPEED_RADIANS_PER_SECOND = 3.0;
  private static final double AIM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED = 2.5;
  private static final double AIM_ANGULAR_ACCELERATION_RADIANS_PER_SECOND_SQUARED = 4.0;

  static final Lock odometryLock = new ReentrantLock();
  private final Config config;
  private final double driveBaseRadius;
  private final GyroIO gyroIO;
  private final GyroIOInputsAutoLogged gyroInputs = new GyroIOInputsAutoLogged();
  private final Module[] modules = new Module[4]; // FL, FR, BL, BR
  private final Alert gyroDisconnectedAlert =
      new Alert("Disconnected gyro, using kinematics as fallback.", AlertType.kError);

  private final SwerveDriveKinematics kinematics;
  private Rotation2d rawGyroRotation = Rotation2d.kZero;
  private SwerveModulePosition[] lastModulePositions = // For delta tracking
      new SwerveModulePosition[] {
        new SwerveModulePosition(),
        new SwerveModulePosition(),
        new SwerveModulePosition(),
        new SwerveModulePosition()
      };
  private final SwerveDrivePoseEstimator poseEstimator;
  private final PIDController xController = new PIDController(5.0, 0.0, 0.0);
  private final PIDController yController = new PIDController(5.0, 0.0, 0.0);
  private final PIDController headingController = new PIDController(7.5, 0.0, 0.0);

  public static void configureOdometryFrequency(CANBus canBus) {
    ODOMETRY_FREQUENCY = canBus.isNetworkFD() ? 250.0 : 100.0;
  }

  public Drive(
      GyroIO gyroIO,
      ModuleIO flModuleIO,
      ModuleIO frModuleIO,
      ModuleIO blModuleIO,
      ModuleIO brModuleIO,
      Config config) {
    this.config = config;
    this.gyroIO = gyroIO;
    modules[0] = new Module(flModuleIO, 0, config.frontLeft());
    modules[1] = new Module(frModuleIO, 1, config.frontRight());
    modules[2] = new Module(blModuleIO, 2, config.backLeft());
    modules[3] = new Module(brModuleIO, 3, config.backRight());
    Translation2d[] moduleTranslations = getModuleTranslations();
    kinematics = new SwerveDriveKinematics(moduleTranslations);
    driveBaseRadius =
        Math.max(
            Math.max(moduleTranslations[0].getNorm(), moduleTranslations[1].getNorm()),
            Math.max(moduleTranslations[2].getNorm(), moduleTranslations[3].getNorm()));
    poseEstimator =
        new SwerveDrivePoseEstimator(
            kinematics, rawGyroRotation, lastModulePositions, Pose2d.kZero);
    headingController.enableContinuousInput(-Math.PI, Math.PI);
    // Usage reporting for swerve template
    HAL.report(tResourceType.kResourceType_RobotDrive, tInstances.kRobotDriveSwerve_AdvantageKit);

    // Start odometry thread
    PhoenixOdometryThread.getInstance().start();
  }

  public void followTrajectory(SwerveSample sample) {
    // Get the current pose of the robot
    Pose2d pose = getPose();
    Logger.recordOutput("Autonomous/Trajectory/ElapsedSeconds", sample.t);
    Logger.recordOutput("Autonomous/Trajectory/Setpoint", sample.getPose());

    // Generate the next speeds for the robot
    ChassisSpeeds fieldRelativeSpeeds =
        new ChassisSpeeds(
            sample.vx + xController.calculate(pose.getX(), sample.x),
            sample.vy + yController.calculate(pose.getY(), sample.y),
            sample.omega
                + headingController.calculate(pose.getRotation().getRadians(), sample.heading));

    // Choreo samples and x/y feedback are field-relative; module setpoints are robot-relative.
    runVelocity(ChassisSpeeds.fromFieldRelativeSpeeds(fieldRelativeSpeeds, pose.getRotation()));
  }

  @Override
  public void periodic() {
    odometryLock.lock(); // Prevents odometry updates while reading data
    gyroIO.updateInputs(gyroInputs);
    Logger.processInputs("Drive/Gyro", gyroInputs);
    for (var module : modules) {
      module.periodic();
    }
    odometryLock.unlock();

    // Stop moving when disabled
    if (DriverStation.isDisabled()) {
      for (var module : modules) {
        module.stop();
      }
    }

    // Log empty setpoint states when disabled
    if (DriverStation.isDisabled()) {
      Logger.recordOutput("SwerveStates/Setpoints", new SwerveModuleState[] {});
      Logger.recordOutput("SwerveStates/SetpointsOptimized", new SwerveModuleState[] {});
    }

    // Update odometry
    double[] sampleTimestamps =
        modules[0].getOdometryTimestamps(); // All signals are sampled together
    int sampleCount = sampleTimestamps.length;
    for (int i = 0; i < sampleCount; i++) {
      // Read wheel positions and deltas from each module
      SwerveModulePosition[] modulePositions = new SwerveModulePosition[4];
      SwerveModulePosition[] moduleDeltas = new SwerveModulePosition[4];
      for (int moduleIndex = 0; moduleIndex < 4; moduleIndex++) {
        modulePositions[moduleIndex] = modules[moduleIndex].getOdometryPositions()[i];
        moduleDeltas[moduleIndex] =
            new SwerveModulePosition(
                modulePositions[moduleIndex].distanceMeters
                    - lastModulePositions[moduleIndex].distanceMeters,
                modulePositions[moduleIndex].angle);
        lastModulePositions[moduleIndex] = modulePositions[moduleIndex];
      }

      // Update gyro angle
      if (gyroInputs.connected) {
        // Use the real gyro angle
        rawGyroRotation = gyroInputs.odometryYawPositions[i];
      } else {
        // Use the angle delta from the kinematics and module deltas
        Twist2d twist = kinematics.toTwist2d(moduleDeltas);
        rawGyroRotation = rawGyroRotation.plus(new Rotation2d(twist.dtheta));
      }

      // Apply update
      poseEstimator.updateWithTime(sampleTimestamps[i], rawGyroRotation, modulePositions);
    }

    // Update gyro alert
    gyroDisconnectedAlert.set(!gyroInputs.connected && Constants.getMode() != Mode.SIM);
  }

  /**
   * Runs the drive at the desired velocity.
   *
   * @param speeds Speeds in meters/sec
   */
  private void runVelocity(ChassisSpeeds speeds) {
    // Calculate module setpoints
    ChassisSpeeds discreteSpeeds = ChassisSpeeds.discretize(speeds, 0.02);
    SwerveModuleState[] setpointStates = kinematics.toSwerveModuleStates(discreteSpeeds);
    SwerveDriveKinematics.desaturateWheelSpeeds(setpointStates, config.speedAt12Volts());

    // Log unoptimized setpoints and setpoint speeds
    Logger.recordOutput("SwerveStates/Setpoints", setpointStates);
    Logger.recordOutput("SwerveChassisSpeeds/Setpoints", discreteSpeeds);

    // Send setpoints to modules
    for (int i = 0; i < 4; i++) {
      modules[i].runSetpoint(setpointStates[i]);
    }

    // Log optimized setpoints (runSetpoint mutates each state)
    Logger.recordOutput("SwerveStates/SetpointsOptimized", setpointStates);
  }

  /** Stops the drive. */
  private void stop() {
    runVelocity(new ChassisSpeeds());
  }

  /** Returns the module states (turn angles and drive velocities) for all of the modules. */
  @AutoLogOutput(key = "SwerveStates/Measured")
  private SwerveModuleState[] getModuleStates() {
    SwerveModuleState[] states = new SwerveModuleState[4];
    for (int i = 0; i < 4; i++) {
      states[i] = modules[i].getState();
    }
    return states;
  }

  /** Returns the module positions (turn angles and drive positions) for all of the modules. */
  private SwerveModulePosition[] getModulePositions() {
    SwerveModulePosition[] states = new SwerveModulePosition[4];
    for (int i = 0; i < 4; i++) {
      states[i] = modules[i].getPosition();
    }
    return states;
  }

  /** Returns the measured chassis speeds of the robot. */
  @AutoLogOutput(key = "SwerveChassisSpeeds/Measured")
  private ChassisSpeeds getChassisSpeeds() {
    return kinematics.toChassisSpeeds(getModuleStates());
  }

  /** True once translational and angular motion are low enough to shoot safely. */
  public boolean isStopped() {
    ChassisSpeeds speeds = getChassisSpeeds();
    return Math.hypot(speeds.vxMetersPerSecond, speeds.vyMetersPerSecond) < 0.10
        && Math.abs(speeds.omegaRadiansPerSecond) < 0.20;
  }

  /** Returns the position of each module in radians. */
  public double[] getWheelRadiusCharacterizationPositions() {
    double[] values = new double[4];
    for (int i = 0; i < 4; i++) {
      values[i] = modules[i].getWheelRadiusCharacterizationPosition();
    }
    return values;
  }

  /** Returns the current odometry pose. */
  @AutoLogOutput(key = "Odometry/Robot")
  public Pose2d getPose() {
    return poseEstimator.getEstimatedPosition();
  }

  /** Returns the current odometry rotation. */
  public Rotation2d getRotation() {
    return getPose().getRotation();
  }

  /** Resets the current odometry pose. */
  public void setPose(Pose2d pose) {
    poseEstimator.resetPosition(rawGyroRotation, getModulePositions(), pose);
  }

  /** Adds a new timestamped vision measurement. */
  public void addVisionMeasurement(
      Pose2d visionRobotPoseMeters,
      double timestampSeconds,
      Matrix<N3, N1> visionMeasurementStdDevs) {
    poseEstimator.addVisionMeasurement(
        visionRobotPoseMeters, timestampSeconds, visionMeasurementStdDevs);
  }

  /** Returns the maximum linear speed in meters per sec. */
  public double getMaxLinearSpeedMetersPerSec() {
    return config.speedAt12Volts().in(MetersPerSecond);
  }

  /** Returns the maximum angular speed in radians per sec. */
  public double getMaxAngularSpeedRadPerSec() {
    return getMaxLinearSpeedMetersPerSec() / driveBaseRadius;
  }

  /** Returns an array of module translations. */
  public Translation2d[] getModuleTranslations() {
    return new Translation2d[] {
      new Translation2d(config.frontLeft().LocationX, config.frontLeft().LocationY),
      new Translation2d(config.frontRight().LocationX, config.frontRight().LocationY),
      new Translation2d(config.backLeft().LocationX, config.backLeft().LocationY),
      new Translation2d(config.backRight().LocationX, config.backRight().LocationY)
    };
  }

  public record Config(
      SwerveModuleConstants<TalonFXConfiguration, TalonFXConfiguration, CANcoderConfiguration>
          frontLeft,
      SwerveModuleConstants<TalonFXConfiguration, TalonFXConfiguration, CANcoderConfiguration>
          frontRight,
      SwerveModuleConstants<TalonFXConfiguration, TalonFXConfiguration, CANcoderConfiguration>
          backLeft,
      SwerveModuleConstants<TalonFXConfiguration, TalonFXConfiguration, CANcoderConfiguration>
          backRight,
      LinearVelocity speedAt12Volts) {}

  private static Translation2d getLinearVelocityFromJoysticks(double x, double y) {
    // Apply deadband
    double linearMagnitude = Math.hypot(x, y);
    Rotation2d linearDirection = new Rotation2d(Math.atan2(y, x));

    // Square magnitude for more precise control
    linearMagnitude = linearMagnitude * linearMagnitude;

    // Return new linear velocity
    return new Pose2d(Translation2d.kZero, linearDirection)
        .transformBy(new Transform2d(linearMagnitude, 0.0, Rotation2d.kZero))
        .getTranslation();
  }

  /**
   * Field relative drive command using two joysticks (controlling linear and angular velocities).
   */
  public Command joystickDrive(
      DoubleSupplier xSupplier, DoubleSupplier ySupplier, DoubleSupplier omegaSupplier) {
    SlewRateLimiter xLimiter = new SlewRateLimiter(JOYSTICK_TRANSLATION_SLEW_RATE_PER_SECOND);
    SlewRateLimiter yLimiter = new SlewRateLimiter(JOYSTICK_TRANSLATION_SLEW_RATE_PER_SECOND);
    SlewRateLimiter omegaLimiter = new SlewRateLimiter(JOYSTICK_ROTATION_SLEW_RATE_PER_SECOND);
    return run(() -> {
          // Get linear velocity
          Translation2d linearVelocity =
              getLinearVelocityFromJoysticks(
                  xLimiter.calculate(xSupplier.getAsDouble()),
                  yLimiter.calculate(ySupplier.getAsDouble()));

          // Apply rotation deadband
          double omega = omegaLimiter.calculate(omegaSupplier.getAsDouble());

          // Square rotation value for more precise control
          omega = Math.copySign(omega * omega, omega);

          // Convert to field relative speeds & send command
          ChassisSpeeds speeds =
              new ChassisSpeeds(
                  linearVelocity.getX() * getMaxLinearSpeedMetersPerSec(),
                  linearVelocity.getY() * getMaxLinearSpeedMetersPerSec(),
                  omega * getMaxAngularSpeedRadPerSec());
          boolean isFlipped =
              DriverStation.getAlliance().isPresent()
                  && DriverStation.getAlliance().get() == Alliance.Red;
          runVelocity(
              ChassisSpeeds.fromFieldRelativeSpeeds(
                  speeds, isFlipped ? getRotation().plus(new Rotation2d(Math.PI)) : getRotation()));
        })
        .beforeStarting(
            () -> {
              xLimiter.reset(0.0);
              yLimiter.reset(0.0);
              omegaLimiter.reset(0.0);
            });
  }

  public Command hubCommand() {
    return shotAimCommand();
  }

  /** Turns the rear-facing shooter toward its current hub or alliance-side return target. */
  public Command shotAimCommand() {
    SlewRateLimiter xSpeedLimiter =
        new SlewRateLimiter(AIM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED);
    SlewRateLimiter ySpeedLimiter =
        new SlewRateLimiter(AIM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED);
    SlewRateLimiter omegaLimiter =
        new SlewRateLimiter(AIM_ANGULAR_ACCELERATION_RADIANS_PER_SECOND_SQUARED);
    return run(() -> {
          Pose2d currentPose = getPose();
          ShotTargeting.Target shotTarget = getShotTarget();
          Translation2d targetPosition = shotTarget.position();
          Rotation2d targetHeading =
              HubAiming.headingToTarget(currentPose.getTranslation(), targetPosition)
                  .rotateBy(new Rotation2d(Math.PI));
          Translation2d alignmentPosition =
              HubAiming.nearestPointOnCircle(
                  currentPose.getTranslation(), targetPosition, SHOT_ALIGNMENT_RADIUS_METERS);

          double turnSpeed =
              omegaLimiter.calculate(
                  MathUtil.clamp(
                      headingController.calculate(
                          currentPose.getRotation().getRadians(), targetHeading.getRadians()),
                      -AIM_MAX_ANGULAR_SPEED_RADIANS_PER_SECOND,
                      AIM_MAX_ANGULAR_SPEED_RADIANS_PER_SECOND));
          double xSpeed =
              xSpeedLimiter.calculate(
                  MathUtil.clamp(
                      xController.calculate(currentPose.getX(), alignmentPosition.getX()),
                      -AIM_MAX_LINEAR_SPEED_METERS_PER_SECOND,
                      AIM_MAX_LINEAR_SPEED_METERS_PER_SECOND));
          double ySpeed =
              ySpeedLimiter.calculate(
                  MathUtil.clamp(
                      yController.calculate(currentPose.getY(), alignmentPosition.getY()),
                      -AIM_MAX_LINEAR_SPEED_METERS_PER_SECOND,
                      AIM_MAX_LINEAR_SPEED_METERS_PER_SECOND));
          Logger.recordOutput("Drive/ShotAim/Target", targetPosition);
          Logger.recordOutput("Drive/ShotAim/AlignmentPosition", alignmentPosition);
          Logger.recordOutput("Drive/ShotAim/TargetHeading", targetHeading);
          Logger.recordOutput(
              "Drive/ShotAim/ReturningToAllianceSide", shotTarget.returnsToAllianceSide());
          runVelocity(
              ChassisSpeeds.fromFieldRelativeSpeeds(
                  new ChassisSpeeds(xSpeed, ySpeed, turnSpeed), currentPose.getRotation()));
        })
        .beforeStarting(
            () -> {
              xSpeedLimiter.reset(0.0);
              ySpeedLimiter.reset(0.0);
              omegaLimiter.reset(0.0);
            });
  }

  /** Returns the center of the scoring hub belonging to the current alliance. */
  public static Translation2d getAllianceHubPosition() {
    return FieldConstants.Hub.getAllianceCenter(DriverStation.getAlliance().orElse(Alliance.Blue));
  }

  /** Returns this robot's current planar distance from the center of its alliance hub. */
  public double getDistanceToAllianceHub() {
    return getPose().getTranslation().getDistance(getAllianceHubPosition());
  }

  /** Returns the active shot target, mirrored for alliance and midfield position. */
  private ShotTargeting.Target getShotTarget() {
    return ShotTargeting.targetFor(getPose(), DriverStation.getAlliance().orElse(Alliance.Blue));
  }

  /** Returns the planar distance to the active hub or alliance-side floor landing target. */
  public double getDistanceToShotTarget() {
    return getPose().getTranslation().getDistance(getShotTarget().position());
  }

  /** True when midfield shots should return FUEL to the floor on this robot's alliance side. */
  public boolean isReturningFuelToAllianceSide() {
    return getShotTarget().returnsToAllianceSide();
  }
}
