package org.curtinfrc.frc2026.drive;

import com.ctre.phoenix6.CANBus;
import com.ctre.phoenix6.StatusSignal;
import com.ctre.phoenix6.configs.Pigeon2Configuration;
import com.ctre.phoenix6.hardware.Pigeon2;
import com.ctre.phoenix6.swerve.SwerveDrivetrainConstants;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.util.Units;
import edu.wpi.first.units.measure.Angle;
import edu.wpi.first.units.measure.AngularVelocity;
import java.util.Queue;
import org.curtinfrc.frc2026.util.PhoenixUtil;

/** IO implementation for Pigeon 2. */
public class GyroIOPigeon2 implements GyroIO {
  private final Pigeon2 pigeon;
  private final StatusSignal<Angle> yaw;
  private final StatusSignal<Angle> pitch;
  private final StatusSignal<Angle> roll;
  private final Queue<Double> yawPositionQueue;
  private final Queue<Double> yawTimestampQueue;
  private final StatusSignal<AngularVelocity> yawVelocity;
  private final double mountPoseYawDegrees;
  private final double mountPosePitchDegrees;
  private final double mountPoseRollDegrees;

  public GyroIOPigeon2(SwerveDrivetrainConstants constants) {
    pigeon = new Pigeon2(constants.Pigeon2Id, new CANBus(constants.CANBusName));
    yaw = pigeon.getYaw();
    pitch = pigeon.getPitch();
    roll = pigeon.getRoll();
    yawVelocity = pigeon.getAngularVelocityZWorld();
    if (constants.Pigeon2Configs != null) {
      pigeon.getConfigurator().apply(constants.Pigeon2Configs);
    }

    // When no configuration is supplied, preserve the mount calibration saved by Phoenix Tuner X
    // instead of replacing it with a zero-angle default configuration on every robot startup.
    Pigeon2Configuration appliedConfiguration = new Pigeon2Configuration();
    pigeon.getConfigurator().refresh(appliedConfiguration);
    mountPoseYawDegrees = appliedConfiguration.MountPose.MountPoseYaw;
    mountPosePitchDegrees = appliedConfiguration.MountPose.MountPosePitch;
    mountPoseRollDegrees = appliedConfiguration.MountPose.MountPoseRoll;

    pigeon.getConfigurator().setYaw(0.0);
    yaw.setUpdateFrequency(Drive.ODOMETRY_FREQUENCY);
    pitch.setUpdateFrequency(50.0);
    roll.setUpdateFrequency(50.0);
    yawVelocity.setUpdateFrequency(50.0);
    pigeon.optimizeBusUtilization();
    yawTimestampQueue = PhoenixOdometryThread.getInstance().makeTimestampQueue();
    yawPositionQueue = PhoenixOdometryThread.getInstance().registerSignal(yaw.clone());

    PhoenixUtil.registerSignals(true, yaw, pitch, roll, yawVelocity);
  }

  @Override
  public void updateInputs(GyroIOInputs inputs) {
    inputs.connected =
        yaw.getStatus().isOK()
            && pitch.getStatus().isOK()
            && roll.getStatus().isOK()
            && yawVelocity.getStatus().isOK();
    inputs.yawPosition = Rotation2d.fromDegrees(yaw.getValueAsDouble());
    inputs.pitchPosition = Rotation2d.fromDegrees(pitch.getValueAsDouble());
    inputs.rollPosition = Rotation2d.fromDegrees(roll.getValueAsDouble());
    inputs.yawVelocityRadPerSec = Units.degreesToRadians(yawVelocity.getValueAsDouble());
    inputs.mountPoseYawDegrees = mountPoseYawDegrees;
    inputs.mountPosePitchDegrees = mountPosePitchDegrees;
    inputs.mountPoseRollDegrees = mountPoseRollDegrees;

    inputs.odometryYawTimestamps =
        yawTimestampQueue.stream().mapToDouble((Double value) -> value).toArray();
    inputs.odometryYawPositions =
        yawPositionQueue.stream()
            .map((Double value) -> Rotation2d.fromDegrees(value))
            .toArray(Rotation2d[]::new);
    yawTimestampQueue.clear();
    yawPositionQueue.clear();
  }
}
