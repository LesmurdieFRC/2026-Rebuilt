package org.curtinfrc.frc2026.drive.Superstructure;

import com.revrobotics.PersistMode;
import com.revrobotics.RelativeEncoder;
import com.revrobotics.ResetMode;
import com.revrobotics.spark.SparkLowLevel.MotorType;
import com.revrobotics.spark.SparkMax;
import com.revrobotics.spark.config.SparkBaseConfig.IdleMode;
import com.revrobotics.spark.config.SparkMaxConfig;

/** Real-hardware IO for the intake and shooter Spark MAX controllers. */
public class SuperstructureIOSparkMax implements SuperstructureIO {
  private static final int INTAKE_CAN_ID = 4;
  private static final int SHOOTER_CAN_ID = 5;
  private static final int INTAKE_CURRENT_LIMIT_AMPS = 30;
  private static final int SHOOTER_CURRENT_LIMIT_AMPS = 40;
  private static final double INTAKE_RAMP_SECONDS = 0.20;
  private static final double SHOOTER_RAMP_SECONDS = 0.35;

  private final SparkMax intakeMotor = new SparkMax(INTAKE_CAN_ID, MotorType.kBrushless);
  private final SparkMax shooterMotor = new SparkMax(SHOOTER_CAN_ID, MotorType.kBrushless);
  private final RelativeEncoder intakeEncoder = intakeMotor.getEncoder();
  private final RelativeEncoder shooterEncoder = shooterMotor.getEncoder();

  public SuperstructureIOSparkMax() {
    SparkMaxConfig intakeConfig = new SparkMaxConfig();
    intakeConfig
        .idleMode(IdleMode.kCoast)
        .smartCurrentLimit(INTAKE_CURRENT_LIMIT_AMPS)
        .openLoopRampRate(INTAKE_RAMP_SECONDS);
    intakeMotor.configure(
        intakeConfig, ResetMode.kNoResetSafeParameters, PersistMode.kPersistParameters);

    SparkMaxConfig shooterConfig = new SparkMaxConfig();
    shooterConfig
        .idleMode(IdleMode.kCoast)
        .smartCurrentLimit(SHOOTER_CURRENT_LIMIT_AMPS)
        .openLoopRampRate(SHOOTER_RAMP_SECONDS);
    shooterMotor.configure(
        shooterConfig, ResetMode.kNoResetSafeParameters, PersistMode.kPersistParameters);
  }

  @Override
  public void updateInputs(SuperstructureIOInputs inputs) {
    inputs.intakeConnected = true;
    inputs.intakeVelocityRpm = intakeEncoder.getVelocity();
    inputs.intakeAppliedVolts = intakeMotor.getAppliedOutput() * intakeMotor.getBusVoltage();
    inputs.intakeCurrentAmps = intakeMotor.getOutputCurrent();

    inputs.shooterConnected = true;
    inputs.shooterVelocityRpm = shooterEncoder.getVelocity();
    inputs.shooterAppliedVolts = shooterMotor.getAppliedOutput() * shooterMotor.getBusVoltage();
    inputs.shooterCurrentAmps = shooterMotor.getOutputCurrent();
  }

  @Override
  public void setIntakeVoltage(double volts) {
    intakeMotor.setVoltage(volts);
  }

  @Override
  public void setShooterVoltage(double volts) {
    shooterMotor.setVoltage(volts);
  }
}
