package org.curtinfrc.frc2026.drive.Superstructure;

import com.revrobotics.RelativeEncoder;
import com.revrobotics.spark.SparkLowLevel.MotorType;
import com.revrobotics.spark.SparkMax;

/** Real-hardware IO for the intake and shooter Spark MAX controllers. */
public class SuperstructureIOSparkMax implements SuperstructureIO {
  private static final int INTAKE_CAN_ID = 4;
  private static final int SHOOTER_CAN_ID = 5;

  private final SparkMax intakeMotor = new SparkMax(INTAKE_CAN_ID, MotorType.kBrushless);
  private final SparkMax shooterMotor = new SparkMax(SHOOTER_CAN_ID, MotorType.kBrushless);
  private final RelativeEncoder intakeEncoder = intakeMotor.getEncoder();
  private final RelativeEncoder shooterEncoder = shooterMotor.getEncoder();

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
