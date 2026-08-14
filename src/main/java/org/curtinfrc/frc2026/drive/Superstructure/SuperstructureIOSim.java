package org.curtinfrc.frc2026.drive.Superstructure;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.math.system.plant.LinearSystemId;
import edu.wpi.first.math.util.Units;
import edu.wpi.first.wpilibj.simulation.DCMotorSim;

/** Physics simulation for the intake and shooter rollers. */
public class SuperstructureIOSim implements SuperstructureIO {
  private static final double LOOP_PERIOD_SECONDS = 0.02;
  private static final double GEARING = 1.0;
  private static final double MOMENT_OF_INERTIA_KG_METERS_SQUARED = 0.001;
  private static final DCMotor MOTOR = DCMotor.getNEO(1);

  private final DCMotorSim intakeSim = createMotorSim();
  private final DCMotorSim shooterSim = createMotorSim();
  private double intakeAppliedVolts = 0.0;
  private double shooterAppliedVolts = 0.0;

  private static DCMotorSim createMotorSim() {
    return new DCMotorSim(
        LinearSystemId.createDCMotorSystem(MOTOR, MOMENT_OF_INERTIA_KG_METERS_SQUARED, GEARING),
        MOTOR);
  }

  @Override
  public void updateInputs(SuperstructureIOInputs inputs) {
    intakeSim.setInputVoltage(intakeAppliedVolts);
    shooterSim.setInputVoltage(shooterAppliedVolts);
    intakeSim.update(LOOP_PERIOD_SECONDS);
    shooterSim.update(LOOP_PERIOD_SECONDS);

    inputs.intakeConnected = true;
    inputs.intakeVelocityRpm =
        Units.radiansPerSecondToRotationsPerMinute(intakeSim.getAngularVelocityRadPerSec());
    inputs.intakeAppliedVolts = intakeAppliedVolts;
    inputs.intakeCurrentAmps = Math.abs(intakeSim.getCurrentDrawAmps());

    inputs.shooterConnected = true;
    inputs.shooterVelocityRpm =
        Units.radiansPerSecondToRotationsPerMinute(shooterSim.getAngularVelocityRadPerSec());
    inputs.shooterAppliedVolts = shooterAppliedVolts;
    inputs.shooterCurrentAmps = Math.abs(shooterSim.getCurrentDrawAmps());
  }

  @Override
  public void setIntakeVoltage(double volts) {
    intakeAppliedVolts = MathUtil.clamp(volts, -12.0, 12.0);
  }

  @Override
  public void setShooterVoltage(double volts) {
    shooterAppliedVolts = MathUtil.clamp(volts, -12.0, 12.0);
  }
}
