package org.curtinfrc.frc2026.drive.Superstructure;

import org.littletonrobotics.junction.AutoLog;

/** Hardware abstraction for the intake and shooter motors. */
public interface SuperstructureIO {
  @AutoLog
  class SuperstructureIOInputs {
    public boolean intakeConnected = false;
    public double intakeVelocityRpm = 0.0;
    public double intakeAppliedVolts = 0.0;
    public double intakeCurrentAmps = 0.0;

    public boolean shooterConnected = false;
    public double shooterVelocityRpm = 0.0;
    public double shooterAppliedVolts = 0.0;
    public double shooterCurrentAmps = 0.0;
  }

  /** Updates all loggable inputs. */
  default void updateInputs(SuperstructureIOInputs inputs) {}

  default void setIntakeVoltage(double volts) {}

  default void setShooterVoltage(double volts) {}
}
