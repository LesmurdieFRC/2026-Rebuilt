package org.curtinfrc.frc2026.drive.Superstructure;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.controller.SimpleMotorFeedforward;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import org.littletonrobotics.junction.Logger;

public class Superstructure extends SubsystemBase {
  private static final double MAX_CONTROL_VOLTS = 12.0;
  private final SuperstructureIO io;
  private final SuperstructureIOInputsAutoLogged inputs = new SuperstructureIOInputsAutoLogged();
  private final PIDController intakeController = new PIDController(0.0003, 0, 0);
  private final PIDController shooterController = new PIDController(0.0009, 0, 0);
  private final SimpleMotorFeedforward intakeFeedforward = new SimpleMotorFeedforward(0.15, 0.0019);
  private final SimpleMotorFeedforward shooterFeedforward = new SimpleMotorFeedforward(0.45, 0.004);

  public Superstructure(SuperstructureIO io) {
    this.io = io;
  }

  @Override
  public void periodic() {
    io.updateInputs(inputs);
    Logger.processInputs("Superstructure", inputs);
  }

  public Command intake() {
    return run(
        () -> {
          setIntakeVelocity(1000);
          setShooterVelocity(1400);
        });
  }

  public Command shooter(double speed) {
    return run(() -> {
          setShooterVelocity(speed);
          setIntakeVelocity(1000);
        })
        .withTimeout(0.67)
        .andThen(
            run(
                () -> {
                  setIntakeVelocity(-3000);
                  setShooterVelocity(speed);
                }))
        .finallyDo(interrupted -> stopMotors());
  }

  public Command jam(double speed) {
    return run(() -> {
          setShooterVelocity(speed);
          setIntakeVelocity(-2000);
        })
        .finallyDo(interrupted -> stopMotors());
  }

  public Command stop() {
    return run(this::stopMotors);
  }

  public Command stopOnce() {
    return runOnce(this::stopMotors);
  }

  private void setIntakeVelocity(double targetRpm) {
    double requestedVolts =
        intakeController.calculate(inputs.intakeVelocityRpm, targetRpm)
            + intakeFeedforward.calculate(targetRpm);
    double commandedVolts = MathUtil.clamp(requestedVolts, -MAX_CONTROL_VOLTS, MAX_CONTROL_VOLTS);
    io.setIntakeVoltage(commandedVolts);
    Logger.recordOutput("Superstructure/Intake/TargetRPM", targetRpm);
    Logger.recordOutput("Superstructure/Intake/ErrorRPM", intakeController.getError());
    Logger.recordOutput("Superstructure/Intake/RequestedVolts", requestedVolts);
    Logger.recordOutput("Superstructure/Intake/CommandedVolts", commandedVolts);
  }

  private void setShooterVelocity(double targetRpm) {
    double requestedVolts =
        shooterController.calculate(inputs.shooterVelocityRpm, targetRpm)
            + shooterFeedforward.calculate(targetRpm);
    double commandedVolts = MathUtil.clamp(requestedVolts, -MAX_CONTROL_VOLTS, MAX_CONTROL_VOLTS);
    io.setShooterVoltage(commandedVolts);
    Logger.recordOutput("Superstructure/Shooter/TargetRPM", targetRpm);
    Logger.recordOutput("Superstructure/Shooter/ErrorRPM", shooterController.getError());
    Logger.recordOutput("Superstructure/Shooter/RequestedVolts", requestedVolts);
    Logger.recordOutput("Superstructure/Shooter/CommandedVolts", commandedVolts);
  }

  private void stopMotors() {
    io.setIntakeVoltage(0);
    io.setShooterVoltage(0);
    Logger.recordOutput("Superstructure/Intake/TargetRPM", 0.0);
    Logger.recordOutput("Superstructure/Shooter/TargetRPM", 0.0);
    Logger.recordOutput("Superstructure/Intake/RequestedVolts", 0.0);
    Logger.recordOutput("Superstructure/Intake/CommandedVolts", 0.0);
    Logger.recordOutput("Superstructure/Shooter/RequestedVolts", 0.0);
    Logger.recordOutput("Superstructure/Shooter/CommandedVolts", 0.0);
  }
}
