package org.curtinfrc.frc2026.drive.Superstructure;

import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.controller.SimpleMotorFeedforward;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import org.littletonrobotics.junction.Logger;

public class Superstructure extends SubsystemBase {
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

  public Command stop() {
    return run(this::stopMotors);
  }

  public Command stopOnce() {
    return runOnce(this::stopMotors);
  }

  private void setIntakeVelocity(double targetRpm) {
    io.setIntakeVoltage(
        intakeController.calculate(inputs.intakeVelocityRpm, targetRpm)
            + intakeFeedforward.calculate(targetRpm));
    Logger.recordOutput("Superstructure/Intake/TargetRPM", targetRpm);
    Logger.recordOutput("Superstructure/Intake/ErrorRPM", intakeController.getError());
  }

  private void setShooterVelocity(double targetRpm) {
    io.setShooterVoltage(
        shooterController.calculate(inputs.shooterVelocityRpm, targetRpm)
            + shooterFeedforward.calculate(targetRpm));
    Logger.recordOutput("Superstructure/Shooter/TargetRPM", targetRpm);
    Logger.recordOutput("Superstructure/Shooter/ErrorRPM", shooterController.getError());
  }

  private void stopMotors() {
    io.setIntakeVoltage(0);
    io.setShooterVoltage(0);
    Logger.recordOutput("Superstructure/Intake/TargetRPM", 0.0);
    Logger.recordOutput("Superstructure/Shooter/TargetRPM", 0.0);
  }
}
