package org.curtinfrc.frc2026.drive.Superstructure;

import com.revrobotics.RelativeEncoder;
import com.revrobotics.spark.SparkLowLevel.MotorType;
import com.revrobotics.spark.SparkMax;
import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.controller.SimpleMotorFeedforward;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import org.littletonrobotics.junction.Logger;

public class Superstructure extends SubsystemBase {
  private SparkMax Intake = new SparkMax(4, MotorType.kBrushless); // intake + shoot
  private SparkMax Shooter = new SparkMax(5, MotorType.kBrushless); // pass
  private PIDController intake1 = new PIDController(0.0003, 0, 0);
  private PIDController shooter1 = new PIDController(0.0009, 0, 0);
  private SimpleMotorFeedforward intakeFeedFeedforward = new SimpleMotorFeedforward(0.15, 0.0019);
  private SimpleMotorFeedforward shooterFeedForward = new SimpleMotorFeedforward(0.45, 0.004);
  private RelativeEncoder intake2 = Intake.getEncoder();
  private RelativeEncoder shooter2 = Shooter.getEncoder();

  public Command intake() {
    return run(
        () -> {
          Intake.setVoltage(
              intake1.calculate(intake2.getVelocity(), 1000)
                  + intakeFeedFeedforward.calculate(1000));
          Shooter.setVoltage(
              shooter1.calculate(shooter2.getVelocity(), 1400)
                  + shooterFeedForward.calculate(1400));
          Logger.recordOutput("intake error", intake1.getError());
          Logger.recordOutput("shooter error", shooter1.getError());
        });
  }

  public Command shooter(double speed) {
    return run(() -> {
          Shooter.setVoltage(
              shooter1.calculate(shooter2.getVelocity(), speed)
                  + shooterFeedForward.calculate(speed));
          Intake.setVoltage(
              intake1.calculate(intake2.getVelocity(), 1000)
                  + intakeFeedFeedforward.calculate(1000));
          Logger.recordOutput("intake error", intake1.getError());
          Logger.recordOutput("shooter error", shooter1.getError());
        })
        .withTimeout(0.67)
        .andThen(
            run(
                () -> {
                  Intake.setVoltage(
                      intake1.calculate(intake2.getVelocity(), -3000)
                          + intakeFeedFeedforward.calculate(-3000));
                  Shooter.setVoltage(
                      shooter1.calculate(shooter2.getVelocity(), speed)
                          + shooterFeedForward.calculate(speed));
                  Logger.recordOutput("intake error", intake1.getError());
                  Logger.recordOutput("shooter error", shooter1.getError());
                }));
  }

  public Command stop() {
    return run(
        () -> {
          Intake.setVoltage(0);
          Shooter.setVoltage(0);
        });
  }
}
