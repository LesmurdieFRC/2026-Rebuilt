package org.curtinfrc.frc2026.drive.Superstructure;

import com.revrobotics.spark.SparkLowLevel.MotorType;

import org.littletonrobotics.junction.Logger;

import com.revrobotics.RelativeEncoder;
import com.revrobotics.spark.SparkMax;

import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

public class Superstructure extends SubsystemBase {
  private SparkMax Intake = new SparkMax(4, MotorType.kBrushless); // intake + shoot
  private SparkMax Shooter = new SparkMax(5, MotorType.kBrushless); // pass
private PIDController intakE=new PIDController(1, 0, 0);
private PIDController shooteR=new PIDController(1, 0, 0);
private RelativeEncoder intaKe=Intake.getEncoder();
private RelativeEncoder shooTer=Shooter.getEncoder();
  public Command intake() {
    return run(
        () -> {
          Intake.setVoltage(intakE.calculate(intaKe.getVelocity(),9));
          Shooter.setVoltage(shooteR.calculate(shooTer.getVelocity(),9));
          Logger.recordOutput("intake error",intakE.getError());
          Logger.recordOutput("shooter error",shooteR.getError());
        });
  }

  public Command shooter() {
    return run(() -> {
          Shooter.setVoltage(shooteR.calculate(shooTer.getVelocity(),12));
          Logger.recordOutput("intake error",intakE.getError());
          Logger.recordOutput("shooter error",shooteR.getError());
        })
        .withTimeout(2)
        .andThen(
            run(
                () -> {
                  Intake.setVoltage(intakE.calculate(intaKe.getVelocity(),12));
                  Shooter.setVoltage(shooteR.calculate(shooTer.getVelocity(),12));
                  Logger.recordOutput("intake error",intakE.getError());
          Logger.recordOutput("shooter error",shooteR.getError());
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
