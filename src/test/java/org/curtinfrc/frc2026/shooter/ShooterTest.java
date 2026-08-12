package org.curtinfrc.frc2026.shooter;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ShooterTest {
  @Test
  void preservesExistingSpeedAtTwoMeters() {
    assertEquals(1400.0, Shooter.flywheelSpeedForDistance(2.0));
  }

  @Test
  void interpolatesBetweenCalibrationPoints() {
    assertEquals(1462.5, Shooter.flywheelSpeedForDistance(2.25));
  }

  @Test
  void clampsOutsideCalibratedRange() {
    assertEquals(1175.0, Shooter.flywheelSpeedForDistance(0.5));
    assertEquals(2350.0, Shooter.flywheelSpeedForDistance(7.0));
  }

  @Test
  void fallsBackToTwoMeterCalibrationForInvalidDistance() {
    assertEquals(1400.0, Shooter.flywheelSpeedForDistance(Double.NaN));
    assertEquals(1400.0, Shooter.flywheelSpeedForDistance(Double.POSITIVE_INFINITY));
  }
}
