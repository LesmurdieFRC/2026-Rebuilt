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
    assertEquals(1450.0, Shooter.flywheelSpeedForDistance(2.25));
  }

  @Test
  void clampsOutsideCalibratedRange() {
    assertEquals(1225.0, Shooter.flywheelSpeedForDistance(0.5));
    assertEquals(2375.0, Shooter.flywheelSpeedForDistance(7.0));
  }

  @Test
  void fallsBackToTwoMeterCalibrationForInvalidDistance() {
    assertEquals(1400.0, Shooter.flywheelSpeedForDistance(Double.NaN));
    assertEquals(1400.0, Shooter.flywheelSpeedForDistance(Double.POSITIVE_INFINITY));
  }

  @Test
  void interpolatesFloorLandingModel() {
    assertEquals(1257.0, Shooter.floorLandingSpeedForDistance(2.25));
    assertEquals(510.0, Shooter.floorLandingSpeedForDistance(0.0));
    assertEquals(2991.0, Shooter.floorLandingSpeedForDistance(10.0));
  }
}
