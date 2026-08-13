package org.curtinfrc.frc2026.shooter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SensorlessShotSequencerTest {
  private static final SensorlessShotSequencer.Config CONFIG =
      new SensorlessShotSequencer.Config(50, 0.10, 600, 100, 0.04, 0.20, 0.12, -2500, -800, 8, 6);

  @Test
  void waitsForStableSpeedBeforeFeeding() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 0);

    assertFalse(sequencer.update(0.00, sample(1300, 1400, 0, 5, 2)));
    assertFalse(sequencer.update(0.02, sample(1400, 1400, 0, 5, 2)));
    assertFalse(sequencer.update(0.10, sample(1400, 1400, 0, 5, 2)));
    assertTrue(sequencer.update(0.13, sample(1400, 1400, 0, 5, 2)));
  }

  @Test
  void waitsForAccelerationToSettleBeforeFeeding() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 0);

    assertFalse(sequencer.update(0.00, sample(1400, 1400, 900, 5, 2)));
    assertFalse(sequencer.update(0.10, sample(1400, 1400, 700, 5, 2)));
    assertFalse(sequencer.update(0.12, sample(1400, 1400, 100, 5, 2)));
    assertFalse(sequencer.update(0.20, sample(1400, 1400, 100, 5, 2)));
    assertTrue(sequencer.update(0.23, sample(1400, 1400, 100, 5, 2)));
  }

  @Test
  void stopsFeederEarlyWhenImpactDecelerationIsDetected() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 1);
    sequencer.update(0.00, sample(1400, 1400, 0, 5, 2));
    assertTrue(sequencer.update(0.10, sample(1400, 1400, 0, 5, 2)));

    assertTrue(sequencer.update(0.12, sample(1370, 1400, -3000, 12, 5)));
    assertFalse(sequencer.update(0.15, sample(1300, 1400, -3000, 15, 8)));

    assertTrue(sequencer.isFinished());
    assertEquals(1, sequencer.getEstimatedShotEvents());
    assertEquals("FlywheelDroop", sequencer.getLastEventReason());
    assertEquals(1300, sequencer.getLastPulseMinimumRpm());
  }

  @Test
  void combinesCurrentRiseWithSupportingDeceleration() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 1);
    sequencer.update(0.00, sample(1400, 1400, 0, 5, 2));
    sequencer.update(0.10, sample(1400, 1400, 0, 5, 2));

    assertFalse(sequencer.update(0.15, sample(1360, 1400, -1000, 14, 3)));
    assertEquals(1, sequencer.getEstimatedShotEvents());
    assertEquals("FlywheelCurrent", sequencer.getLastEventReason());
  }

  @Test
  void ignoresCurrentSpikeWithoutFlywheelImpactEvidence() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 1);
    sequencer.update(0.00, sample(1400, 1400, 0, 5, 2));
    sequencer.update(0.10, sample(1400, 1400, 0, 5, 2));

    assertTrue(sequencer.update(0.15, sample(1400, 1400, 0, 20, 12)));
    assertFalse(sequencer.update(0.31, sample(1400, 1400, 0, 20, 12)));
    assertEquals(0, sequencer.getEstimatedShotEvents());
    assertEquals("Timeout", sequencer.getLastEventReason());
  }

  @Test
  void requiresRecoveryBeforeAnotherPulse() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(CONFIG, 2);
    sequencer.update(0.00, sample(1400, 1400, 0, 5, 2));
    sequencer.update(0.10, sample(1400, 1400, 0, 5, 2));
    sequencer.update(0.15, sample(1250, 1400, -3000, 15, 8));

    assertFalse(sequencer.update(0.17, sample(1300, 1400, 1000, 5, 2)));
    assertFalse(sequencer.update(0.25, sample(1400, 1400, 100, 5, 2)));
    assertTrue(sequencer.update(0.36, sample(1400, 1400, 0, 5, 2)));
  }

  private static SensorlessShotSequencer.Sample sample(
      double measuredRpm,
      double targetRpm,
      double accelerationRpmPerSecond,
      double flywheelCurrentAmps,
      double feederCurrentAmps) {
    return new SensorlessShotSequencer.Sample(
        measuredRpm, targetRpm, accelerationRpmPerSecond, flywheelCurrentAmps, feederCurrentAmps);
  }
}
