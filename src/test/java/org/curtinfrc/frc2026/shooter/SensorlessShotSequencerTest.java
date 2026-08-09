package org.curtinfrc.frc2026.shooter;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SensorlessShotSequencerTest {
  @Test
  void waitsForStableSpeedBeforeFeeding() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(50, 0.10, 100, 0.10, 0.20, 0);

    assertFalse(sequencer.update(0.00, 1300, 1400));
    assertFalse(sequencer.update(0.02, 1400, 1400));
    assertFalse(sequencer.update(0.10, 1400, 1400));
    assertTrue(sequencer.update(0.13, 1400, 1400));
  }

  @Test
  void finishesPulseAfterMinimumTimeWhenFlywheelDrops() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(50, 0.10, 100, 0.10, 0.20, 1);
    sequencer.update(0.00, 1400, 1400);
    assertTrue(sequencer.update(0.10, 1400, 1400));
    assertTrue(sequencer.update(0.15, 1200, 1400));
    assertFalse(sequencer.update(0.20, 1200, 1400));
    assertTrue(sequencer.isFinished());
  }

  @Test
  void requiresRecoveryBeforeAnotherPulse() {
    SensorlessShotSequencer sequencer = new SensorlessShotSequencer(50, 0.10, 100, 0.10, 0.20, 2);
    sequencer.update(0.00, 1400, 1400);
    sequencer.update(0.10, 1400, 1400);
    sequencer.update(0.20, 1200, 1400);

    assertFalse(sequencer.update(0.22, 1300, 1400));
    assertFalse(sequencer.update(0.30, 1400, 1400));
    assertTrue(sequencer.update(0.40, 1400, 1400));
  }
}
