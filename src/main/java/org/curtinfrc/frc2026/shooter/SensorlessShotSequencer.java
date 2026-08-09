package org.curtinfrc.frc2026.shooter;

/** Controls feeder pulses using flywheel speed when no game-piece sensor is available. */
final class SensorlessShotSequencer {
  private final double readyToleranceRpm;
  private final double readyStableSeconds;
  private final double recoveryDropRpm;
  private final double minimumFeedSeconds;
  private final double maximumFeedSeconds;
  private final int requestedPulses;

  private boolean feeding;
  private double readySinceSeconds = Double.NaN;
  private double feedStartedSeconds;
  private int completedPulses;

  SensorlessShotSequencer(
      double readyToleranceRpm,
      double readyStableSeconds,
      double recoveryDropRpm,
      double minimumFeedSeconds,
      double maximumFeedSeconds,
      int requestedPulses) {
    this.readyToleranceRpm = readyToleranceRpm;
    this.readyStableSeconds = readyStableSeconds;
    this.recoveryDropRpm = recoveryDropRpm;
    this.minimumFeedSeconds = minimumFeedSeconds;
    this.maximumFeedSeconds = maximumFeedSeconds;
    this.requestedPulses = requestedPulses;
  }

  void reset() {
    feeding = false;
    readySinceSeconds = Double.NaN;
    feedStartedSeconds = 0.0;
    completedPulses = 0;
  }

  boolean update(double nowSeconds, double measuredRpm, double targetRpm) {
    if (feeding) {
      double feedTime = nowSeconds - feedStartedSeconds;
      boolean speedDropped = measuredRpm < targetRpm - recoveryDropRpm;
      if (feedTime >= maximumFeedSeconds || (feedTime >= minimumFeedSeconds && speedDropped)) {
        feeding = false;
        completedPulses++;
        readySinceSeconds = Double.NaN;
      }
      return feeding;
    }

    if (isFinished()) {
      return false;
    }

    boolean atSpeed = Math.abs(targetRpm - measuredRpm) <= readyToleranceRpm;
    if (!atSpeed) {
      readySinceSeconds = Double.NaN;
      return false;
    }

    if (Double.isNaN(readySinceSeconds)) {
      readySinceSeconds = nowSeconds;
    } else if (nowSeconds - readySinceSeconds >= readyStableSeconds) {
      feeding = true;
      feedStartedSeconds = nowSeconds;
      readySinceSeconds = Double.NaN;
    }
    return feeding;
  }

  boolean isFeeding() {
    return feeding;
  }

  boolean isFinished() {
    return requestedPulses > 0 && completedPulses >= requestedPulses;
  }

  int getCompletedPulses() {
    return completedPulses;
  }

  String getState() {
    if (feeding) return "Feeding";
    if (isFinished()) return "Complete";
    return completedPulses == 0 ? "SpinningUp" : "Recovering";
  }
}
