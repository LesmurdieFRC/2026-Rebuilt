package org.curtinfrc.frc2026.shooter;

/** Controls feeder pulses and estimates ball passage without a game-piece sensor. */
final class SensorlessShotSequencer {
  record Config(
      double readyToleranceRpm,
      double readyStableSeconds,
      double recoveryDropRpm,
      double eventArmingSeconds,
      double maximumFeedSeconds,
      double minimumEventSeparationSeconds,
      double impactDecelerationRpmPerSecond,
      double supportingDecelerationRpmPerSecond,
      double flywheelCurrentRiseAmps,
      double feederCurrentRiseAmps) {}

  record Sample(
      double measuredRpm,
      double targetRpm,
      double filteredAccelerationRpmPerSecond,
      double flywheelCurrentAmps,
      double feederCurrentAmps) {}

  private final Config config;
  private final int requestedPulses;

  private boolean feeding;
  private double readySinceSeconds = Double.NaN;
  private double feedStartedSeconds;
  private double lastDetectedEventSeconds = Double.NEGATIVE_INFINITY;
  private double flywheelCurrentBaselineAmps;
  private double feederCurrentBaselineAmps;
  private double pulseMinimumRpm;
  private double pulseMaximumDroopRpm;
  private double pulseMaximumFlywheelCurrentAmps;
  private double pulseMaximumFeederCurrentAmps;
  private double lastPulseMinimumRpm;
  private double lastPulseMaximumDroopRpm;
  private double lastPulseMaximumFlywheelCurrentAmps;
  private double lastPulseMaximumFeederCurrentAmps;
  private int completedPulses;
  private int estimatedShotEvents;
  private String lastEventReason = "None";

  SensorlessShotSequencer(Config config, int requestedPulses) {
    this.config = config;
    this.requestedPulses = requestedPulses;
  }

  void reset() {
    feeding = false;
    readySinceSeconds = Double.NaN;
    feedStartedSeconds = 0.0;
    lastDetectedEventSeconds = Double.NEGATIVE_INFINITY;
    completedPulses = 0;
    estimatedShotEvents = 0;
    lastEventReason = "None";
    lastPulseMinimumRpm = 0.0;
    lastPulseMaximumDroopRpm = 0.0;
    lastPulseMaximumFlywheelCurrentAmps = 0.0;
    lastPulseMaximumFeederCurrentAmps = 0.0;
  }

  boolean update(double nowSeconds, Sample sample) {
    if (feeding) {
      updatePulseExtrema(sample);
      double feedTime = nowSeconds - feedStartedSeconds;
      double droopRpm = sample.targetRpm() - sample.measuredRpm();
      double flywheelCurrentRise = sample.flywheelCurrentAmps() - flywheelCurrentBaselineAmps;
      double feederCurrentRise = sample.feederCurrentAmps() - feederCurrentBaselineAmps;

      boolean eventArmed =
          feedTime >= config.eventArmingSeconds()
              && nowSeconds - lastDetectedEventSeconds >= config.minimumEventSeparationSeconds();
      boolean excessiveDroop = droopRpm >= config.recoveryDropRpm();
      boolean strongDeceleration =
          sample.filteredAccelerationRpmPerSecond() <= config.impactDecelerationRpmPerSecond();
      boolean currentSupportedImpact =
          sample.filteredAccelerationRpmPerSecond() <= config.supportingDecelerationRpmPerSecond()
              && (flywheelCurrentRise >= config.flywheelCurrentRiseAmps()
                  || feederCurrentRise >= config.feederCurrentRiseAmps());

      if (eventArmed && (excessiveDroop || strongDeceleration || currentSupportedImpact)) {
        estimatedShotEvents++;
        lastDetectedEventSeconds = nowSeconds;
        lastEventReason =
            eventReason(
                excessiveDroop,
                strongDeceleration,
                flywheelCurrentRise >= config.flywheelCurrentRiseAmps());
        finishPulse();
      } else if (feedTime >= config.maximumFeedSeconds()) {
        lastEventReason = "Timeout";
        finishPulse();
      }
      return feeding;
    }

    if (isFinished()) {
      return false;
    }

    boolean atSpeed =
        Math.abs(sample.targetRpm() - sample.measuredRpm()) <= config.readyToleranceRpm();
    if (!atSpeed) {
      readySinceSeconds = Double.NaN;
      return false;
    }

    if (Double.isNaN(readySinceSeconds)) {
      readySinceSeconds = nowSeconds;
    } else if (nowSeconds - readySinceSeconds >= config.readyStableSeconds()) {
      startPulse(nowSeconds, sample);
    }
    return feeding;
  }

  private void startPulse(double nowSeconds, Sample sample) {
    feeding = true;
    feedStartedSeconds = nowSeconds;
    readySinceSeconds = Double.NaN;
    flywheelCurrentBaselineAmps = sample.flywheelCurrentAmps();
    feederCurrentBaselineAmps = sample.feederCurrentAmps();
    pulseMinimumRpm = sample.measuredRpm();
    pulseMaximumDroopRpm = Math.max(0.0, sample.targetRpm() - sample.measuredRpm());
    pulseMaximumFlywheelCurrentAmps = sample.flywheelCurrentAmps();
    pulseMaximumFeederCurrentAmps = sample.feederCurrentAmps();
  }

  private void updatePulseExtrema(Sample sample) {
    pulseMinimumRpm = Math.min(pulseMinimumRpm, sample.measuredRpm());
    pulseMaximumDroopRpm =
        Math.max(pulseMaximumDroopRpm, sample.targetRpm() - sample.measuredRpm());
    pulseMaximumFlywheelCurrentAmps =
        Math.max(pulseMaximumFlywheelCurrentAmps, sample.flywheelCurrentAmps());
    pulseMaximumFeederCurrentAmps =
        Math.max(pulseMaximumFeederCurrentAmps, sample.feederCurrentAmps());
  }

  private void finishPulse() {
    feeding = false;
    completedPulses++;
    readySinceSeconds = Double.NaN;
    lastPulseMinimumRpm = pulseMinimumRpm;
    lastPulseMaximumDroopRpm = pulseMaximumDroopRpm;
    lastPulseMaximumFlywheelCurrentAmps = pulseMaximumFlywheelCurrentAmps;
    lastPulseMaximumFeederCurrentAmps = pulseMaximumFeederCurrentAmps;
  }

  private static String eventReason(
      boolean excessiveDroop, boolean strongDeceleration, boolean flywheelCurrentSpike) {
    if (excessiveDroop) return "FlywheelDroop";
    if (strongDeceleration) return "FlywheelDeceleration";
    return flywheelCurrentSpike ? "FlywheelCurrent" : "FeederCurrent";
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

  int getEstimatedShotEvents() {
    return estimatedShotEvents;
  }

  double getLastDetectedEventSeconds() {
    return Double.isFinite(lastDetectedEventSeconds) ? lastDetectedEventSeconds : Double.NaN;
  }

  double getLastPulseMinimumRpm() {
    return lastPulseMinimumRpm;
  }

  double getLastPulseMaximumDroopRpm() {
    return lastPulseMaximumDroopRpm;
  }

  double getLastPulseMaximumFlywheelCurrentAmps() {
    return lastPulseMaximumFlywheelCurrentAmps;
  }

  double getLastPulseMaximumFeederCurrentAmps() {
    return lastPulseMaximumFeederCurrentAmps;
  }

  double getFlywheelCurrentBaselineAmps() {
    return flywheelCurrentBaselineAmps;
  }

  double getFeederCurrentBaselineAmps() {
    return feederCurrentBaselineAmps;
  }

  String getLastEventReason() {
    return lastEventReason;
  }

  String getState() {
    if (feeding) return "Feeding";
    if (isFinished()) return "Complete";
    return completedPulses == 0 ? "SpinningUp" : "Recovering";
  }
}
