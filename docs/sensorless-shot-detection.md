# Sensorless shot detection tuning

The final intake roller is the feeder and there is no beam break, so the robot
estimates ball passage using several synchronized signals. A shot event requires
one of the following after the feeder has run for 40 ms:

- Flywheel speed has dropped at least 100 RPM.
- Filtered flywheel deceleration is at least 2500 RPM/s downward.
- Filtered deceleration is at least 800 RPM/s downward and either flywheel
  current rose by 8 A or feeder current rose by 6 A relative to the start of
  that feeder pulse.

A current spike by itself is not counted because it could be a feeder jam. On a
detected event, the feeder immediately returns to its staging speed. The next
pulse cannot start until the flywheel has recovered and remained within 50 RPM
of target for 100 ms. Empty pulses time out after 200 ms.

## AdvantageScope tuning

Plot these signals together over several single-, double-, and triple-ball
tests:

- `Shooter/Feeder/FeedingShot`
- `Shooter/Flywheel/MeasuredRPM`
- `Shooter/Flywheel/TargetRPM`
- `Shooter/Flywheel/FilteredAccelerationRPMPerSecond`
- `Shooter/Flywheel/OutputCurrentAmps`
- `Shooter/Feeder/OutputCurrentAmps`
- `Shooter/ShotDetection/FlywheelCurrentRiseAmps`
- `Shooter/ShotDetection/FeederCurrentRiseAmps`
- `Shooter/EstimatedShotEvents`
- `Shooter/LastShotEventReason`

Also inspect the last-pulse minimum RPM, maximum droop, peak currents, and
recovery duration. Adjust the named detection constants near the top of
`Shooter.java` only after comparing real shot events with empty feeds and jams.

This estimator cannot guarantee an exact ball count when multiple balls are
already touching the flywheel. Its purpose is to stop adding energy demand as
soon as an impact is visible, then force recovery before another feeder pulse.
