# Shooter SysId procedure

The robot includes a WPILib `SysIdRoutine` for the flywheel only. It bypasses
the normal PID/feedforward controller, holds the feeder stopped, applies the
requested characterization voltage directly, and records the required signals
through AdvantageKit.

## Safety and preparation

1. Put the robot on blocks and establish a clear emergency-stop path.
2. Remove every FUEL from the intake, feeder, and shooter.
3. Check that the flywheel can safely rotate in both directions. Do not run the
   reverse tests if the mechanism can be damaged by reversing.
4. Deploy the code, connect a controller on port 0, and enable **Test** mode.
5. Hold a binding only while the test should run. Releasing it immediately
   commands zero volts. Every command also has a six-second safety timeout.

## Test-mode bindings

| Controller input | Test |
| --- | --- |
| D-pad up | Quasistatic forward, 1 V/s ramp |
| D-pad down | Quasistatic reverse, 1 V/s ramp |
| D-pad right | Dynamic forward, 6 V step |
| D-pad left | Dynamic reverse, 6 V step |

Run all four tests in that order. Let the flywheel stop fully between tests.
The quasistatic tests may run to the six-second timeout. For dynamic tests,
roughly two seconds is normally enough; release the D-pad once the flywheel is
near steady speed.

## Getting the data into SysId

Real-robot WPILOG recording is enabled. Download the AdvantageKit log from the
robot, then:

1. Open the log in AdvantageScope.
2. Choose **File > Export Data**.
3. Select **WPILOG** and **AdvantageKit Cycles** timestamps.
4. Export the `/RealOutputs/Shooter` fields.
5. Open the exported log in the WPILib SysId tool and choose the angular/general
   mechanism analysis.
6. Assign these fields:
   - Test state: `Shooter/SysIdState`
   - Voltage: `Shooter/Flywheel/AppliedVolts`
   - Position: `Shooter/Flywheel/Position`
   - Velocity: `Shooter/Flywheel/Velocity`

Inspect the diagnostics and remove clearly invalid samples before accepting the
fit. Set the angular units to rotations and seconds. The Java flywheel
controller uses rotations per second internally, matching the resulting SysId
units, so `kS`, `kV`, and `kA` can be copied directly into the named constants
in `Shooter.java`. RPM remains available separately in the logs and distance
map because it is convenient for drivers and on-field tuning.
