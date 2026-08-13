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
5. Press **A** once to start the complete sequence. Press **A** again at any
   time to cancel it and command zero volts. Every individual test also has a
   six-second safety timeout.

## Test-mode bindings

| Controller input | Action |
| --- | --- |
| A | Start the complete four-test sequence; press again to cancel |

The command automatically runs quasistatic forward, quasistatic reverse,
dynamic forward, and dynamic reverse. It commands zero volts after every test,
waits until the flywheel is below 50 RPM, and allows another 250 ms to settle
before continuing. The whole sequence takes roughly 30 seconds.

## Getting the data into SysId

The robot records a WPILOG to internal storage only while Test mode is enabled.
Disable the robot after the sequence finishes so the file is closed, then set
AdvantageScope's robot log folder to `/home/lvuser/logs` and use **File >
Download Logs...**. After downloading the newest AdvantageKit log:

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
