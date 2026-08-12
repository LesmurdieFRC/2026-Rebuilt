# Shooter velocity model

This model estimates starting flywheel setpoints from 1-6 m. It uses official
2026 FUEL mass/diameter and HUB height, quadratic aerodynamic drag, and the
known 2 m / 1400 RPM shot to calibrate the otherwise unknown conversion from
motor RPM to ball launch speed.

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts\requirements.txt
.\.venv\Scripts\python.exe scripts\model_shooter.py
```

Generated CSV data and the sensitivity plot are written to `scripts/output/`.
The model assumptions are constants near the top of `model_shooter.py` so the
launch angle and exit height can be replaced with measured robot values.

The current assumed shooter exit is 0.60 m above the carpet at a fixed 70
degree elevation. That angle is the shallowest round-number assumption that
still has the nominal 1 m shot descending as it reaches the center of the HUB
opening. Measure both values before treating the projection as anything more
than a starting point. Runtime distance is robot-center to HUB-center, so the
model deliberately targets the center of the opening rather than its near lip.

The model is deliberately anchored to the measured 2 m point. It cannot model
wheel compression, ball-to-ball variability, spin/Magnus lift, battery sag, or
slip without robot data. Treat every generated value other than 2 m as an
initial tuning setpoint and record the final on-field setpoints back into the
Java interpolation map.

Official dimensions come from the [2026 FIRST Robotics Competition game
manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm):
the FUEL is a 5.91 in diameter, 0.448-0.500 lb foam ball, and the HUB top opening
starts 72 in above the carpet.
