Read agent_specs/python_implementer.md and run its role.

Read PROJECT_GUIDE.md and inspect the existing source, tests and
validation script before editing anything.

Modify the transient Couette-flow project to calculate and display the
channel-scale viscous Fourier number:

Fo_nu = nu*t/H**2

where:

- nu is the kinematic viscosity;
- t is the physical time;
- H is the channel height.

Fo_nu is the dimensionless channel-scale momentum-diffusion time. The
purpose of this change is to show how the transient velocity profile
relaxes towards the steady Couette profile as Fo_nu increases.

Implementation requirements
---------------------------

1. In src/transient_velocity_profile.py, add a reusable function such as:

   fourier_viscous_number(
       time,
       height,
       kinematic_viscosity,
   )

   which returns:

   Fo_nu = kinematic_viscosity*time/height**2

2. The function must:

   - accept either a scalar time or a NumPy array of times;
   - reject non-positive channel height;
   - reject non-positive kinematic viscosity;
   - reject negative time values;
   - return Fo_nu = 0 at t = 0.

3. Evaluate Fo_nu for every stored numerical profile.

   For the current reference case:

   H = 0.01 m
   nu = 1.0e-6 m**2/s
   t = 0, 5, 10, ..., 100 s

   the expected Fo_nu values are:

   0.00, 0.05, 0.10, ..., 1.00

4. Add the Fo_nu value to the generated CSV for every time-position
   pair.

   The CSV must contain a column named:

   fourier_viscous_number

5. Update the transient-profile graph so that every numerical profile
   is identified by its Fo_nu value.

   Place a text label near the upper portion of each numerical curve
   using the format:

   Fo_nu = 0.00
   Fo_nu = 0.05
   Fo_nu = 0.10

   and so forth.

   Use proper mathematical rendering in the final plot if practical,
   for example:

   Fo_ν = 0.05

   Position the labels so that they remain readable and do not overlap
   unnecessarily. Do not place all labels at the moving-wall endpoint,
   because all profiles meet at u = U there. Place each label beside a
   distinguishable upper section of its corresponding curve and use
   small offsets or collision avoidance where necessary.

6. Keep the existing time legend unless it makes the figure unreadable.
   If both the time legend and the direct Fo_nu labels cause excessive
   duplication, retain time in the legend and Fo_nu as the direct curve
   labels. Do not remove the Fo_nu labels.

7. Calculate the analytical transient Couette profile at:

   t = 5.0 s

   using the existing analytical_startup_couette function.

8. Plot this analytical profile on the same graph using:

   - a black dashed line;
   - a slightly greater linewidth than the numerical curves;
   - the legend label exactly:

     analyt sol

9. Do not duplicate or reimplement the analytical solution. Reuse the
   existing analytical_startup_couette function.

10. Preserve:

    - the Backward Euler numerical method;
    - 51 spatial points;
    - 20 time intervals;
    - the existing initial and boundary conditions;
    - all existing output filenames;
    - the existing analytical-reference implementation;
    - the existing validation thresholds unless a change is physically
      justified and clearly reported.

11. Update PROJECT_GUIDE.md and README.md briefly to define:

    Fo_nu = nu*t/H**2

    and explain that:

    - Fo_nu << 1 indicates an early transient;
    - Fo_nu of order 1 indicates that momentum has diffused across the
      channel scale;
    - closeness to the steady state must still be quantified from the
      velocity-profile error, rather than inferred from Fo_nu alone.

Deterministic tests
-------------------

Add or update tests to verify:

1. Fo_nu = 0 at t = 0.
2. Fo_nu = 0.05 at t = 5 s.
3. Fo_nu = 0.25 at t = 25 s.
4. Fo_nu = 0.50 at t = 50 s.
5. Fo_nu = 1.00 at t = 100 s.
6. Array input returns the correct array of Fo_nu values.
7. Invalid height, viscosity and negative time are rejected.
8. The CSV contains the fourier_viscous_number column.
9. The CSV contains 21 distinct Fo_nu values corresponding to the 21
   stored profiles.
10. The plotting implementation contains all numerical profiles.
11. The analytical t = 5 s curve is present once, is black, is dashed
    and has the legend label "analyt sol".
12. Every numerical profile has an associated Fo_nu label.
13. Existing physics and numerical tests continue to pass.

If necessary, refactor plotting into a small function that returns the
Matplotlib Figure and Axes objects so the plotted lines, labels and
legend can be tested directly without duplicating plotting logic.

Validation during implementation
--------------------------------

After making the changes:

1. Show the exact git diff.
2. Run:

   python src/transient_velocity_profile.py
   python -m pytest -q
   python scripts/validate_project.py

3. Inspect:

   results/transient_velocity_profiles.csv
   results/transient_velocity_profiles.png

4. Report:

   - every changed file;
   - the computed Fo_nu range and increment;
   - the complete pytest result;
   - the integrated validation result;
   - whether the analytical t = 5 s line is present;
   - whether all curve labels are readable;
   - any remaining limitation.

Do not commit or push the changes.

Independent validation stage
----------------------------

After implementation is complete, read
agent_specs/physics_validator.md and run its role.

This second stage is a read-only independent review. Do not modify any
file.

Run:

python src/transient_velocity_profile.py
python -m pytest -q
python scripts/validate_project.py

Independently verify:

1. Fo_nu is calculated as nu*t/H**2.
2. Fo_nu is dimensionless.
3. The reference case gives Fo_nu = t/100 s.
4. The 21 profiles therefore span Fo_nu = 0.00 to 1.00 in increments
   of 0.05.
5. Every numerical profile is associated with the correct Fo_nu label.
6. The CSV contains the correct Fo_nu value for every time-position
   pair.
7. The black dashed line labelled "analyt sol" is the existing
   analytical solution evaluated at t = 5 s.
8. The numerical t = 5 s profile is compared with the analytical
   t = 5 s profile at the same physical time.
9. The original initial conditions, boundary conditions, discrete
   momentum balance and steady-state limit remain correct.
10. All existing and new tests pass.
11. The generated plot is readable and the labels do not obscure the
    curves.
12. No test, tolerance or analytical reference was weakened merely to
    obtain a passing result.

Also quantify relaxation towards steady state by reporting, for each
stored Fo_nu value:

max_y |u_numerical(y,t) - U*y/H| / U

Do not declare the flow "almost relaxed" solely from Fo_nu. Identify
the first stored Fo_nu at which this normalised distance from the
steady profile falls below:

- 5%;
- 1%;
- 0.1%.

If a threshold is not reached, report that explicitly.

Return the report using the headings required by
agent_specs/physics_validator.md, with a clear PASS or FAIL for each
new requirement.