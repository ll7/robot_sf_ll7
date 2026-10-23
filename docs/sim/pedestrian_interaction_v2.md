# Versioned pedestrian interaction laws

`simulation_config.pedestrian_force_profile: pedestrian_interaction_v2` selects
three opt-in laws for successor experiments. Omit the selector to retain the
legacy force kernels, force configuration hashes and serialized simulation settings.
No released scenario or frozen artifact selects the new profile.

The group gaze law implements Moussaïd et al. (2010), equation 2:
`F_gaze = -beta_1 * alpha * velocity`. The minimum head turn is
`alpha = max(0, bearing_to_other_members_centroid - phi)`, in radians.
`fov_phi` is the half-angle to either side of the velocity heading, in degrees.
Visible companions produce no gaze force; companions behind produce braking.
A stationary member or a coincident centroid produces zero. Waypoint distances do
not enter the law. The group repulsion uses unit vectors away from neighbours
within the existing threshold, as in equation 4. Undefined coincident pair
vectors contribute zero. Factors, cohesion and the repulsion threshold remain
unchanged; this is not a recalibration of the full group model.

Source: [Moussaïd et al., PLOS ONE 5(4), e10047](https://doi.org/10.1371/journal.pone.0010047).

The robot law retains inverse-cubic radial repulsion and adds a lateral term for
approaching pedestrians whose current velocity ray intersects the robot's
contact disc. It chooses the nearer passing side; exact head-on ties choose the
pedestrian's left. Steering ramps from zero at activation to full strength at
contact and vanishes on receding or non-intersecting trajectories. Repulsion is
bounded at contact inside the disc. It predicts against the robot's current
position, not its future trajectory; it does not guarantee non-contact.

Activation is `robot_radius + pedestrian_radius + edge_onset`. Inspection of
both main and the 0.0.8 freeze shows the legacy implementation already adds both
radii to its 2.0 m setting: it activates at **3.35 m**, not 2.0 m, for radii
1.0/0.35 m. The new profile explicitly names the surface gap and retains a
2.0 m gap by default. No cutoff increase is justified by the erroneous 2.0 m
centre-distance premise. Steering strength is relative to radial strength;
response-law multipliers scale both terms together.

Example sensitivity override (the three proposed centre cutoffs are 2.65, 3.0
and 3.5 m at those radii, using edge gaps 1.3, 1.65 and 2.15 m):

```yaml
simulation_config:
  pedestrian_force_profile: pedestrian_interaction_v2
  prf_config:
    edge_onset: 1.65
    steering_factor: 1.0
```

In Python, `GroupGazeForceV2Config`, `GroupRepulsiveForceV2Config` and
`PedRobotForceV2Config` allow selecting laws independently. Their legacy base
classes keep their original fields. `activation_threshold` remains a legacy knob;
`edge_onset` controls the new robot law.

Diagnostic checks on development seeds 1001–1030, a single pedestrian approaching
a stationary 1.0 m robot at 1.3 m/s from x=-5 m with lateral offsets drawn uniformly
in [-0.3, 0.3] m, gave the following surface clearances. This two-force fixture
uses the production desired and robot forces and integrator, not a full campaign.
All settings passed the robot in 30/30 cases with zero overlaps. These results
are implementation diagnostics, not a calibrated reaction-distance claim or the
required behaviour gate.

| Law | Centre cutoff (m) | Minimum clearance (m) | Median clearance (m) |
| --- | ---: | ---: | ---: |
| Legacy | 3.35 | 0.116 | 0.124 |
| New | 2.65 | 0.162 | 0.169 |
| New | 3.00 | 0.172 | 0.179 |
| New default | 3.35 | 0.176 | 0.184 |
| New | 3.50 | 0.177 | 0.185 |

The default preserves existing onset geometry; it is not chosen by maximizing
success. The empty-world gate and pedestrian-interaction validation remain
necessary before experiment adoption. Holds, roles, walls and physical contact
resolution are outside this profile.

Reproduce the diagnostic table with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=.:fast-pysf uv run python scripts/validation/probe_ped_robot_avoidance.py
```
