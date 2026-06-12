"""Base configuration for the in-hand cube reorientation task."""

from mjlab.asset_zoo.props.qwerty_cube import CUBE_HALF_EXTENT
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.reorient import mdp as reorient_mdp
from mjlab.tasks.reorient.mdp import ReorientationCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import OutlierNoiseCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_reorient_cube_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base in-hand cube reorientation task configuration."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      # ~ joint_pos noise (0.01 rad) differenced over the 0.02 s control step: real
      # hand velocity is encoder-differenced, so its noise scales as pos_noise / dt
      # (~0.5 rad/s). The old +-1.5 was ~half the typical joint speed -- it drowned
      # the signal rather than modeling the sensor.
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # Cube-pose terms come from vision on the real robot, so they get the perception
    # gap: position/orientation noise plus an occasional per-env "blip" on the
    # orientation (a gross pose-estimate glitch, ~2% of steps). (Observation delay is
    # deliberately not modeled: a realistic constant lag is a later, hardware-measured
    # add, and the per-step-random delay we tried was both unrealistic and unhelpful.)
    "cube_pos": ObservationTermCfg(
      func=reorient_mdp.ee_to_object_distance,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "cube_ori": ObservationTermCfg(
      func=reorient_mdp.object_orientation_6d,
      params={"object_name": "cube"},
      # ~3 deg baseline orientation noise on the 6D rep, plus a 2% blip of +-0.5.
      noise=OutlierNoiseCfg(
        n_min=-0.05, n_max=0.05, outlier_prob=0.02, outlier_min=-0.5, outlier_max=0.5
      ),
    ),
    "cube_to_goal_ori": ObservationTermCfg(
      func=reorient_mdp.object_to_goal_orientation_6d,
      params={"object_name": "cube", "command_name": "goal"},
      noise=OutlierNoiseCfg(
        n_min=-0.05, n_max=0.05, outlier_prob=0.02, outlier_min=-0.5, outlier_max=0.5
      ),
    ),
    "cube_lin_vel": ObservationTermCfg(
      func=reorient_mdp.object_lin_vel_b,
      params={"object_name": "cube"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "cube_ang_vel": ObservationTermCfg(
      func=reorient_mdp.object_ang_vel_b,
      params={"object_name": "cube"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "fingertip_to_cube": ObservationTermCfg(
      func=reorient_mdp.fingertip_to_object,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "fingertip_to_palm": ObservationTermCfg(
      func=reorient_mdp.fingertip_to_palm,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "prev_actions": ObservationTermCfg(func=mdp.second_last_action),
  }

  # Privileged critic-only terms: the success state machine (hold + window) is
  # internal command state the policy can't see, so feeding it to the value
  # function makes the sparse success bonus Markovian for the critic.
  critic_terms = {
    **actor_terms,
    # The critic is privileged: enable_corruption=False (below) strips the noise/blip
    # so it sees the clean, current cube pose.
    "goal_hold_progress": ObservationTermCfg(
      func=reorient_mdp.goal_hold_progress,
      params={"command_name": "goal"},
    ),
    "goal_window_progress": ObservationTermCfg(
      func=reorient_mdp.goal_window_progress,
      params={"command_name": "goal"},
    ),
  }

  observations = {
    # Actor sees the noisy + blipped vision obs (the sim-to-real perception gap); the
    # critic is privileged and clean (enable_corruption=False strips the noise/blip).
    # No observation history: with no modeled delay, history's main job (de-lagging) is
    # gone, and a head-to-head showed it did not help on this task at equal compute.
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,
      # Hold the home grasp for the first 0.4 s so the cube drops and settles
      # into the cage before the policy starts acting.
      warmup_time_s=0.4,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "goal": ReorientationCommandCfg(
      entity_name="cube",
      robot_name="robot",
      success_threshold=0.2,
      # 13 control steps @ 50 Hz = 0.26 s, matching wuji's 5-step hold @ 20 Hz.
      success_hold_steps=13,
      # 50 control steps @ 50 Hz = 1.0 s, matching wuji's 20-step window @ 20 Hz.
      goal_switch_delay=50,
      # Bounded (<=45 deg) perturbation of the held goal each switch: a built-in
      # curriculum that lets the policy chain reorientations. Flip to True for
      # full-SO(3) goals once the policy can chain.
      success_resample_full_so3=True,
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      viz=ReorientationCommandCfg.VizCfg(cube_half_extent=CUBE_HALF_EXTENT),
    )
  }

  events = {
    # Tilt the hand and nestle the cube in the (tilted) cradle with a random
    # SO(3) orientation. cradle_offset_b is filled in per-robot.
    "reset_hand_and_cube": EventTermCfg(
      func=reorient_mdp.reset_hand_and_object,
      mode="reset",
      params={
        "hand_pitch_range": (-0.4, 0.1),
        "position_noise": 0.005,
        "cradle_offset_b": (0.0, 0.0, 0.0),  # Set per-robot.
        "object_cfg": SceneEntityCfg("cube"),
      },
    ),
    # Per-robot finger-joint resets are added in the robot config (the joint
    # groups are robot-specific).
    #
    # Grip-friction domain randomization. The pads and cube are condim 4, so each
    # contact has two live friction axes: sliding (axis 0) and torsional (axis 1).
    # Friction combines by max at equal priority, so for each axis the cube (the
    # object) carries the full target range and usually wins, while the pads sit on
    # a lower/overlapping range that sets the floor and adds variation when the cube
    # draws low. One term per (geom, axis) because each axis needs its own
    # distribution: sliding is uniform; torsional is log_uniform (a small length
    # spanning a multiplicative range, interpreted as the contact-patch diameter in
    # metres). shared_random => all pads share one draw per env. startup mode =
    # sampled once per env at build (friction is a material property). Realized grip
    # values are max(pad, cube); verify the distributions with
    # scripts/audit_reorient_friction.py. Pad geom pattern is filled in per-robot.
    "cube_friction_slide": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.5, 1.2),
        "shared_random": True,
      },
    ),
    "cube_friction_spin": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (0.002, 0.006),
        "shared_random": True,
      },
    ),
    "pad_friction_slide": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.4, 1.0),
        "shared_random": True,
      },
    ),
    "pad_friction_spin": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (0.002, 0.006),
        "shared_random": True,
      },
    ),
  }

  # Task rewards are bounded to [0, 1] and gated by "cube held" (zeroed when the
  # cube has fallen), so a drop costs *lost reward* rather than a magic penalty.
  # Costs are reserved for the genuinely unbounded regularizers (jerk, speed) plus
  # a small cage-escape gradient.
  _HELD_GATE = {"gate_object_name": "cube", "gate_min_height": 0.05}
  rewards = {
    # Coarse kernel: flat inside the bound, gentle gradient over the whole range --
    # pulls the cube in from a cold goal.
    "orientation_tolerance": RewardTermCfg(
      func=reorient_mdp.cube_orientation_tolerance,
      weight=1.0,
      params={"command_name": "goal", **_HELD_GATE},
    ),
    # Precise kernel: a sharp ramp responsive only within ~0.4 rad of the goal (zero
    # beyond, so it never touches the cold approach). Gives the steep last-mile
    # gradient the flat coarse kernel lacks, so the policy is pulled to actually nail
    # the pose instead of hovering just outside the success threshold. Rewards only
    # *being close*, never the act of rotating, so it can't form a spinning optimum.
    "orientation_precise": RewardTermCfg(
      func=reorient_mdp.cube_orientation_tolerance,
      weight=1.0,
      params={
        "command_name": "goal",
        "bound": 0.0,
        "margin": 0.4,
        "value_at_margin": 0.0,
        **_HELD_GATE,
      },
    ),
    # Dense, monotonic "reach and keep the pose" reward: grows with cumulative
    # in-threshold time (never dips at a goal switch), saturating at 1. Replaces a
    # sparse success bonus.
    "sustained_hold": RewardTermCfg(
      func=reorient_mdp.sustained_hold,
      weight=2.0,
      # ~one episode of in-threshold time to reach full reward, so a good policy
      # keeps earning marginal reward for holding longer instead of saturating early.
      params={"command_name": "goal", "saturation_steps": 500.0, **_HELD_GATE},
    ),
    # Grasp-quality task rewards (bounded [0, 1], self-gating via contact): grip
    # with the fingertips and keep the cube off the palm.
    "fingertip_contact": RewardTermCfg(
      func=reorient_mdp.fingertip_object_contact,
      weight=0.5,
      params={"sensor_name": "tip_object_contact"},
    ),
    "cube_off_palm": RewardTermCfg(
      func=reorient_mdp.cube_off_palm,
      weight=0.5,
      params={
        "tip_sensor_name": "tip_object_contact",
        "palm_sensor_name": "palm_object_contact",
      },
    ),
    # Posture anchor: positive, bounded-[0,1] reward (1.0 at the home pose) that keeps
    # the hand in an open, engaged posture instead of collapsing into a minimal
    # fingertip pinch. The action is relative-to-current with no other pose cost, so
    # without this the home pose is forgotten after the 0.4 s warmup. Scalar std=0.5
    # rad; the mean over joints stays forgiving of the finger motion gaiting needs.
    "posture": RewardTermCfg(
      func=reorient_mdp.posture,
      weight=0.5,
      params={
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Costs: unbounded regularizers + a small anti-drop gradient (escape distance).
    "cage_escape": RewardTermCfg(
      func=reorient_mdp.CageEscapePenalty,
      weight=-1.0,
      params={
        "object_name": "cube",
        "margin": 0.02,
        "asset_cfg": SceneEntityCfg("robot", body_names=(), site_names=()),
      },
    ),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
    "joint_vel_hinge": RewardTermCfg(
      func=reorient_mdp.joint_velocity_hinge_penalty,
      weight=-0.005,
      params={
        "max_vel": 1.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Joint-limit cost: a soft wall that penalizes only the amount a joint crosses its
    # soft limit (zero inside the range), so it costs nothing in normal operation and
    # only bites when the policy slams a joint into its stop -- which is jerky and hard
    # on the real hand. Same term as the lift (-10) and velocity (-1) tasks; -1.0 here
    # since the hand legitimately uses most of its flexion range for gaiting.
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Effort penalty: sum of squared per-joint torque fractions (tau/tau_max)^2, so
    # each joint is penalized by how hard it works relative to its own limit rather
    # than the big proximal joints dominating a raw sum of squares.
    "joint_torque": RewardTermCfg(
      func=reorient_mdp.NormalizedJointTorquePenalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    # Gentleness: penalize intra-hand self-contact force. Kept small (-0.01) on purpose.
    # The `posture` reward already removes most self-contact by spreading the fingers
    # onto the cube instead of each other, so this only needs to trim the residual.
    # There is a real contention with all-finger use: a stronger weight (-0.05/-0.1)
    # finds it cheaper to retract a finger entirely than to hold it gently, partly
    # disabling it despite posture; -0.01 crushes the bulk self-force while keeping all
    # fingers engaged. (The dynamic impact spikes are better addressed by slowness than
    # by a larger force weight.)
    "finger_self_force": RewardTermCfg(
      func=reorient_mdp.self_contact_force,
      weight=-0.01,
      params={"sensor_name": "finger_self_contact"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Dynamic hand-cage drop: cube outside the palm-frame AABB of the fingertips
    # + wrist for N steps. Robust to hand pitch, omnidirectional, and debounced.
    # The palm body and cage sites are filled in per-robot.
    "cube_dropped": TerminationTermCfg(
      func=reorient_mdp.cage_drop,
      params={
        "object_name": "cube",
        "margin": 0.02,
        "max_outside_steps": 10,
        # Don't count drops during the 0.4 s (20-step) action warmup, while the
        # cube is dropping and settling into the cage.
        "grace_steps": 20,
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=(), site_names=()
        ),  # Set per-robot.
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=None,
      num_envs=1,
      env_spacing=1.0,
      extent=0.3,
      # Shrink decoration scale (force/contact arrow width, frames) for this small
      # hand. Mean body size is ~3 cm; 1 cm keeps decorations readable, not bulky.
      meansize=0.01,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=0.6,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=150,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=0.005,
        integrator="implicitfast",
        # Elliptic friction cone + high impratio for the soft-finger grasp. impratio
        # hardens the friction constraint relative to normal, which kills the
        # tangential creep (fingertips sliding across the cube during a static hold):
        # measured ~5x less slip at impratio 10 vs the pyramidal/impratio-1 default,
        # with the cube held as well or better. impratio is only principled on the
        # elliptic cone. The cost is small -- on an idle GPU at condim 4, elliptic +
        # impratio 10 is only ~7% slower than pyramidal + impratio 1 (the elliptic
        # cone itself is ~2%, impratio the other ~5%). If a training run shows NaNs /
        # conditioning trouble on float32, drop impratio to 5 (loses little slip).
        cone="elliptic",
        impratio=10,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
