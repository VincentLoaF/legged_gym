from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class SoloCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 108
        num_actions = 12
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands:
        curriculum = False
        min_curriculum_x = -0.5
        max_curriculum_x = 1.5
        min_curriculum_y = -1.
        max_curriculum_y = 1.
        min_curriculum_yaw = -1.
        max_curriculum_yaw = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            # lin_vel_x = [0.5, 0.5] # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            lin_vel_x = [-0.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.04,  # [rad]
            'RL_hip_joint': 0.04,  # [rad]
            'FR_hip_joint': -0.04,  # [rad]
            'RR_hip_joint': -0.04,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.4,  # [rad]
            'RL_calf_joint': -1.4,  # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            'RR_calf_joint': -1.4,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 3.}  # [N*m/rad]
        damping = {'joint': 0.3}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # use_actuator_network = True
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/solo12/solo12_isaac.urdf"
        name = "solo12"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.0]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.

    class rewards(LeggedRobotCfg.rewards):
        # base_height_target = 0.25
        # max_contact_force = 500.
        soft_dof_pos_limit = 0.9
        only_positive_rewards = False
        reward_curriculum = False

        class scales(LeggedRobotCfg.rewards.scales):
            # termination = -0.0
            # tracking_lin_vel = 4.0
            # tracking_ang_vel = 2.0
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.05
            # orientation = -3.0
            # torques = -0.00002
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time = 0.5
            # collision = -2.
            # feet_stumble = -0.0
            # action_rate = -0.01
            # stand_still = -0.

            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0

            lin_vel_tracking = 8.0
            ang_vel_tracking = 4.0
            foot_clearance = -25.0
            foot_slip = -0.1
            roll_pitch_orientation = -3.0
            velocity_z = -1.2
            joint_pose = -0.25
            power_loss = -0.1
            action_smoothness = -0.1

    class noise:
        add_noise = False
        noise_curriculum = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1


class SoloCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        run_name = ''
        experiment_name = 'flat_solo'
        load_run = -1
