from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TestCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "motor_front_left_hip": 0,
            "motor_front_left_upper_leg": 0,
            "motor_front_left_lower_leg": 0,

            "motor_back_left_hip": 0,
            "motor_back_left_upper_leg": 0,
            "motor_back_left_lower_leg": 0,

            "motor_front_right_hip": 0,
            "motor_front_right_upper_leg": 0,
            "motor_front_right_lower_leg": 0,

            "motor_back_right_hip": 0,
            "motor_back_right_upper_leg": 0,
            "motor_back_right_lower_leg": 0,

        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:????
        control_type = 'P'
        stiffness = {'hip': 3, 'upper_leg': 3, 'lower_leg': 3}  # [N*m/rad]
        damping = {'hip': 0.2, 'upper_leg': 0.2, 'lower_leg': 0.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # use_actuator_network = True
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_ros/urdf/spot.urdf"
        name = "test"
        foot_name = "foot"
        # penalize_contacts_on = ["SHOULDER", "LEG"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    # class domain_rand(LeggedRobotCfg.domain_rand):
    #     randomize_base_mass = True
    #     added_mass_range = [-5., 5.]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.25
        max_contact_force = 500.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            pass


class TestCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_test'
        load_run = -1
