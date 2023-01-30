from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class SoloCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_HAA": 0.0,
            "HL_HAA": 0.0,
            "FR_HAA": -0.0,
            "HR_HAA": -0.0,

            "FL_HFE": 0.4,
            "HL_HFE": -0.4,
            "FR_HFE": 0.4,
            "HR_HFE": -0.4,

            "FL_KFE": -0.8,
            "HL_KFE": 0.8,
            "FR_KFE": -0.8,
            "HR_KFE": 0.8,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:????
        stiffness = {'HAA': 3., 'HFE': 3., 'KFE': 3.}  # [N*m/rad]
        damping = {'HAA': 0.2, 'HFE': 0.2, 'KFE': 0.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/solo/urdf/solo12.urdf"
        name = "solo12"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHOULDER", "LEG"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            pass


class SoloCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_solo'
        load_run = -1
