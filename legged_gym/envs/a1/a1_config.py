# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1FlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 108
        num_actions = 12
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands:
        curriculum = True
        min_curriculum_x = -0.5
        max_curriculum_x = 1.5
        min_curriculum_y = -1.
        max_curriculum_y = 1.
        min_curriculum_yaw = -1.
        max_curriculum_yaw = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            # lin_vel_x = [0.5, 0.5] # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            lin_vel_x = [-0.5, 1.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'FL_hip_joint': 0.04,   # [rad]
            # 'RL_hip_joint': 0.04,   # [rad]
            # 'FR_hip_joint': -0.04 ,  # [rad]
            # 'RR_hip_joint': -0.04,   # [rad]
            #
            # 'FL_thigh_joint': 0.7,     # [rad]
            # 'RL_thigh_joint': 0.8,   # [rad]
            # 'FR_thigh_joint': 0.7,     # [rad]
            # 'RR_thigh_joint': 0.8,   # [rad]
            #
            # 'FL_calf_joint': -1.4,   # [rad]
            # 'RL_calf_joint': -1.4,    # [rad]
            # 'FR_calf_joint': -1.4,  # [rad]
            # 'RR_calf_joint': -1.4,    # [rad]
            'FL_hip_joint': 0.04,   # [rad]
            'RL_hip_joint': 0.04,   # [rad]
            'FR_hip_joint': -0.04,  # [rad]
            'RR_hip_joint': -0.04,  # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.4,   # [rad]
            'RL_calf_joint': -1.4,    # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            'RR_calf_joint': -1.4,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.0]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        only_positive_rewards = True
        reward_curriculum = False
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
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
            foot_clearance = -50.0
            foot_slip = -0.1
            roll_pitch_orientation = -3.0
            velocity_z = -1.2
            joint_pose = -0.5
            power_loss = -0.05
            action_smoothness = -0.2

    class noise:
        add_noise = False
        noise_curriculum = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

class A1FlatCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_a1'
