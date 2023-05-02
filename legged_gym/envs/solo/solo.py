from typing import Dict

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs import LeggedRobot


class Solo(LeggedRobot):
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_pos2[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel2[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_pos2[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_dof_vel2[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        # k_c = (self.common_step_counter / self.total_steps - 0.25) * 2
        curr_iter = self.common_step_counter // self.cfg_train.runner.num_steps_per_env
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print('*' * 20)
            # print(name)
            # print(rew.shape)
            if rew[0] <= 0. and self.cfg.rewards.reward_curriculum:
                # if self.common_step_counter / self.total_steps < 0.25:
                #     rew *= 0.
                # elif self.common_step_counter / self.total_steps < 0.75:
                #     rew *= k_c
                if curr_iter < 100:
                    rew *= 0
                elif curr_iter < 300:
                    rew *= (curr_iter - 100) / 200
                # if curr_iter < 200:
                #     rew *= 0.5 + curr_iter / 400
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # print('*' * 30)
        # print(self.rew_buf)
        # if self.cfg.rewards.gradually_increase_negative_rewards:
        #     if self.common_step_counter / self.total_steps < 0.25:
        #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        #     elif self.common_step_counter / self.total_steps < 0.75:
        #         k_c = (self.common_step_counter / self.total_steps - 0.25) * 2
        #         self.rew_buf[:] = torch.tensor([rew if rew > 0 else rew * k_c for rew in self.rew_buf[:]])
        # print(self.rew_buf)
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  (self.last_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  (self.last_dof_pos2 - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.last_dof_vel * self.obs_scales.dof_vel,
                                  self.last_dof_vel2 * self.obs_scales.dof_vel,
                                  self.actions,
                                  self.last_actions,
                                  self.commands[:, :3] * self.commands_scale,
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            if self.cfg.noise.noise_curriculum:
                curr_iter = self.common_step_counter // self.cfg_train.runner.num_steps_per_env
                if curr_iter < 100:
                    noise *= 0
                elif curr_iter < 500:
                    noise *= (curr_iter - 100) / 400

            self.obs_buf += noise

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["lin_vel_tracking"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales[
                    "lin_vel_tracking"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.25,
                                                          self.cfg.commands.min_curriculum_x, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.25, 0.,
                                                          self.cfg.commands.max_curriculum_x)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.25,
                                                          self.cfg.commands.min_curriculum_y, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.25, 0.,
                                                          self.cfg.commands.max_curriculum_y)

        if torch.mean(self.episode_sums["ang_vel_tracking"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales[
                    "ang_vel_tracking"]:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.25,
                                                            self.cfg.commands.min_curriculum_yaw, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.25, 0.,
                                                            self.cfg.commands.max_curriculum_yaw)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.gravity * noise_level
        noise_vec[3:6] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:45] = 0.  # previous dof_pos
        noise_vec[46:58] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[58:82] = 0.  # previous dof_vel
        noise_vec[82:106] = 0.  # previous actions
        noise_vec[106:109] = 0.  # commands
        if self.cfg.terrain.measure_heights:
            noise_vec[109:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rb_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.total_steps = self.cfg_train.runner.num_steps_per_env * self.cfg_train.runner.max_iterations
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_pos2 = torch.zeros_like(self.dof_pos)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel2 = torch.zeros_like(self.dof_vel)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    # ------------ reward functions----------------
    def _reward_lin_vel_tracking(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_ang_vel_tracking(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # def _reward_hip_pos(self):
    #     # Penalize hip positions
    #     return torch.sum(torch.square(self.dof_pos[:, (0, 3, 6, 9)] - self.default_dof_pos[:, (0, 3, 6, 9)]), dim=1)

    def _reward_foot_clearance(self):
        feet_pos = self.rb_state[:, self.feet_indices, :3]
        feet_vel = self.rb_state[:, self.feet_indices, 7:10]
        # print(self.rb_state.shape)
        # print(feet_pos.shape)
        # print(feet_vel.shape)
        # print('Position: ', feet_pos)
        # print('Velocity: ', feet_vel)
        threshold = 0.1
        return torch.sum(
            torch.square(feet_pos[:, :, 2] - threshold) * torch.sqrt(torch.norm(feet_vel[:, :, :2], dim=2)), dim=1)

    def _reward_foot_slip(self):
        feet_vel = self.rb_state[:, self.feet_indices, 7:10]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(contact * torch.sum(torch.square(feet_vel[:, :, :2]), dim=2), dim=1)

    def _reward_roll_pitch_orientation(self):
        # Penalize base orientation on roll and pitch direction
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_velocity_z(self):
        # Penalize base linear velocity on z axis
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_joint_pose(self):
        # Penalize dof positions from initial positions
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_power_loss(self):
        # Penalize large power loss
        tau_u, b, K = 0.0477, 0.000135, 4.81
        tau_f = tau_u * torch.sign(self.dof_vel) + b * self.dof_vel
        P_f = tau_f * self.dof_vel
        P_j = torch.square(self.torques + tau_f) / K
        return torch.sum(P_f + P_j, dim=1)

    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
