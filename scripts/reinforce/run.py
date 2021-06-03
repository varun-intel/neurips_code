import json
import torch
import tabulate
import os
import numpy as np
import git
import csv
import itertools
import copy
import time

import utils
from envs.vec_env import SubprocVecEnv, DummyVecEnv


def make_env(config, env_config):
    if env_config['name'] == 'secret':
        from envs.secret import Secret
        return Secret(**env_config['kwargs'])
    elif env_config['name'] == 'secret_pairs':
        from envs.secret_pairs import SecretPairs
        return SecretPairs(**env_config['kwargs'])
    elif env_config['name'] == 'meet':
        from envs.meet import Meet
        return Meet(**env_config['kwargs'])
    else:
        raise ValueError('Unknown environment {}.'.format(env_config['name']))


def make_wrapper(env, config, wrapper_config):
    if wrapper_config['name'] == 'comm_monitor':
        from envs.comm_monitor import CommMonitor
        penalties = [config['teams'][team_idx]['final_comm_penalty'] for team_idx in range(config['env']['num_teams'])]
        return CommMonitor(env, penalties=penalties, **wrapper_config['kwargs'])
    else:
        raise ValueError('Unknown wrapper {}.'.format(wrapper_config['name']))


def make_env_with_wrappers(config, env_config, wrapper_configs):
    env = make_env(config, env_config)
    for wrapper_config in wrapper_configs:
        env = make_wrapper(env, config, wrapper_config)
    return env


def get_curriculum_value(timestep, final_value, curr_end):
    if curr_end == 0:
        comm_penalty = final_value
    else:
        timestep = min(timestep, curr_end)
        comm_penalty = final_value * (timestep / curr_end)
    return comm_penalty


def masked_mean(tensor, mask):
    if tensor.shape != mask.shape:
        raise ValueError('Mismatched shapes in masked mean.')
    num_elem_valid = mask.sum()
    num_elem_valid[num_elem_valid == 0] = 1
    mask = mask.float() / num_elem_valid
    return (tensor * mask).sum()


def masked_var(tensor, mask):
    mean = masked_mean(tensor, mask)
    tensor = (tensor - mean).pow(2)
    return masked_mean(tensor, mask)


def mix_rewards(tensor, ratio, mask, dim):
    num_elem_valid = mask.sum(dim=dim, keepdims=True)
    num_elem_valid[num_elem_valid == 0] = 1
    mask = mask.float() / num_elem_valid.float()
    mean = (tensor * mask).sum(dim=dim, keepdims=True)
    return ratio * mean + (1 - ratio) * tensor


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config

        self.log_dir = utils.create_log_dir(self.config['log_dir'])
        with open(os.path.join(self.log_dir, 'config.json'), 'w+') as f:
            json.dump(self.config, f, indent=4, sort_keys=True)
        with open(os.path.join(self.log_dir, 'params.json'), 'w+') as f:
            json.dump(utils.flatten_dict(self.config), f, indent=4, sort_keys=True)

        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_dirty = repo.is_dirty()
        git_info = dict(git_sha=git_sha, git_dirty=git_dirty)
        with open(os.path.join(self.log_dir, 'gitinfo.json'), 'w+') as f:
            json.dump(git_info, f, indent=4, sort_keys=True)
        os.makedirs(os.path.join(self.log_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)

    def reset_buffers(self):
        self.buffers = []
        for _ in range(self.num_teams):
            self.buffers.append({
                'actions': [],
                'action_dists': [],
                'comm_masks': [],
                'gate_log_probs': [],
                'gate_entropies': [],
                'reward': [],
                'alive_masks': [],
                'values': [],
                'dones': [],
                'timed_out': [],
                'gt_actions': [],
                'comm_penalty_masks': []
            })

    def run(self):
        log_file = os.path.join(self.log_dir, 'log.txt')
        self.progress_file = open(os.path.join(self.log_dir, 'progress.csv'), 'w+')
        with utils.capture_output(log_file, log_file):
            self.run_inner()
        self.progress_file.close()

    def run_inner(self):
        utils.print_bar()
        print('Config')
        print(json.dumps(self.config, indent=4, sort_keys=True))
        utils.print_bar()
        print()

        self.num_teams = self.config['env']['num_teams']
        if len(self.config['teams']) == 0:
            self.config['teams'] = [{} for _ in range(self.num_teams)]

        for team_idx in range(self.num_teams):
            self.config['teams'][team_idx] = utils.deep_update_dict(
                copy.deepcopy(self.config['default_teams']),
                copy.deepcopy(self.config['teams'][team_idx])
            )

        env = make_env_with_wrappers(self.config, self.config['env'], self.config['env_wrappers'])
        self.team_num_agents = env.team_num_agents
        self.observation_space = env.observation_space
        self.p2p_observation_space = env.p2p_observation_space
        self.action_space = env.action_space
        utils.print_bar()
        print('Environment')
        print('Num teams: {}'.format(self.num_teams))
        for team_idx in range(self.num_teams):
            print('Num agents in team {}: {}'.format(team_idx + 1, self.team_num_agents[team_idx]))
        print('Observation space: {}'.format(self.observation_space))
        print('P2P Observation space: {}'.format(self.p2p_observation_space))
        print('Action space: {}'.format(self.action_space))
        utils.print_bar()
        print()

        if self.config['mode'] == 'train':
            self.train()
        elif self.config['mode'] == 'test':
            self.test()
        elif self.config['mode'] == 'render':
            self.test_render()
        else:
            raise ValueError('Unknown run mode {}.'.format(self.config['mode']))

    def create_networks(self):
        from network import PolicyNetwork, ValueNetwork
        self.device = torch.device(self.config['device'])
        self.policy_nets = []
        self.value_nets = []
        self.policy_optimizers = []
        self.value_optimizers = []
        for team_idx in range(self.num_teams):
            team_config = self.config['teams'][team_idx]
            optim_cls = getattr(torch.optim, team_config['optim_name'])
            policy_net = PolicyNetwork(
                obs_size=np.product(self.observation_space.shape),
                p2p_obs_size=np.product(self.p2p_observation_space.shape),
                action_space=self.action_space,
                hidden_size=team_config['policy_hidden_size'],
                message_size=team_config['message_size'],
                key_size=team_config['key_size'],
                value_size=team_config['value_size'],
                num_agents=self.team_num_agents[team_idx],
                global_comm_gate=team_config['global_comm_gate'],
                p2p_comm_gate=team_config['p2p_comm_gate'],
                num_comm_rounds=team_config['num_comm_rounds'],
                comm_type=team_config['comm_type'],
                comm_gate_gen=team_config['comm_gate_gen'],
                temperature=team_config['temperature'],
                activation=team_config['activation'],
                forward_messages=team_config['forward_messages'],
                use_select_comm_one=team_config['use_select_comm_one'],
                p2p_key_size=team_config['p2p_key_size'],
                p2p_num_keys=team_config['p2p_num_keys'],
                p2p_gen_type=team_config['p2p_gen_type']
            ).to(self.device)
            policy_optimizer = optim_cls(policy_net.parameters(), **team_config['optim_kwargs'])
            value_net = ValueNetwork(
                team_num_agents=self.team_num_agents,
                team_idx=team_idx,
                num_comm_rounds=team_config['num_comm_rounds'],
                comm_type=team_config['value_comm_type'],
                obs_size=np.product(self.observation_space.shape),
                hidden_size=team_config['value_hidden_size'],
                message_size=team_config['message_size'],
                key_size=team_config['key_size'],
                value_size=team_config['value_size'],
                activation=team_config['activation'],
                shared_comm_penalty_ratio=team_config['shared_comm_penalty_ratio']
            ).to(self.device)
            value_optimizer = optim_cls(value_net.parameters(), **team_config['optim_kwargs'])
            self.policy_nets.append(policy_net)
            self.value_nets.append(value_net)
            self.policy_optimizers.append(policy_optimizer)
            self.value_optimizers.append(value_optimizer)

            utils.print_bar()
            print('Team {}. Policy Network'.format(team_idx))
            print(policy_net)
            print('Team {}. Value Network'.format(team_idx))
            print(value_net)
            utils.print_bar()

        if self.config['load_checkpoint_file'] != '':
            checkpoint = torch.load(self.config['load_checkpoint_file'], map_location=self.device)
            print('Loaded checkpoint from {}'.format(self.config['load_checkpoint_file']))
            for team_idx in range(self.num_teams):
                self.policy_nets[team_idx].load_state_dict(checkpoint['teams'][team_idx]['policy'])
                self.value_nets[team_idx].load_state_dict(checkpoint['teams'][team_idx]['value'])

    def run_eval_policy_networks(self, obs, p2p_obs, comm_penalty_masks, alive_masks, policy_hidden_states, is_first_step):
        device_obs = [torch.from_numpy(np.array(team_obs)).float().to(self.device)[None] for team_obs in obs]
        device_p2p_obs = [torch.from_numpy(np.array(team_p2p_obs)).float().to(self.device)[None] for team_p2p_obs in p2p_obs]
        device_comm_penalty_masks = [torch.from_numpy(np.array(team_comm_penalty_masks)).float().to(self.device) for team_comm_penalty_masks in comm_penalty_masks]
        device_alive_masks = [torch.from_numpy(np.array(team_alive_masks)).to(self.device)[None] for team_alive_masks in alive_masks]
        if is_first_step:
            device_masks = torch.tensor([0], dtype=torch.float32, device=self.device)
        else:
            device_masks = torch.tensor([1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions = []
            comm_masks = []
            hidden_states = []
            for team_idx in range(self.num_teams):
                policy_output = self.policy_nets[team_idx].act(
                    obs=device_obs[team_idx],
                    p2p_obs=device_p2p_obs[team_idx],
                    comm_penalty_masks=device_comm_penalty_masks[team_idx],
                    hidden_states=policy_hidden_states[team_idx],
                    prev_step_masks=device_masks,
                    alive_masks=device_alive_masks[team_idx],
                    stochastic=self.config['eval_stochastic'],
                    is_train=False
                )
                actions.append(policy_output.actions[0].cpu().numpy())
                comm_masks.append(policy_output.comm_masks[0].cpu().numpy())
                hidden_states.append(policy_output.hidden_states)
            return [actions, comm_masks], hidden_states

    def run_train_networks(self, obs, p2p_obs, comm_penalty_masks, masks, alive_masks, policy_hidden_states, value_hidden_states):
        obs, p2p_obs, alive_masks = list(zip(*obs)), list(zip(*p2p_obs)), list(zip(*alive_masks))
        device_obs = [torch.from_numpy(np.array(team_obs)).float().to(self.device) for team_obs in obs]
        device_p2p_obs = [torch.from_numpy(np.array(team_p2p_obs)).float().to(self.device) for team_p2p_obs in p2p_obs]
        device_comm_penalty_masks = [torch.from_numpy(np.array(team_comm_penalty_masks)).float().to(self.device) for team_comm_penalty_masks in comm_penalty_masks]
        device_alive_masks = [torch.from_numpy(np.array(team_alive_masks)).to(self.device) for team_alive_masks in alive_masks]
        device_masks = torch.from_numpy(np.array(masks)).float().to(self.device)
        step_actions = []
        step_action_dists = []
        step_comm_masks = []
        step_gate_log_probs = []
        step_gate_entropies = []
        step_values = []
        new_policy_hidden_states = []
        new_value_hidden_states = []
        for team_idx in range(self.num_teams):
            policy_output = self.policy_nets[team_idx].act(
                obs=device_obs[team_idx],
                p2p_obs=device_p2p_obs[team_idx],
                comm_penalty_masks=device_comm_penalty_masks[team_idx],
                hidden_states=policy_hidden_states[team_idx],
                prev_step_masks=device_masks,
                alive_masks=device_alive_masks[team_idx],
                stochastic=True,
                is_train=True
            )
            new_policy_hidden_states.append(policy_output.hidden_states)
            step_actions.append(policy_output.actions)
            step_action_dists.append(policy_output.action_dists)
            step_comm_masks.append(policy_output.comm_masks)
            step_gate_log_probs.append(policy_output.gate_log_probs)
            step_gate_entropies.append(policy_output.gate_entropies)

        for team_idx in range(self.num_teams):
            value_output = self.value_nets[team_idx](
                obs=device_obs,
                comm_penalty_masks=device_comm_penalty_masks,
                comm_masks=step_comm_masks,
                hidden_states=value_hidden_states[team_idx],
                prev_step_masks=device_masks,
                alive_masks=device_alive_masks
            )
            new_value_hidden_states.append(value_output.hidden_states)
            step_values.append(value_output.values)
        policy_hidden_states = new_policy_hidden_states
        value_hidden_states = new_value_hidden_states

        return step_actions, step_action_dists, step_comm_masks, step_gate_log_probs, step_gate_entropies, step_values, policy_hidden_states, value_hidden_states

    def run_train_last_values(self, obs, p2p_obs, comm_penalty_masks, masks, alive_masks, policy_hidden_states, value_hidden_states):
        obs, p2p_obs, alive_masks = list(zip(*obs)), list(zip(*p2p_obs)), list(zip(*alive_masks))
        device_obs = [torch.from_numpy(np.array(team_obs)).float().to(self.device) for team_obs in obs]
        device_p2p_obs = [torch.from_numpy(np.array(team_p2p_obs)).float().to(self.device) for team_p2p_obs in p2p_obs]
        device_comm_penalty_masks = [torch.from_numpy(np.array(team_comm_penalty_masks)).float().to(self.device) for team_comm_penalty_masks in comm_penalty_masks]
        device_alive_masks = [torch.from_numpy(np.array(team_alive_masks)).to(self.device) for team_alive_masks in alive_masks]
        device_masks = torch.from_numpy(np.array(masks)).float().to(self.device)
        step_comm_masks = []
        step_values = []

        for team_idx in range(self.num_teams):
            policy_output = self.policy_nets[team_idx].act(
                obs=device_obs[team_idx],
                p2p_obs=device_p2p_obs[team_idx],
                comm_penalty_masks=device_comm_penalty_masks[team_idx],
                hidden_states=policy_hidden_states[team_idx],
                prev_step_masks=device_masks,
                alive_masks=device_alive_masks[team_idx],
                stochastic=True,
                is_train=True
            )
            step_comm_masks.append(policy_output.comm_masks)

        for team_idx in range(self.num_teams):
            team_values, team_value_hidden_states = self.value_nets[team_idx](
                obs=device_obs,
                comm_penalty_masks=device_comm_penalty_masks,
                comm_masks=step_comm_masks,
                hidden_states=value_hidden_states[team_idx],
                prev_step_masks=device_masks,
                alive_masks=device_alive_masks
            )
            step_values.append(team_values)
        return step_values

    def update_team(self, timestep, team_idx):
        team_config = self.config['teams'][team_idx]
        losses = {}
        buffers = self.buffers[team_idx]
        value_optimizer = self.value_optimizers[team_idx]
        policy_optimizer = self.policy_optimizers[team_idx]

        buffers = {k: v for k, v in buffers.items() if len(v) > 0}
        buffers = {k: torch.stack(v) if k not in ['action_dists', 'gt_actions'] else v for k, v in buffers.items()}
        expected_shapes = {
            'actions': (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx]),
            'action_dists': {'length': self.config['traj_length'], 'batch_shape': (self.config['num_train_envs'], self.team_num_agents[team_idx]), 'event_shape': ()},
            'comm_masks': (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx], self.team_num_agents[team_idx], team_config['num_comm_rounds']),
            'gate_log_probs': (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx], team_config['num_comm_rounds']),
            'gate_entropies': (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx], team_config['num_comm_rounds']),
            'reward': (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx]),
            'alive_masks': (self.config['traj_length'] + 1, self.config['num_train_envs'], self.team_num_agents[team_idx]),
            'values': (self.config['traj_length'] + 1, self.config['num_train_envs'], self.team_num_agents[team_idx], team_config['num_comm_rounds'] + 1),
            'dones': (self.config['traj_length'], self.config['num_train_envs']),
            'timed_out': (self.config['traj_length'], self.config['num_train_envs']),
            'comm_penalty_masks': (self.config['traj_length'], self.config['num_train_envs'])
        }
        if team_config['use_supervised_policy_loss']:
            buffers['gt_actions'] = torch.stack(buffers['gt_actions'])
            expected_shapes['gt_actions'] = (self.config['traj_length'], self.config['num_train_envs'], self.team_num_agents[team_idx])
        for k in expected_shapes:
            if k in ['action_dists']:
                if len(buffers[k]) != expected_shapes[k]['length']:
                    raise ValueError('Incorrect length of {}'.format(k))
                for elem in buffers[k]:
                    if elem.batch_shape != expected_shapes[k]['batch_shape']:
                        raise ValueError('Incorrect batch shape of {}'.format(k))
                    if elem.event_shape != expected_shapes[k]['event_shape']:
                        raise ValueError('Incorrect event shape of {}'.format(k))
            else:
                buffer_shape = tuple(buffers[k].shape)
                expected_shape = expected_shapes[k]
                if buffer_shape != expected_shape:
                    raise ValueError('Shape mismatch on key {}. Buffer contains {} but expected {}.'.format(k, buffer_shape, expected_shape))

        elem_valid_mask = buffers['alive_masks'][:-1] * (1 - buffers['timed_out'][..., None])
        losses['batch_size'] = elem_valid_mask.sum()

        if team_config['use_select_comm_one']:
            comm_gate_elem_valid_mask = elem_valid_mask * buffers['comm_penalty_masks'][:, :, None]
        else:
            comm_gate_elem_valid_mask = elem_valid_mask
        losses['comm_batch_size'] = comm_gate_elem_valid_mask.sum()

        # Share rewards
        buffers['reward'] = mix_rewards(buffers['reward'], team_config['shared_reward_ratio'], buffers['alive_masks'][:-1], dim=2)

        # Get the communication penalty value for this timestep
        comm_penalty_factor = get_curriculum_value(timestep=timestep, final_value=team_config['final_comm_penalty'], curr_end=team_config['comm_penalty_curr_end'])
        losses['comm_penalty_factor'] = float(comm_penalty_factor)
        num_comms = buffers['comm_masks'].sum(-1).sum(-1)
        losses['num_comms'] = masked_mean(num_comms, elem_valid_mask)

        if team_config['comm_penalty_type'] == 'linear':
            comm_penalty = num_comms * comm_penalty_factor
        elif team_config['comm_penalty_type'] == 'square':
            comm_penalty = num_comms.pow(2) * comm_penalty_factor
        else:
            raise ValueError('Unknown comm penalty type {}.'.format(team_config['comm_penalty_type']))

        # Share the comm penalty
        shared_comm_penalty = mix_rewards(comm_penalty, team_config['shared_comm_penalty_ratio'], buffers['alive_masks'][:-1], dim=2)

        # Add the communication penalty to the rewards
        if team_config['add_comm_penalty_to_return']:
            buffers['reward'] = buffers['reward'] - shared_comm_penalty * buffers['comm_penalty_masks'][:, :, None]

        # Mix rewards only for the comm network
        comm_mix_rewards = mix_rewards(buffers['reward'], team_config['extra_comm_shared_ratio'], buffers['alive_masks'][:-1], dim=2)
        comm_mix_rewards = comm_mix_rewards[..., None].expand(*comm_mix_rewards.shape, team_config['num_comm_rounds'])
        buffers['reward'] = torch.cat([comm_mix_rewards, buffers['reward'][..., None]], dim=-1)

        # Compute returns
        policy_returns = [buffers['values'][-1].detach() * buffers['alive_masks'][-1][..., None]]
        for i in reversed(range(len(buffers['reward']))):
            step_returns = (
                (1 - buffers['timed_out'][i][..., None, None]) * (
                    buffers['reward'][i] + team_config['gamma'] * (1 - buffers['dones'][i][..., None, None]) * policy_returns[-1]
                ) + buffers['timed_out'][i][..., None, None] * (
                    buffers['values'][i].detach()
                )
            )
            step_returns *= buffers['alive_masks'][i][..., None]
            policy_returns.append(step_returns)
        policy_returns = policy_returns[::-1][:-1]
        policy_returns = torch.stack(policy_returns)
        policy_returns = policy_returns.detach()
        losses['policy_return_mean'] = masked_mean(policy_returns[..., -1], elem_valid_mask)
        losses['comm_policy_return_mean'] = masked_mean(policy_returns[..., :-1].mean(-1), elem_valid_mask)

        # Compute the value loss and optimize the value network
        value_preds = buffers['values'][:-1]
        advantages = policy_returns - value_preds
        value_loss = masked_mean(advantages.pow(2).mean(-1), elem_valid_mask)
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_nets[team_idx].parameters(), team_config['clip_grad_norm'])
        value_optimizer.step()

        losses['value_loss'] = value_loss
        for comm_round in range(team_config['num_comm_rounds'] + 1):
            losses['value_loss_{}'.format(comm_round)] = masked_mean(advantages[..., comm_round].pow(2), elem_valid_mask)
            losses['value_mean_{}'.format(comm_round)] = masked_mean(value_preds[..., comm_round], elem_valid_mask)
            losses['value_explained_var_{}'.format(comm_round)] = (1 - masked_var(advantages[..., comm_round], elem_valid_mask)) / (masked_var(value_preds[..., comm_round], elem_valid_mask) + 1e-20)

        advantages = advantages.detach()

        # Compute the policy loss
        if team_config['use_supervised_policy_loss']:
            supervised_policy_loss = -torch.stack([action_dist.log_prob(gt_action) for action_dist, gt_action in zip(buffers['action_dists'], buffers['gt_actions'])])
            total_policy_loss = supervised_policy_loss = masked_mean(supervised_policy_loss, elem_valid_mask)
            losses['supervised_policy_loss'] = supervised_policy_loss
        else:
            action_entropies = torch.stack([action_dist.entropy() for action_dist in buffers['action_dists']])
            action_log_probs = torch.stack([action_dist.log_prob(action) for action_dist, action in zip(buffers['action_dists'], buffers['actions'])])
            action_policy_loss = masked_mean(-(action_log_probs * advantages[..., -1]), elem_valid_mask)
            total_policy_loss = action_policy_loss - team_config['entropy_coeff'] * masked_mean(action_entropies, elem_valid_mask)
            losses['action_policy_loss'] = action_policy_loss
            losses['action_entropies'] = masked_mean(action_entropies, elem_valid_mask)

        # Add communication losses to the policy loss
        if team_config['use_comm_gate_penalty_loss']:
            comm_penalty_loss = masked_mean(comm_penalty * buffers['comm_penalty_masks'][:, :, None], comm_gate_elem_valid_mask)
            total_policy_loss = total_policy_loss + comm_penalty_loss
            losses['comm_penalty_loss'] = comm_penalty_loss

        if team_config['use_comm_gate_policy_loss']:
            gate_log_probs = buffers['gate_log_probs']
            gate_entropies = buffers['gate_entropies']
            gate_policy_loss = masked_mean(-(gate_log_probs * advantages[..., :-1]).mean(-1), comm_gate_elem_valid_mask)
            total_policy_loss = total_policy_loss + gate_policy_loss - team_config['gate_entropy_coeff'] * masked_mean(gate_entropies.mean(-1), comm_gate_elem_valid_mask)
            losses['gate_entropy'] = masked_mean(gate_entropies.mean(-1), comm_gate_elem_valid_mask)
            losses['gate_policy_loss'] = gate_policy_loss

        # Optimize the policy
        policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_nets[team_idx].parameters(), team_config['clip_grad_norm'])
        policy_optimizer.step()
        losses['total_policy_loss'] = total_policy_loss

        losses = {k: float(v) for k, v in losses.items()}

        return losses

    def test(self):
        self.eval_env = make_env_with_wrappers(self.config, self.config['env'], self.config['env_wrappers'])
        self.create_networks()
        metrics = self.eval()

        def exclude_metric_key(key):
            if key.endswith('/std'):
                return True
            else:
                return False

        print_metrics = {k: v for k, v in metrics.items() if not exclude_metric_key(k)}
        metrics_table = [[k, v] for k, v in print_metrics.items()]
        print(tabulate.tabulate(metrics_table))

    def eval(self):
        comm_penalty_metrics = {}
        for use_comm_penalty in [False, True]:
            comm_penalty_masks = [[use_comm_penalty] for _ in range(self.num_teams)]
            eval_episode_stats = []
            for _ in range(self.config['num_eval_episodes']):
                obs, p2p_obs, alive_masks = self.eval_env.reset()
                policy_hidden_states = [net.reset_hidden_states(1, self.device) for net in self.policy_nets]
                done = False
                for t in range(self.config['max_episode_steps']):
                    actions, policy_hidden_states = self.run_eval_policy_networks(obs, p2p_obs, comm_penalty_masks, alive_masks, policy_hidden_states, t == 0)
                    obs, p2p_obs, reward, alive_masks, done, info = self.eval_env.step(actions)
                    if done:
                        break
                info = copy.deepcopy(info)
                info['timed_out'] = not done
                if 'gt_actions' in info:
                    info.pop('gt_actions')
                eval_episode_stats.append(info)
            if use_comm_penalty:
                prefix = 'pen'
            else:
                prefix = 'nopen'
            comm_penalty_metrics.update(utils.prefix_dict(utils.reduce_losses(eval_episode_stats), prefix))
        return comm_penalty_metrics

    def train(self):
        epoch_losses = [[] for _ in range(self.num_teams)]
        epoch_train_episode_stats = []

        steps = 0
        episodes = 0
        last_detach_t = -1
        last_update_t = -1
        last_eval_step = 0
        last_checkpoint_step = 0
        completed_first_eval = False
        times = {'total': [], 'update': [], 'eval': [], 'env': [], 'forward': []}
        start_total_time = time.time()

        if self.config['vec_env_type'] == 'subproc':
            self.train_env = SubprocVecEnv(
                [lambda: make_env_with_wrappers(self.config, self.config['env'], self.config['env_wrappers']) for _ in range(self.config['num_train_envs'])],
                time_limit=self.config['max_episode_steps'],
                in_series=self.config['num_train_envs'] // self.config['num_train_env_procs']
            )
        elif self.config['vec_env_type'] == 'dummy':
            self.train_env = DummyVecEnv(
                [lambda: make_env_with_wrappers(self.config, self.config['env'], self.config['env_wrappers']) for _ in range(self.config['num_train_envs'])],
                time_limit=self.config['max_episode_steps']
            )
        else:
            raise ValueError('Unknown vec env type {}.'.format(self.config['vec_env_type']))
        self.eval_env = make_env_with_wrappers(self.config, self.config['env'], self.config['env_wrappers'])

        # Create networks after envs to avoid cuda initialization in subprocs.
        self.create_networks()

        env_start_time = time.time()
        obs, p2p_obs, alive_masks = self.train_env.reset()
        self.reset_buffers()
        for team_idx in range(self.num_teams):
            self.buffers[team_idx]['alive_masks'].append(torch.from_numpy(np.array([env_alive_masks[team_idx] for env_alive_masks in alive_masks])).to(self.device))
        env_end_time = time.time()
        times['env'].append(env_end_time - env_start_time)
        obs = list(obs)
        p2p_obs = list(p2p_obs)
        alive_masks = list(alive_masks)
        masks = [False] * self.config['num_train_envs']
        comm_penalty_masks = [[np.random.rand() <= self.config['teams'][team_idx]['use_comm_penalty_prob'] for _ in range(self.config['num_train_envs'])]
                              for team_idx in range(self.num_teams)]
        policy_hidden_states = [net.reset_hidden_states(self.config['num_train_envs'], self.device) for net in self.policy_nets]
        value_hidden_states = [net.reset_hidden_states(self.config['num_train_envs'], self.device) for net in self.value_nets]
        for t in itertools.count():
            self.train_env.set_timestep(steps)

            # Run the networks to obtain actions and values.
            forward_start_time = time.time()
            step_actions, step_action_dists, step_comm_masks, step_gate_log_probs, step_gate_entropies, step_values, policy_hidden_states, value_hidden_states \
                = self.run_train_networks(obs, p2p_obs, comm_penalty_masks, masks, alive_masks, policy_hidden_states, value_hidden_states)
            forward_end_time = time.time()
            times['forward'].append(forward_end_time - forward_start_time)

            # Take an environment step
            cpu_step_actions = [team_actions.detach().cpu().numpy() for team_actions in step_actions]
            cpu_step_actions = list(zip(*cpu_step_actions))
            cpu_step_comm_masks = [team_comm_masks.detach().cpu().numpy() for team_comm_masks in step_comm_masks]
            cpu_step_comm_masks = list(zip(*cpu_step_comm_masks))
            cpu_action_inputs = list(zip(cpu_step_actions, cpu_step_comm_masks))
            env_start_time = time.time()
            next_obs, next_p2p_obs, reward, next_alive_masks, dones, timed_out, infos = self.train_env.step(cpu_action_inputs)
            env_end_time = time.time()
            times['env'].append(env_end_time - env_start_time)
            infos = [dict(info) for info in infos]
            next_obs = list(next_obs)
            next_p2p_obs = list(next_p2p_obs)
            next_alive_masks = list(next_alive_masks)
            steps += sum([1 for i in range(self.config['num_train_envs']) if not timed_out[i]])

            # Add data to the buffers
            for team_idx in range(self.num_teams):
                self.buffers[team_idx]['actions'].append(step_actions[team_idx])
                self.buffers[team_idx]['action_dists'].append(step_action_dists[team_idx])
                self.buffers[team_idx]['comm_masks'].append(step_comm_masks[team_idx])
                self.buffers[team_idx]['gate_log_probs'].append(step_gate_log_probs[team_idx])
                self.buffers[team_idx]['gate_entropies'].append(step_gate_entropies[team_idx])
                self.buffers[team_idx]['reward'].append(torch.from_numpy(np.array([env_reward[team_idx] for env_reward in reward])).to(self.device).float())
                self.buffers[team_idx]['dones'].append(torch.from_numpy(np.array(dones)).to(self.device).float())
                self.buffers[team_idx]['values'].append(step_values[team_idx])
                self.buffers[team_idx]['timed_out'].append(torch.from_numpy(np.array(timed_out)).to(self.device).float())
                self.buffers[team_idx]['alive_masks'].append(torch.from_numpy(np.array([env_alive_masks[team_idx] for env_alive_masks in alive_masks])).to(self.device))
                self.buffers[team_idx]['comm_penalty_masks'].append(torch.from_numpy(np.array(comm_penalty_masks[team_idx])).float().to(self.device))

                if 'gt_actions' in infos[0]:
                    self.buffers[team_idx]['gt_actions'].append(torch.from_numpy(np.array([info['gt_actions'][team_idx] for info in infos])).to(self.device).float())

            obs = next_obs
            p2p_obs = next_p2p_obs
            alive_masks = next_alive_masks

            # Update networks
            if t >= last_update_t + self.config['traj_length']:
                start_update_time = time.time()
                final_alive_masks = [self.buffers[team_idx]['alive_masks'][-1] for team_idx in range(self.num_teams)]
                step_values = self.run_train_last_values(obs, p2p_obs, comm_penalty_masks, masks, alive_masks, policy_hidden_states, value_hidden_states)
                for team_idx in range(self.num_teams):
                    self.buffers[team_idx]['values'].append(step_values[team_idx])
                for team_idx in range(self.num_teams):
                    losses = self.update_team(steps, team_idx)
                    epoch_losses[team_idx].append(losses)
                last_update_t = t
                self.reset_buffers()
                for team_idx in range(self.num_teams):
                    self.buffers[team_idx]['alive_masks'].append(final_alive_masks[team_idx])
                end_update_time = time.time()
                times['update'].append(end_update_time - start_update_time)

                # Must detach after an update
                if t < last_detach_t + self.config['detach_gap']:
                    raise ValueError('Invalid detach gap')

            if t >= last_detach_t + self.config['detach_gap']:
                policy_hidden_states = [self.policy_nets[team_idx].detach_hidden_states(policy_hidden_states[team_idx]) for team_idx in range(self.num_teams)]
                value_hidden_states = [self.value_nets[team_idx].detach_hidden_states(value_hidden_states[team_idx]) for team_idx in range(self.num_teams)]
                last_detach_t = t

            # Reset envs if necessary
            masks = [True] * self.config['num_train_envs']
            for i in range(self.config['num_train_envs']):
                if dones[i] or timed_out[i]:
                    stat = dict(infos[i])
                    stat['timed_out'] = timed_out[i]
                    if 'gt_actions' in stat:
                        stat.pop('gt_actions')
                    epoch_train_episode_stats.append(stat)
                    env_start_time = time.time()
                    env_end_time = time.time()
                    times['env'].append(env_end_time - env_start_time)
                    masks[i] = False
                    for team_idx in range(self.num_teams):
                        comm_penalty_masks[team_idx][i] = np.random.rand() <= self.config['teams'][team_idx]['use_comm_penalty_prob']
                    episodes += 1

            # Evaluate and print metrics
            if steps >= last_eval_step + self.config['eval_freq']:
                start_eval_time = time.time()
                eval_metrics = self.eval()
                end_total_time = end_eval_time = time.time()
                times['eval'].append(end_eval_time - start_eval_time)
                times['total'].append(end_total_time - start_total_time)
                train_metrics = utils.reduce_losses(epoch_train_episode_stats)
                epoch_losses = [utils.reduce_losses(team_epoch_losses) for team_epoch_losses in epoch_losses]
                opt_metrics = {}
                for team_idx in range(self.num_teams):
                    opt_metrics.update(utils.prefix_dict(epoch_losses[team_idx], 'team_{}'.format(team_idx)))

                metrics = {}
                metrics['steps'] = steps
                metrics['episodes'] = episodes
                metrics.update(utils.prefix_dict(train_metrics, 'train'))
                metrics.update(utils.prefix_dict(eval_metrics, 'eval'))
                metrics.update(utils.prefix_dict(opt_metrics, 'opt'))

                for k, v in times.items():
                    v = np.sum(v) if len(v) > 0 else 0
                    metrics['{}_time (s)'.format(k)] = v

                def exclude_metric_key(key):
                    if key.endswith('/std'):
                        return True
                    else:
                        return False

                print_metrics = {k: v for k, v in metrics.items() if not exclude_metric_key(k)}

                metrics_table = [[k, v] for k, v in print_metrics.items()]
                print(tabulate.tabulate(metrics_table))

                if not completed_first_eval:
                    progress_metrics = metrics.keys()
                    progress_writer = csv.DictWriter(self.progress_file, fieldnames=progress_metrics)
                    progress_writer.writeheader()

                progress_metrics = {k: metrics[k] for k in progress_metrics}
                progress_writer.writerow(progress_metrics)
                self.progress_file.flush()

                epoch_train_episode_stats = []
                epoch_losses = [[] for _ in range(self.num_teams)]
                last_eval_step = steps
                completed_first_eval = True
                times = {'total': [], 'update': [], 'eval': [], 'env': [], 'forward': []}
                start_total_time = time.time()

            if steps >= last_checkpoint_step + self.config['checkpoint_freq']:
                timestep_checkpoint_file = os.path.join(self.log_dir, "checkpoints", 'checkpoint_{}.pth'.format(steps))
                last_checkpoint_file = os.path.join(self.log_dir, "checkpoints", 'checkpoint_last.pth')

                checkpoint = {'teams': []}
                for team_idx in range(self.num_teams):
                    team_checkpoint = {
                        'policy': self.policy_nets[team_idx].state_dict(),
                        'value': self.value_nets[team_idx].state_dict()
                    }
                    checkpoint['teams'].append(team_checkpoint)
                torch.save(checkpoint, timestep_checkpoint_file)
                torch.save(checkpoint, last_checkpoint_file)

                last_checkpoint_step = steps

            if steps >= self.config['num_train_steps']:
                break


def main():
    config = utils.get_config()

    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
