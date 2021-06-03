import gym
import numpy as np


class CommMonitor(gym.Wrapper):
    def __init__(self, env, penalties, report_pairwise):
        super(CommMonitor, self).__init__(env)
        self.num_teams = env.num_teams
        self.team_num_agents = env.team_num_agents
        self.penalties = penalties
        self.report_pairwise = report_pairwise

    def reset(self, timestep=None):
        self.tstep = 0
        self.stat = {}
        return self.env.reset(timestep=timestep)

    def step(self, action):
        obs, p2p_obs, reward, alive_masks, done, inner_stat = self.env.step(action)
        self.tstep += 1

        comm_masks = action[1]
        num_comms = [comm_mask.sum(-1).sum(-1) for comm_mask in comm_masks]
        for team_idx in range(len(num_comms)):
            for agent_idx in range(len(num_comms[team_idx])):
                key = 'agent_{}_{}_comms'.format(team_idx, agent_idx)
                self.stat[key] = self.stat.get(key, 0) + num_comms[team_idx][agent_idx]

        if self.report_pairwise:
            for team_idx in range(len(comm_masks)):
                for agent_idx in range(len(comm_masks[team_idx])):
                    for other_agent_idx in range(len(comm_masks[team_idx][agent_idx])):
                        key = 'team_{}_from_{}_to_{}_comms'.format(team_idx, agent_idx, other_agent_idx)
                        self.stat[key] = self.stat.get(key, 0) + comm_masks[team_idx][agent_idx][other_agent_idx]

        for team_idx in range(len(num_comms)):
            self.stat['team_{}_comm_penalty'.format(team_idx)] = self.penalties[team_idx]
            team_mean_num_comms = np.mean(num_comms[team_idx])
            for agent_idx in range(len(num_comms[team_idx])):
                key = 'agent_{}_{}_comm_return'.format(team_idx, agent_idx)
                self.stat[key] = self.stat.get(key, 0) + float(reward[team_idx][agent_idx]) - self.penalties[team_idx] * team_mean_num_comms

        self.stat['episode_length'] = self.tstep

        inner_stat = dict(inner_stat)
        inner_stat.update(self.stat)
        return obs, p2p_obs, reward, alive_masks, done, inner_stat
