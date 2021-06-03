import gym
import numpy as np


class Meet(gym.Env):
    def __init__(self, nagents, dim, vision, reward_type):
        super(Meet, self).__init__()
        self.nagents = nagents
        self.num_teams = 1
        self.team_num_agents = [self.nagents]
        self.dim = dim
        self.vision = vision
        self.reward_type = reward_type

        obs_dim = 2  # agent's position
        obs_dim += 3 * (self.nagents - 1)  # other agents' positions and mask
        obs_dim += 3  # landmark position and mask
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,))
        self.p2p_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,))

        self.action_d = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, timestep=None):
        self.agent_pos = np.random.randint(0, self.dim, size=(self.nagents, 2))
        self.landmark_pos = np.random.randint(0, self.dim, size=(2,))
        self.stat = {'success': False, 'total_reward': 0}
        alive = np.ones((1, self.nagents), dtype=np.float64)
        obs, p2p_obs = self.get_obs()
        return obs, p2p_obs, alive

    def step(self, action):
        action = action[0][0]
        action_d = self.action_d[action]
        self.agent_pos = self.agent_pos + action_d
        self.agent_pos = np.clip(self.agent_pos, 0, self.dim - 1)

        success, reward = self.get_success_and_reward()
        self.stat['success'] |= success
        self.stat['total_reward'] += reward.mean()
        alive = np.ones((1, self.nagents), dtype=np.float64)
        obs, p2p_obs = self.get_obs()
        return obs, p2p_obs, reward[None], alive, False, self.stat

    def get_obs(self):
        obs = np.stack([self.get_agent_obs(i) for i in range(self.nagents)], axis=0)
        p2p_obs = np.stack([self.get_agent_p2p_obs(i) for i in range(self.nagents)], axis=0)
        return obs[None], p2p_obs[None]

    def get_agent_obs(self, agent_idx):
        obs = []
        obs.append(self.agent_pos[agent_idx])

        other_agent_pos = self.agent_pos - self.agent_pos[agent_idx][None]
        other_agent_dist = np.linalg.norm(other_agent_pos, axis=1, ord=1)
        other_agent_idxs = np.argsort(other_agent_dist)
        other_agent_pos = other_agent_pos[other_agent_idxs][1:].copy()
        other_agent_mask = (other_agent_dist[other_agent_idxs] <= self.vision)[1:]
        other_agent_pos[~other_agent_mask] = 0.
        obs.append(other_agent_pos.reshape(-1))
        obs.append(other_agent_mask)

        landmark_pos = self.landmark_pos - self.agent_pos[agent_idx]
        landmark_dist = np.linalg.norm(landmark_pos, ord=1)
        landmark_mask = landmark_dist <= self.vision
        landmark_pos[~landmark_mask] = 0.
        obs.append(landmark_pos)
        obs.append(landmark_mask[None])

        obs = [elem.astype(np.float64) for elem in obs]
        return np.concatenate(obs)

    def get_agent_p2p_obs(self, agent_idx):
        other_agent_pos = self.agent_pos - self.agent_pos[agent_idx][None]
        other_agent_dist = np.linalg.norm(other_agent_pos, axis=1, ord=1)
        other_agent_mask = (other_agent_dist <= self.vision)
        other_agent_pos = other_agent_pos * other_agent_mask[:, None]
        p2p_obs = np.concatenate([other_agent_pos, other_agent_mask[:, None]], axis=1)
        return p2p_obs

    def get_success_and_reward(self):
        agent_landmark_equals = (np.abs(self.agent_pos - self.landmark_pos[None]).sum(1) == 0).astype(np.float32)
        agents_on_landmark = np.sum(agent_landmark_equals)
        success = agents_on_landmark == self.nagents

        if self.reward_type == 'mixed':
            reward = agent_landmark_equals
        elif self.reward_type == 'cooperative':
            reward = agent_landmark_equals * agents_on_landmark
        else:
            raise ValueError('Unknown reward type {}.'.format(self.reward_type))
        return success, reward
