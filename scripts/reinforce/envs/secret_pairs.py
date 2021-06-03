import gym
import numpy as np


class SecretPairs(gym.Env):
    def __init__(self, nagents, nwords, single_timestep, change_freq, share_reward):
        super(SecretPairs, self).__init__()
        self.nagents = nagents
        if self.nagents % 2 != 0:
            raise ValueError('Must have an even number of agents.')
        self.nwords = nwords
        self.num_teams = 1
        self.team_num_agents = [self.nagents]
        self.action_space = gym.spaces.Discrete(self.nwords)
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.nagents // 2 + self.nwords,),
            dtype=np.float32
        )
        self.p2p_observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.nagents // 2,),
            dtype=np.float32
        )
        self.single_timestep = single_timestep
        self.change_freq = change_freq
        self.share_reward = share_reward

    def set_obs(self):
        self.secret_words = np.random.randint(low=0, high=self.nwords, size=(self.nagents // 2,))

        pair_ids = np.array(list(range(self.nagents // 2)) + list(range(self.nagents // 2)))
        id_obs = np.zeros((self.nagents, self.nagents // 2), dtype=np.float32)
        id_obs[range(self.nagents), pair_ids] = 1

        obs = np.zeros((self.nagents, self.nwords), dtype=np.float32)
        obs[range(self.nagents // 2), self.secret_words] = 1

        obs = np.concatenate([id_obs, obs], axis=1)
        self.obs = obs

        self.p2p_obs = np.stack([id_obs for _ in range(self.nagents)], axis=0)

        self.all_secret_words = np.concatenate([self.secret_words, self.secret_words])
        self.stat['gt_actions'] = self.all_secret_words[None]

    def reset(self, timestep=None):
        self.timestep = 0
        self.stat = {'agent_{}_return'.format(i): 0 for i in range(self.nagents)}
        self.set_obs()
        return self.obs[None], self.p2p_obs[None], np.ones((1, self.nagents), dtype=np.int64)

    def step(self, action):
        action = action[0][0]
        reward = (action == self.all_secret_words).astype(np.float32)
        if self.share_reward:
            reward = reward[:self.nagents // 2] + reward[self.nagents // 2:]
            reward = np.concatenate([reward, reward])
        for i in range(self.nagents):
            self.stat['agent_{}_return'.format(i)] += reward[i]
        self.timestep += 1
        if self.timestep % self.change_freq == 0:
            self.set_obs()

        done = self.single_timestep and self.timestep == 2
        alive = np.ones((1, self.nagents), dtype=np.int64) * float(not done)
        return self.obs[None], self.p2p_obs[None], reward[None], alive, done, dict(self.stat)
