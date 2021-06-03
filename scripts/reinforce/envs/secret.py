import gym
import numpy as np


class Secret(gym.Env):
    def __init__(self, nagents, nwords, single_timestep):
        super(Secret, self).__init__()
        self.nagents = nagents
        self.nwords = nwords
        self.num_teams = 1
        self.team_num_agents = [self.nagents]
        self.action_space = gym.spaces.Discrete(self.nwords)
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.nwords,),
            dtype=np.float32
        )
        self.p2p_observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.nwords,),
            dtype=np.float32
        )
        self.single_timestep = single_timestep

    def reset(self, timestep=None):
        self.timestep = 0
        self.secret_word = np.random.randint(self.nwords)
        self.obs = np.zeros((self.nagents, self.nwords), dtype=np.float32)
        self.p2p_obs = np.zeros((self.nagents, self.nagents, self.nwords), dtype=np.float32)
        self.obs[0, self.secret_word] = 1
        self.stat = {'agent_{}_return'.format(i): 0 for i in range(self.nagents)}
        self.stat['gt_actions'] = np.array([self.secret_word] * self.nagents)[None]
        return self.obs[None], self.p2p_obs[None], np.ones((1, self.nagents), dtype=np.int64)

    def step(self, action):
        action = action[0][0]
        reward = (action == self.secret_word).astype(np.float32)
        for i in range(self.nagents):
            self.stat['agent_{}_return'.format(i)] += reward[i]
        self.timestep += 1

        done = self.single_timestep and self.timestep == 2
        alive = np.ones((1, self.nagents), dtype=np.int64) * float(not done)
        return self.obs[None], self.p2p_obs[None], reward[None], alive, done, dict(self.stat)
