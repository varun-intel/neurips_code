import gym


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, time_limit):
        super(TimeLimitWrapper, self).__init__(env)
        self.time_limit = time_limit
        self._timestep = 0

    def set_timestep(self, timestep):
        self._timestep = timestep

    def reset(self):
        self._steps = 0
        self._prev_info = None
        return self.env.reset(timestep=self._timestep)

    def step(self, action):
        self._steps += 1
        obs, p2p_obs, rew, alive_mask, done, info = self.env.step(action)
        if self._steps > self.time_limit:
            timed_out = True
            info = self._prev_info
        else:
            timed_out = False
            self._prev_info = dict(info)
        if done or timed_out:
            obs, p2p_obs, alive_mask = self.reset()
        return obs, p2p_obs, rew, alive_mask, done, timed_out, info
