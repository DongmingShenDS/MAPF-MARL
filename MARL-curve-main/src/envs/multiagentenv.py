class MultiAgentEnv(object):
    """MultiAgentEnv, the base of customized envs"""

    def step(self, actions):
        """ Handle Transition. Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        """ Returns the global state (all agent states) as a list """
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the global state """
        raise NotImplementedError

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list """
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states """
        raise NotImplementedError

    def render(self):
        """ Render something when have a monitor """
        raise NotImplementedError

    def close(self):
        """ Close the environment """
        raise NotImplementedError

    def seed(self):
        """ Set seed for the environment """
        raise NotImplementedError

    def save_replay(self):
        """ Save a replay """
        raise NotImplementedError

    def get_env_info(self):
        """ Return all key information about the environment """
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
