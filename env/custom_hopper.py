"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from SAC.distributions import DegenerateMassDistribution, UniformMassDistribution, NormalMassDistribution

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, udr=False, to_optimize=False, full_obs=False):
        self.full_obs = full_obs
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        
        self.udr = udr
        
        if udr: 
            self.distribution = UniformMassDistribution(low=0.5, high=1.5)
        else:
            self.distribution = DegenerateMassDistribution() # Default: no UDR
        self.to_optimize, self.scale = to_optimize, np.ones(3)
        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7



    def set_distribution(self, distribution):
        self.distribution = distribution

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""

        random_masses = self.original_masses[1:] * self.distribution.sample()

        return random_masses


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def set_scale(self, scale):
        self.scale = scale
        
    def set_masses(self):
        scaled_masses = self.scale_parameters(self.scale)
        self.sim.model.body_mass[2:] = scaled_masses

    def scale_parameters(self, scale):
        scaled_masses = self.original_masses[1:] * scale
        return scaled_masses
    


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        if self.full_obs: 
            ob = self._get_total_obs()
        else:
            ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def _get_total_obs(self):
        """Get current state including x position"""
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        # Reset the masses if using UDR 
        if self.udr: 
            random_masses = self.sample_parameters()
            self.sim.model.body_mass[2:] = random_masses
            
        elif self.to_optimize:
            self.set_masses()

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        self.set_state(qpos, qvel)
        if self.full_obs: 
            ob = self._get_total_obs()
        else:
            ob = self._get_obs()
            
        return ob

    def set_initial_state(self, qpos, qvel):
        """Set the simulator to a specific initial state

        Parameters:
        ----------
        qpos: ndarray,
               desired initial position
        qvel: ndarray,
               desired initial velocity
        """
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

