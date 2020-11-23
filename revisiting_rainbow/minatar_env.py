"""MinAtar environment made compatible for Dopamine."""

from dopamine.discrete_domains import atari_lib
from flax import nn
import gin 
import jax 
import jax.numpy as jnp 
import minatar


gin.constant('minatar_env.ASTERIX_SHAPE', (10, 10, 4)) 
gin.constant('minatar_env.BREAKOUT_SHAPE', (10, 10, 4)) 
gin.constant('minatar_env.FREEWAY_SHAPE', (10, 10, 7)) 
gin.constant('minatar_env.SEAQUEST_SHAPE', (10, 10, 10))
gin.constant('minatar_env.SPACE_INVADERS_SHAPE', (10, 10, 6)) 
gin.constant('minatar_env.DTYPE', jnp.float64)


class MinAtarEnv(object):
  def __init__(self, game_name):
    self.env = minatar.Environment(env_name=game_name)
    self.env.n = self.env.num_actions()
    self.game_over = False

  @property
  def observation_space(self):
    return self.env.state_shape()

  @property
  def action_space(self):
    return self.env  # Only used for the `n` parameter.

  @property
  def reward_range(self):
    pass  # Unused

  @property
  def metadata(self):
    pass  # Unused

  def reset(self):
    self.game_over = False
    self.env.reset()
    return self.env.state()

  def step(self, action):
    r, terminal = self.env.act(action)
    self.game_over = terminal
    return self.env.state(), r, terminal, None


@gin.configurable
def create_minatar_env(game_name):
  return MinAtarEnv(game_name)
