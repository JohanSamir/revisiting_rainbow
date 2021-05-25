"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from jax import random
import math

from jax.tree_util import tree_flatten, tree_map

#---------------------------------------------------------------------------------------------------------------------

env_inf = {"CartPole":{"MIN_VALS": jnp.array([-2.4, -5., -math.pi/12., -math.pi*2.]),"MAX_VALS": jnp.array([2.4, 5., math.pi/12., math.pi*2.])},
            "Acrobot":{"MIN_VALS": jnp.array([-1., -1., -1., -1., -5., -5.]),"MAX_VALS": jnp.array([1., 1., 1., 1., 5., 5.])},
            "MountainCar":{"MIN_VALS":jnp.array([-1.2, -0.07]),"MAX_VALS": jnp.array([0.6, 0.07])}
            }

prn_inf = {"count":0, "rng2_":None, "rng3_":None}

#---------------------------------------------------------------------------------------------------------------------
class NoisyNetwork(nn.Module):
  features: int
  rng: int
  bias_in: bool

  @nn.compact
  def __call__(self, x):

    def sample_noise(rng_input, shape):
      noise = jax.random.normal(rng_input,shape)
      return noise

    def f(x):
      return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))
    # Initializer of \mu and \sigma 
   
    def mu_init(key, shape, rng):
        low = -1*1/jnp.power(x.shape[-1], 0.5)
        high = 1*1/jnp.power(x.shape[-1], 0.5)
        return random.uniform(rng, shape=shape, dtype=jnp.float32, minval=low, maxval=high)

    def sigma_init(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)*(0.1 / jnp.sqrt(x.shape[-1]))

    rng, rng2, rng3, rng4, rng5 = jax.random.split(self.rng, 5)

    if prn_inf["count"] == 0:
        prn_inf["rng2_"] = rng2
        prn_inf["rng3_"] = rng3
        prn_inf["count"] = prn_inf["count"]+1

    # Sample noise from gaussian
    p = sample_noise(prn_inf["rng2_"], [x.shape[-1], 1])
    q = sample_noise(prn_inf["rng3_"], [1, self.features])
    f_p = f(p); f_q = f(q)

    w_epsilon = f_p*f_q; b_epsilon = jnp.squeeze(f_q)
    w_mu = self.param('kernel', mu_init, (x.shape[-1], self.features), rng4)
    w_sigma = self.param('kernell', sigma_init, (x.shape[-1], self.features))
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_mu = self.param('bias', mu_init, (self.features,), rng5)
    b_sigma = self.param('biass',sigma_init, (self.features,))
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)

    return jnp.where(self.bias_in, ret + b, ret)

#---------------------------------------------< DQNNetwork >----------------------------------------------------------

@gin.configurable
class DQNNetwork(nn.Module):

  num_actions:int
  net_conf: str
  env: str
  normalize_obs:bool
  noisy: bool
  dueling: bool
  initzer:str
  hidden_layer: int
  neurons: int

  @nn.compact
  def __call__(self, x , rng):

    if self.net_conf == 'minatar':
      x = x.squeeze(3)
      x = x.astype(jnp.float32)
      x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1),  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))

    elif self.net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))  # flatten

    elif self.net_conf == 'classic':
      #classic environments
      x = x.astype(jnp.float32)
      x = x.reshape((-1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    adv = net(x, features=self.num_actions, rng=rng)
    val = net(x, features=1, rng=rng)

    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=self.num_actions, rng=rng)

    q_values = jnp.where(self.dueling, dueling_q, non_dueling_q)
    return atari_lib.DQNNetworkType(q_values)
#---------------------------------------------< RainbowDQN >----------------------------------------------------------

@gin.configurable
class RainbowDQN(nn.Module):

  num_actions:int
  net_conf:str
  env:str
  normalize_obs:bool
  noisy:bool
  dueling:bool
  initzer:str
  num_atoms:int
  hidden_layer:int
  neurons:int

  @nn.compact
  def __call__(self, x, support, rng):

    if self.net_conf == 'minatar':
      x = x.squeeze(3)
      x = x.astype(jnp.float32)
      x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1),  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))

    elif self.net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))  # flatten

    elif self.net_conf == 'classic':
      x = x.astype(jnp.float32)
      x = x.reshape((-1))
      
    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    if self.dueling:
      adv = net(x,features=self.num_actions * self.num_atoms, rng=rng)
      value = net(x, features=self.num_atoms, rng=rng)

      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))

      logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)

    else:
      x = net(x, features=self.num_actions * self.num_atoms, rng=rng)
      logits = x.reshape((self.num_actions, self.num_atoms))
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)

    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
    
#---------------------------------------------< QuantileNetwork >----------------------------------------------------------

@gin.configurable
class QuantileNetwork(nn.Module):

  num_actions:int
  net_conf:str
  env:str
  normalize_obs:bool
  noisy:bool
  dueling:bool
  initzer:str
  num_atoms:int
  hidden_layer:int
  neurons:int

  @nn.compact
  def __call__(self, x, rng):

    if self.net_conf == 'minatar':
      x = x.squeeze(3)
      x = x.astype(jnp.float32)
      x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1),  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))

    elif self.net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))  # flatten

    elif self.net_conf == 'classic':
      #classic environments
      x = x.astype(jnp.float32)
      x = x.reshape((-1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    if self.dueling:
      adv = net(x,features=self.num_actions * self.num_atoms, rng=rng)
      value = net(x, features=self.num_atoms,  rng=rng)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))

      logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
      probabilities = nn.softmax(logits)
      q_values = jnp.mean(logits, axis=1)

    else:
      x = net(x, features=self.num_actions * self.num_atoms, rng=rng)
      logits = x.reshape((self.num_actions, self.num_atoms))
      probabilities = nn.softmax(logits)
      q_values = jnp.mean(logits, axis=1)

    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)

#---------------------------------------------< IQ-Network >----------------------------------------------------------
@gin.configurable
class ImplicitQuantileNetwork(nn.Module):

  num_actions:int
  net_conf:str
  env:str
  noisy:bool
  dueling:bool
  initzer:str
  quantile_embedding_dim:int
  hidden_layer:int
  neurons:int

  @nn.compact
  def __call__(self, x, num_quantiles, rng):

    if self.net_conf == 'minatar':
      x = x.squeeze(3)
      x = x.astype(jnp.float32)
      x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1),  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))

    elif self.net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=self.initzer)(x)
      x = jax.nn.relu(x)
      x = x.reshape((-1))  # flatten

    elif self.net_conf == 'classic':
      #classic environments
      x = x.astype(jnp.float32)
      x = x.reshape((-1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(features=state_vector_length,
                            kernel_init=self.initzer)(quantile_net)
    quantile_net = jax.nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    
    adv = net(x,features=self.num_actions, rng=rng)
    val = net(x, features=1, rng=rng)
    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=self.num_actions, rng=rng)
    quantile_values = jnp.where(self.dueling, dueling_q, non_dueling_q)

    return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)