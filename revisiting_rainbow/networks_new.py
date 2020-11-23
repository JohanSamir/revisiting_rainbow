"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from jax import random
import math

from jax.tree_util import tree_flatten, tree_map

#---------------------------------------------------------------------------------------------------------------------


env_inf = {"CartPole":{"MIN_VALS": onp.array([-2.4, -5., -math.pi/12., -math.pi*2.]),"MAX_VALS": onp.array([2.4, 5., math.pi/12., math.pi*2.])},
            "Acrobot":{"MIN_VALS": onp.array([-1., -1., -1., -1., -5., -5.]),"MAX_VALS": onp.array([1., 1., 1., 1., 5., 5.])},
            "MountainCar":{"MIN_VALS":onp.array([-1.2, -0.07]),"MAX_VALS": onp.array([0.6, 0.07])}
            }

#---------------------------------------------------------------------------------------------------------------------

gin.constant('jax_networks.LUNALANDER_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.MOUNTAINCAR_OBSERVATION_DTYPE', jnp.float64)

#---------------------------------------------------------------------------------------------------------------------

@gin.configurable
class NoisyNetwork(nn.Module):
  def apply(self, x, features, bias=True, kernel_init=None):
    def sample_noise(shape):
      noise = jax.random.normal(random.PRNGKey(0),shape)
      return noise
    def f(x):
      return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))
    # Initializer of \mu and \sigma 
   
    def mu_init(key, shape):
        low = -1*1/jnp.power(x.shape[1], 0.5)
        high = 1*1/jnp.power(x.shape[1], 0.5)
        return onp.random.uniform(low,high,shape)

    def sigma_init(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)*(0.1 / onp.sqrt(x.shape[1]))

    # Sample noise from gaussian
    p = sample_noise([x.shape[1], 1])
    q = sample_noise([1, features])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = jnp.squeeze(f_q)
    w_mu = self.param('kernel',(x.shape[1], features), mu_init)
    w_sigma = self.param('kernell',(x.shape[1], features),sigma_init)
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_mu = self.param('bias',(features,),mu_init)
    b_sigma = self.param('biass',(features,),sigma_init)
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)
    return jnp.where(bias, ret + b, ret)
  
#---------------------------------------------< DQNNetwork >----------------------------------------------------------

@gin.configurable
class DQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions, net_conf, env, normalize_obs, noisy, dueling, hidden_layer=2, neurons=512):
    del normalize_obs

    if net_conf == 'minatar':
      x = x.squeeze(3)
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))

    elif net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x[None, ...]
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))  # flatten

    elif net_conf == 'classic':
      #classic environments
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = x.reshape((x.shape[0], -1))

    if env is not None:
      x = x - env_inf[env]['MIN_VALS']
      x /= env_inf[env]['MAX_VALS'] - env_inf[env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if noisy:
      def net(x, features):
        return NoisyNetwork(x, features)
    else:
      def net(x, features):
        return nn.Dense(x, features, kernel_init=nn.initializers.xavier_uniform())

    for _ in range(hidden_layer):
      x = net(x, features=neurons)
      x = jax.nn.relu(x)

    adv = net(x, features=num_actions)
    val = net(x, features=1)
    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=num_actions)

    q_values = jnp.where(dueling, dueling_q, non_dueling_q)

    return atari_lib.DQNNetworkType(q_values)

#---------------------------------------------< RainbowDQN >----------------------------------------------------------

@gin.configurable
class RainbowDQN(nn.Module):

  def apply(self, x, num_actions, net_conf, env, normalize_obs, noisy, dueling, num_atoms, support, hidden_layer=2, neurons=512):
    del normalize_obs

    if net_conf == 'minatar':
      x = x.squeeze(3)
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))

    elif net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x[None, ...]
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))  # flatten

    elif net_conf == 'classic':
      #classic environments
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = x.reshape((x.shape[0], -1))


    if env is not None:
      x = x - env_inf[env]['MIN_VALS']
      x /= env_inf[env]['MAX_VALS'] - env_inf[env]['MIN_VALS']
      x = 2.0 * x - 1.0


    if noisy:
      def net(x, features):
        return NoisyNetwork(x, features)
    else:
      def net(x, features):
        return nn.Dense(x, features, kernel_init=nn.initializers.xavier_uniform())


    for _ in range(hidden_layer):
      x = net(x, features=neurons)
      x = jax.nn.relu(x)


    if dueling:
      adv = net(x,features=num_actions * num_atoms)
      value = net(x, features=num_atoms)
      adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
      value = value.reshape((value.shape[0], 1, num_atoms))
      logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=2)

    else:
      x = net(x, features=num_actions * num_atoms)
      logits = x.reshape((x.shape[0], num_actions, num_atoms))
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=2)

    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
    
#---------------------------------------------< QuantileNetwork >----------------------------------------------------------

@gin.configurable
class QuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""

  def apply(self, x, num_actions, net_conf, env, normalize_obs, noisy, dueling, num_atoms,hidden_layer=2, neurons=512):
    del normalize_obs

    if net_conf == 'minatar':
      x = x.squeeze(3)
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = nn.Conv(x, features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))

    elif net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x[None, ...]
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=nn.initializers.xavier_uniform())
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))  # flatten

    elif net_conf == 'classic':
      #classic environments
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = x.reshape((x.shape[0], -1))


    if env is not None:
      x = x - env_inf[env]['MIN_VALS']
      x /= env_inf[env]['MAX_VALS'] - env_inf[env]['MIN_VALS']
      x = 2.0 * x - 1.0


    if noisy:
      def net(x, features):
        return NoisyNetwork(x, features)
    else:
      def net(x, features):
        return nn.Dense(x, features, kernel_init=nn.initializers.xavier_uniform())


    for _ in range(hidden_layer):
      x = net(x, features=neurons)
      x = jax.nn.relu(x)

    if dueling:
      adv = net(x,features=num_actions * num_atoms)
      value = net(x, features=num_atoms)
      adv = adv.reshape((adv.shape[0], num_actions, num_atoms))
      value = value.reshape((value.shape[0], 1, num_atoms))
      logits = value + (adv - (jnp.mean(adv, -1, keepdims=True)))
      probabilities = nn.softmax(logits)
      q_values = jnp.mean(logits, axis=2)

    else:
      x = net(x, features=num_actions * num_atoms)
      logits = x.reshape((x.shape[0], num_actions, num_atoms))
      probabilities = nn.softmax(logits)
      q_values = jnp.mean(logits, axis=2)


    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)