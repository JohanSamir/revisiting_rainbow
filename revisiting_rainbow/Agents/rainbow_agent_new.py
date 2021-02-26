"""Compact implementation of a simplified Rainbow agent in Jax.

Specifically, we implement the following components from Rainbow:

  * n-step updates
  * prioritized replay
  * distributional RL
  * double_dqn
  * noisy
  * dueling

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
from flax import nn 
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(9, 10))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, loss_weights, support, cumulative_gamma, double_dqn):
  """Run a training step."""
  def loss_fn(model, target, loss_multipliers):
    logits = jax.vmap(model)(states).logits
    logits = jnp.squeeze(logits)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    loss = jax.vmap(networks.softmax_cross_entropy_loss_with_logits)(target,  chosen_action_logits)

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  # Use the weighted mean loss for gradient computation
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  if double_dqn:
    target = target_distributionDouble(optimizer.target,target_network, next_states, rewards,  terminals, support, cumulative_gamma)
  else:
    target = target_distribution(target_network, next_states, rewards,  terminals, support, cumulative_gamma)
  
  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, loss), grad = grad_fn(optimizer.target, target, loss_weights)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss


@functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None, None))
def target_distributionDouble(model, target_network, next_states, rewards, terminals,
                        support, cumulative_gamma):
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support

  next_state_target_outputs = model(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values) 
  next_qt_argmax = jnp.argmax(q_values)

  next_dist = target_network(next_states)
  probabilities = jnp.squeeze(next_dist.probabilities) 
  next_probabilities = probabilities[next_qt_argmax]

  return jax.lax.stop_gradient(project_distribution(target_support, next_probabilities, support))

@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
def target_distribution(target_network, next_states, rewards, terminals,
                        support, cumulative_gamma):
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support

  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values) 
  next_qt_argmax = jnp.argmax(q_values)

  probabilities = jnp.squeeze(next_state_target_outputs.probabilities)
  next_probabilities = probabilities[next_qt_argmax]

  return jax.lax.stop_gradient(project_distribution(target_support, next_probabilities, support))


@gin.configurable
class JaxRainbowAgentNew(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,
               
               noisy = False,
               dueling = False,
               initzer = 'variance_scaling',
               net_conf = None,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               num_atoms=51,
               vmax=10.,
               double_dqn=False,
               replay_scheme='prioritized',
               optimizer='adam',
               network=networks.RainbowNetwork,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = jnp.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self._double_dqn = double_dqn
    self._net_conf = net_conf
    self._env = env 
    self._normalize_obs = normalize_obs
    self._hidden_layer= hidden_layer
    self._neurons=neurons 
    self._noisy = noisy
    self._dueling = dueling
    self._initzer = initzer

    super(JaxRainbowAgentNew, self).__init__(
        num_actions=num_actions,
        network=network.partial(num_atoms=num_atoms,
                                support=self._support,
                                net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling,
                                initzer=self._initzer),
       
        epsilon_fn = dqn_agent.identity_epsilon if self._noisy == True else epsilon_fn,
        optimizer=optimizer)

  def _create_network(self, name):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      name: str, this name is passed to the Jax Module.
    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init(self._rng,
                                          name=name,
                                          x=self.state,
                                          num_actions=self.num_actions,
                                          num_atoms=self._num_atoms,
                                          support=self._support)
    return nn.Model(self.network, initial_params)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer, loss, mean_loss = train(
            self.target_network,
            self.optimizer,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._support,
            self.cumulative_gamma,
            self._double_dqn)

        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if self.summary_writer is not None:
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='CrossEntropyLoss',
                                         simple_value=mean_loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Stores the following tuple in the replay buffer (last_observation, action,
    reward, is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)


def project_distribution(supports, weights, target_support):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  Args:
    supports: Jax array of shape (num_dims) defining supports for
      the distribution.
    weights: Jax array of shape (num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

  Returns:
    A Jax array of shape (num_dims) with the projection of a batch
    of (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  v_min, v_max = target_support[0], target_support[-1]
  # `N` in Eq7.
  num_dims = target_support.shape[0]
  # delta_z = `\Delta z` in Eq7.
  delta_z = (v_max - v_min) / (num_dims - 1)
  # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
  clipped_support = jnp.clip(supports, v_min, v_max)
  # numerator = `|clipped_support - z_i|` in Eq7.
  numerator = jnp.abs(clipped_support - target_support[:, None])
  quotient = 1 - (numerator / delta_z)
  # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
  clipped_quotient = jnp.clip(quotient, 0, 1)
  # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
  inner_prod = clipped_quotient * weights
  return jnp.squeeze(jnp.sum(inner_prod, -1))
