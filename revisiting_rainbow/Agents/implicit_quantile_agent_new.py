"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer 
from flax import nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import jax.scipy.special as scp
import jax.lax



@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, None, None, None, None, None),
    out_axes=(None, 0))
def target_quantile_values_fun(online_network, target_network,
                           next_states, rewards, terminals,
                           num_tau_prime_samples, num_quantile_samples,
                           cumulative_gamma, double_dqn, rng):
  """Build the target for return values at given quantiles.
  Args:
    online_network: Jax Module used for the online network.
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    num_tau_prime_samples: int, number of tau' samples (static_argnum).
    num_quantile_samples: int, number of quantile samples (static_argnum).
    cumulative_gamma: float, cumulative gamma to use (static_argnum).
    double_dqn: bool, whether to use double DQN (static_argnum).
    rng: Jax random number generator.
  Returns:
    Jax random number generator.
    The target quantile values.
  """
  rewards = jnp.tile(rewards, [num_tau_prime_samples])
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  # Compute Q-values which are used for action selection for the next states
  # in the replay buffer. Compute the argmax over the Q-values.
  if double_dqn:
    outputs_action = online_network(next_states,
                                    num_quantiles=num_quantile_samples,
                                    rng=rng1)
  else:
    outputs_action = target_network(next_states,
                                    num_quantiles=num_quantile_samples,
                                    rng=rng1)
  target_quantile_values_action = outputs_action.quantile_values
  target_q_values = jnp.squeeze(
      jnp.mean(target_quantile_values_action, axis=0))
  # Shape: batch_size.
  next_qt_argmax = jnp.argmax(target_q_values)
  # Get the indices of the maximium Q-value across the action dimension.
  # Shape of next_qt_argmax: (num_tau_prime_samples x batch_size).
  next_state_target_outputs = target_network(
      next_states,
      num_quantiles=num_tau_prime_samples,
      rng=rng2)
  next_qt_argmax = jnp.tile(next_qt_argmax, [num_tau_prime_samples])

  target_quantile_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target_outputs.quantile_values,
                                  next_qt_argmax))

  target_quantile_vals = rewards + gamma_with_terminal * target_quantile_vals
  # We return with an extra dimension, which is expected by train.
  
  return rng, jax.lax.stop_gradient(target_quantile_vals[:, None])


def stable_scaled_log_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  tau_lse = max_x + tau * jnp.log(jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
  return x - tau_lse

def stable_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  return jax.nn.softmax(y/tau, axis=axis)


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None, None, None, None,None, None, None,None),
    out_axes=(None, 0))

def munchau_target_quantile_values_fun(online_network, target_network,
                           states,actions,next_states, rewards, terminals,
                           num_tau_prime_samples, num_quantile_samples,
                           cumulative_gamma, double_dqn, rng,tau,alpha,clip_value_min,num_actions):
  #Build the munchausen target for return values at given quantiles.
  del double_dqn

  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])
  
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  #------------------------------------------------------------------------
  replay_net_target_outputs = target_network(next_states,num_quantiles=num_tau_prime_samples, rng=rng)
  replay_net_target_quantile_values =  replay_net_target_outputs.quantile_values

  target_next_action = target_network(next_states,num_quantiles=num_quantile_samples, rng=rng1)
  target_next_quantile_values_action = target_next_action.quantile_values
  _replay_next_target_q_values = jnp.squeeze(jnp.mean(target_next_quantile_values_action, axis=0))

  outputs_action = target_network(states,num_quantiles=num_quantile_samples, rng=rng2)
  q_state_values = outputs_action.quantile_values
  _replay_target_q_values = jnp.squeeze(jnp.mean(q_state_values, axis=0))
  #------------------------------------------------------------------------

  replay_action_one_hot = jax.nn.one_hot(actions,num_actions)
  replay_next_log_policy = stable_scaled_log_softmax(_replay_next_target_q_values, tau, axis=0)
  replay_next_policy =  stable_softmax(_replay_next_target_q_values,tau, axis=0)
  replay_log_policy =  stable_scaled_log_softmax(_replay_target_q_values, tau, axis=0)

  #------------------------------------------------------------------------

  tau_log_pi_a = jnp.sum(replay_log_policy * replay_action_one_hot, axis=0)
  tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=clip_value_min,a_max=0)
  munchausen_term = alpha * tau_log_pi_a

  rewards = rewards + munchausen_term
  rewards = jnp.tile(rewards, [num_tau_prime_samples])

  weighted_logits = (replay_next_policy * (replay_net_target_quantile_values-replay_next_log_policy))

  target_quantile_values = jnp.sum(weighted_logits, axis=1)
  target_quantile_values = rewards + gamma_with_terminal * target_quantile_values

  return rng, jax.lax.stop_gradient(target_quantile_values[:, None])


@functools.partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12,13,14,15,16,17))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, target_opt, num_tau_samples, num_tau_prime_samples,
          num_quantile_samples, cumulative_gamma, double_dqn, kappa, tau,alpha,clip_value_min, num_actions,rng):
  """Run a training step."""
  def loss_fn(model, rng_input, target_quantile_vals,  mean_loss=True):
    model_output = jax.vmap(
        lambda m, x, y, z: m(x=x, num_quantiles=y, rng=z),
        in_axes=(None, 0, None, None))(
            model, states, num_tau_samples, rng_input)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    chosen_action_quantile_values = jax.vmap(lambda x, y: x[:, y][:, None])(
        quantile_values, actions)
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_vals[:, :, None, :] -
                      chosen_action_quantile_values[:, None, :, :])
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2)
    huber_loss_case_two = (
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))
    huber_loss = huber_loss_case_one + huber_loss_case_two
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(quantiles[:, None, :, :],
                         [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))) * huber_loss) / kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = jnp.sum(quantile_huber_loss, axis=2)
    loss = jnp.squeeze(jnp.mean(loss, axis=1), axis=-1)

    if mean_loss:
      loss = jnp.mean(loss)

    return loss

  if target_opt == 0:
      rng, target_quantile_vals = target_quantile_values_fun(
      optimizer.target,
      target_network,
      next_states,
      rewards,
      terminals,
      num_tau_prime_samples,
      num_quantile_samples,
      cumulative_gamma,
      double_dqn,
      rng)

  elif target_opt == 1:
      rng, target_quantile_vals = munchau_target_quantile_values_fun(
      optimizer.target,
      target_network,
      states,
      actions,
      next_states,
      rewards,
      terminals,
      num_tau_prime_samples,
      num_quantile_samples,
      cumulative_gamma,
      double_dqn,
      rng,
      tau,
      alpha,
      clip_value_min,
      num_actions
      )

  else:
    print('error')

  grad_fn = jax.value_and_grad(loss_fn)
  rng, rng_input = jax.random.split(rng)
  mean_loss, grad = grad_fn(optimizer.target, rng_input, target_quantile_vals)
  loss = loss_fn(optimizer.target, rng_input, target_quantile_vals, mean_loss=False)
  optimizer = optimizer.apply_gradient(grad)

  return rng, optimizer, loss, mean_loss


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 10, 11, 12, 13))
def select_action(network, state, rng, num_quantile_samples, num_actions,
                  eval_mode, epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, interact,tau, model):
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network: Jax Module to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    num_quantile_samples: int, number of quantile samples (static_argnum).
    num_actions: int, number of actions (static_argnum).
    eval_mode: bool, whether we are in eval mode (static_argnum).
    epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
    epsilon_train: float, epsilon value to use in train mode (static_argnum).
    epsilon_decay_period: float, decay period for epsilon value for certain
      epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
    training_steps: int, number of training steps so far.
    min_replay_history: int, minimum number of steps in replay buffer
      (static_argnum).
    epsilon_fn: function used to calculate epsilon value (static_argnum).

  Returns:
    Jax random number generator.
    int, the selected action.
  """
  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))
  rng, rng1, rng2 = jax.random.split(rng, num=3)

  if interact == 'stochastic':

    state = jnp.expand_dims(state, axis=0)
    net_outputs = network(state, num_quantiles=num_quantile_samples, rng=rng2).quantile_values
    q_values = jnp.mean(net_outputs,axis=0)
    policy_logits = jax.nn.softmax(q_values/tau, axis=0) 
    key = jax.random.PRNGKey(seed=0)
    stochastic_action = jax.random.categorical(key, policy_logits, axis=0, shape=None)
    selected_action = stochastic_action

  elif interact == 'greedy':
    selected_action = jnp.argmax(jnp.mean(
                            network(state,
                                    num_quantiles=num_quantile_samples,
                                    rng=rng2).quantile_values, axis=0),
                                   axis=0)
  else:
    print('error interact')
  
  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        selected_action)


@gin.configurable
class JaxImplicitQuantileAgentNew(dqn_agent.JaxDQNAgent):
  """An extension of Rainbow to perform implicit quantile regression."""

  def __init__(self,
               num_actions,

               tau,
               alpha=1,
               clip_value_min=-10,
               target_opt=0,
               interact = 'greedy',

               net_conf = None,
               env = "CartPole",
               hidden_layer=2, 
               neurons=512,
               noisy = False,
               dueling = False,
               initzer = 'variance_scaling',

               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.ImplicitQuantileNetwork,
               kappa=1.0,
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               double_dqn=False,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the necessary components.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.JaxImplicitQuantileNetwork as an
        example.
      kappa: float, Huber loss cutoff.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
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
    """
    

    self._net_conf = net_conf
    self._env = env
    self._hidden_layer = hidden_layer
    self._neurons=neurons 
    self._noisy = noisy
    self._dueling = dueling
    self._initzer = initzer

    self._tau = tau
    self._alpha = alpha
    self._clip_value_min = clip_value_min
    self._target_opt = target_opt
    self._interact = interact

    self.kappa = kappa
    self._replay_scheme = replay_scheme

    # num_tau_samples = N below equation (3) in the paper.
    self.num_tau_samples = num_tau_samples
    # num_tau_prime_samples = N' below equation (3) in the paper.
    self.num_tau_prime_samples = num_tau_prime_samples
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.double_dqn = double_dqn

    super(JaxImplicitQuantileAgentNew, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network.partial(num_actions=num_actions,
                                net_conf=self._net_conf,
                                env=self._env,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling,
                                initzer=self._initzer,
                                quantile_embedding_dim=quantile_embedding_dim),
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

    self._num_actions=num_actions
    self._replay = self._build_replay_buffer()

  def _create_network(self, name):
    r"""Builds an Implicit Quantile ConvNet.

    Args:
      name: str, this name is passed to the Jax Module.
    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init(self._rng,
                                          name=name,
                                          x=self.state,
                                          num_quantiles=self.num_tau_samples,
                                          rng=self._rng)
    return nn.Model(self.network, initial_params)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self._rng, self.action = select_action(self.online_network,
                                           self.state,
                                           self._rng,
                                           self.num_quantile_samples,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           self._interact,
                                           self._tau,
                                           self.optimizer)
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self._rng, self.action = select_action(self.online_network,
                                           self.state,
                                           self._rng,
                                           self.num_quantile_samples,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           self._interact,
                                           self._tau,
                                           self.optimizer)
    self.action = onp.asarray(self.action)
    return self.action

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
        self._rng, self.optimizer, loss, mean_loss= train(
            self.target_network,
            self.optimizer,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self._target_opt,
            self.num_tau_samples,
            self.num_tau_prime_samples,
            self.num_quantile_samples,
            self.cumulative_gamma,
            self.double_dqn,
            self.kappa,
            self._tau,
            self._alpha,
            self._clip_value_min,
            self._num_actions,
            self._rng)

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)

          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))
          # Weight the loss by the inverse priorities.
          loss = loss_weights * loss
          #mean_loss 
          mean_loss = jnp.mean(loss)

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='ImplicitLoss',
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
