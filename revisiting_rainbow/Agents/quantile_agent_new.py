"""An extension of Rainbow to perform quantile regression.

This loss is computed as in "Distributional Reinforcement Learning with Quantile
Regression" - Dabney et. al, 2017"

Specifically, we implement the following components:

  * n-step updates
  * prioritized replay
  * double_dqn
  * noisy
  * dueling

"""
import copy
import time
import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer #check
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf


@functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None))
def target_distributionDouble(model,target_network, next_states, rewards, terminals,
                        cumulative_gamma):
  """Builds the Quantile target distribution as per Dabney et al. (2017).

  Args:
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    cumulative_gamma: float, cumulative gamma to use (static_argnum).

  Returns:
    The target distribution from the replay.
  """
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  next_state_target_outputs = model(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values) 
  next_qt_argmax = jnp.argmax(q_values)
  
  next_dist = target_network(next_states)
  logits = jnp.squeeze(next_dist.logits)
  next_logits = logits[next_qt_argmax]
  return jax.lax.stop_gradient(rewards + gamma_with_terminal * next_logits)


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def target_distribution(target_network, next_states, rewards, terminals,
                        cumulative_gamma):
  """Builds the Quantile target distribution as per Dabney et al. (2017).

  Args:
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    cumulative_gamma: float, cumulative gamma to use (static_argnum).

  Returns:
    The target distribution from the replay.
  """
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  logits = jnp.squeeze(next_state_target_outputs.logits)
  next_logits = logits[next_qt_argmax]
  return jax.lax.stop_gradient(rewards + gamma_with_terminal * next_logits)


@functools.partial(jax.jit, static_argnums=(0, 9, 10, 11, 12))
def train(network_def, target_params, optimizer, states, actions, next_states, rewards,
          terminals, loss_weights, kappa, num_atoms, cumulative_gamma, double_dqn, rng):
  """Run a training step."""

  online_params = optimizer.target
  def loss_fn(params,rng_input, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, rng=rng_input)

    logits = jax.vmap(q_online)(states).logits
    logits = jnp.squeeze(logits)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    bellman_errors = (target[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
    # Eq. 9 of paper.
    huber_loss = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2 +
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))

    tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) /
               num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
    # Eq. 10 of paper.
    tau_bellman_diff = jnp.abs(
        tau_hat[None, :, None] - (bellman_errors < 0).astype(jnp.float32))
    quantile_huber_loss = tau_bellman_diff * huber_loss
    # Sum over tau dimension, average over target value dimension.
    loss = jnp.sum(jnp.mean(quantile_huber_loss, 2), 1)

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss
  
  rng, rng2, rng3, rng4  = jax.random.split(rng, 4)
  def q_target(state):
    return network_def.apply(target_params, state, rng=rng2)

  def q_target_online(state):
    return network_def.apply(online_params, state, rng=rng4)

  if double_dqn:
    target = target_distributionDouble(q_target_online, q_target, next_states, rewards,  terminals, cumulative_gamma)
  else:
    target = target_distribution(q_target, next_states, rewards, terminals, cumulative_gamma)

  grad_fn = jax.value_and_grad(loss_fn,  has_aux=True)
  (mean_loss, loss), grad =  grad_fn(online_params, rng3, target, loss_weights)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss

@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn):

  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))
  
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  selected_action = jnp.argmax(network_def.apply(params, state, rng=rng3).q_values)  
  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        selected_action)

@gin.configurable
class JaxQuantileAgentNew(dqn_agent.JaxDQNAgent):
  """An implementation of Quantile regression DQN agent."""

  def __init__(self,
               num_actions,
               
               kappa=1.0,
               num_atoms=200,
               noisy = False,
               dueling = False,
               initzer = 'variance_scaling',
               net_conf = None,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               double_dqn=False,
               replay_scheme='prioritized',
               optimizer='adam',
               network=networks.QuantileNetwork,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               seed=None):
    """Initializes the agent and constructs the Graph.

    Args:
      num_actions: Int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expects 3 parameters: num_actions, num_atoms,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.jax.networks.QuantileNetwork as an example.
      kappa: Float, Huber loss cutoff.
      num_atoms: Int, the number of buckets for the value function distribution.
      gamma: Float, exponential decay factor as commonly used in the RL
        literature.
      update_horizon: Int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: Int, number of stored transitions for training to
        start.
      update_period: Int, period between DQN updates.
      target_update_period: Int, ppdate period for the target network.
      epsilon_fn: Function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon), and which returns the epsilon value used for
        exploration during training.
      epsilon_train: Float, final epsilon for training.
      epsilon_eval: Float, epsilon during evaluation.
      epsilon_decay_period: Int, number of steps for epsilon to decay.
      replay_scheme: String, replay memory scheme to be used. Choices are:
        uniform - Standard (DQN) replay buffer (Mnih et al., 2015)
        prioritized - Prioritized replay buffer (Schaul et al., 2015)
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    seed = int(time.time() * 1e6) if seed is None else seed
    self._num_atoms = num_atoms
    self._kappa = kappa
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
    self._rng = jax.random.PRNGKey(seed)

    super(JaxQuantileAgentNew, self).__init__(
        num_actions=num_actions,
        optimizer=optimizer,
        epsilon_fn = dqn_agent.identity_epsilon if self._noisy == True else epsilon_fn,
        network=functools.partial(network, num_atoms=self._num_atoms , net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling,
                                initzer=self._initzer))

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    online_network_params = self.network_def.init(rng, x=self.state, rng=self._rng)
    optimizer_def = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer = optimizer_def.create(online_network_params)
    self.target_network_params = copy.deepcopy(online_network_params)


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

  def begin_episode(self, observation):
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action


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
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer, loss, mean_loss = train(
            self.network_def,
            self.target_network_params,
            self.optimizer,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._kappa,
            self._num_atoms,
            self.cumulative_gamma,
            self._double_dqn,
            self._rng)

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
              tf.compat.v1.Summary.Value(tag='QuantileLoss',
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