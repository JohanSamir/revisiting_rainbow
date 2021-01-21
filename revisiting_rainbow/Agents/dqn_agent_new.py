"""Compact implementation of a DQN agent

Specifically, we implement the following components:

  * prioritized replay
  * huber_loss
  * mse_loss
  * double_dqn
  * noisy
  * dueling
  * Munchausen

Details in: 
"Human-level control through deep reinforcement learning" by Mnih et al. (2015).
"Noisy Networks for Exploration" by Fortunato et al. (2017).
"Deep Reinforcement Learning with Double Q-learning" by Hasselt et al. (2015).
"Dueling Network Architectures for Deep Reinforcement Learning" by Wang et al. (2015).
"Munchausen Reinforcement Learning" by Vieillard et al. (2020).

"""

import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import jax.scipy.special as scp


def mse_loss(targets, predictions):
  return jnp.mean(jnp.power((targets - (predictions)),2))


@functools.partial(jax.jit, static_argnums=(7,8,9,10,11,12))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, cumulative_gamma,target_opt, mse_inf,tau,alpha,clip_value_min):
  """Run the training step."""
  def loss_fn(model, target, mse_inf):
    q_values = jax.vmap(model, in_axes=(0))(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

    if mse_inf:
      loss = jnp.mean(jax.vmap(mse_loss)(target, replay_chosen_q))
    else:
      loss = jnp.mean(jax.vmap(dqn_agent.huber_loss)(target, replay_chosen_q))
    return loss


  grad_fn = jax.value_and_grad(loss_fn)

  if target_opt == 0:
    target = dqn_agent.target_q(target_network, next_states, rewards, terminals, cumulative_gamma) 
  elif target_opt == 1:
    #Double DQN
    target = target_DDQN(optimizer, target_network, next_states, rewards,  terminals, cumulative_gamma)
  elif target_opt == 2:
    #Munchausen
    target = target_m_dqn(optimizer,target_network,states,next_states,actions,rewards,terminals,
                cumulative_gamma,tau,alpha,clip_value_min)
  else:
    print('error')

  loss, grad = grad_fn(optimizer.target, target, mse_inf)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss

def target_DDQN(model, target_network, next_states, rewards, terminals, cumulative_gamma):
  """Compute the target Q-value. Double DQN"""
  next_q_values = jax.vmap(model.target, in_axes=(0))(next_states).q_values
  next_q_values = jnp.squeeze(next_q_values)
  replay_next_qt_max = jnp.argmax(next_q_values, axis=1)
  next_q_state_values = jax.vmap(target_network, in_axes=(0))(next_states).q_values

  q_values = jnp.squeeze(next_q_state_values)
  replay_chosen_q = jax.vmap(lambda t, u: t[u])(q_values, replay_next_qt_max)
 
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_chosen_q *
                               (1. - terminals))

def stable_scaled_log_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  tau_lse = max_x + tau * jnp.log(jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
  return x - tau_lse

def stable_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  return jax.nn.softmax(y/tau, axis=axis)

def target_m_dqn(model, target_network, states, next_states, actions,rewards, terminals, 
                cumulative_gamma,tau,alpha,clip_value_min):
  """Compute the target Q-value. Munchausen DQN"""
  q_state_values = jax.vmap(target_network, in_axes=(0))(states).q_values
  q_state_values = jnp.squeeze(q_state_values)
  #print('q_state_values:',q_state_values.shape)

  next_q_values = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  next_q_values = jnp.squeeze(next_q_values)
  #print('next_q_values:',next_q_values.shape)

  tau_log_pi_next =  stable_scaled_log_softmax(next_q_values, tau, axis=1)
  #print('tau_log_pi_next:',tau_log_pi_next.shape)


  #pi_target = jnp.amax(next_q_values, axis=1, keepdims=True)
  pi_target = stable_softmax(next_q_values,tau, axis=1)
  #print('pi_target:',pi_target.shape)

  replay_next_qt_softmax = jnp.sum((next_q_values-tau_log_pi_next)*pi_target,axis=1)
  #Q_target = Q_target.reshape(Q_target.shape[0],1)
  #print('replay_next_qt_softmax:',replay_next_qt_softmax.shape)

  #replay_log_policy =  jnp.amax(q_state_values,axis=1,keepdims=True)
  replay_log_policy = stable_scaled_log_softmax(q_state_values, tau, axis=1)
  #print('replay_log_policy:',replay_log_policy.shape)
  
  replay_action_one_hot = jax.nn.one_hot(actions, q_state_values.shape[-1])
  tau_log_pi_a = jnp.sum(replay_log_policy * replay_action_one_hot, axis=1)
  #print('tau_log_pi_a:',tau_log_pi_a.shape)

  #a_max=1
  tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=clip_value_min,a_max=1)
  #print('tau_log_pi_a:',tau_log_pi_a.shape)

  munchausen_term = alpha * tau_log_pi_a
  #print('munchausen_term:',munchausen_term.shape)

  
  modified_bellman = (rewards + munchausen_term +cumulative_gamma * replay_next_qt_softmax *
        (1. - jnp.float32(terminals)))
  #print('modified_bellman:',modified_bellman.shape)
  
  return jax.lax.stop_gradient(modified_bellman)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 9, 10, 11, 12))
def select_action(network, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, interact,tau, model):

  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))

  if interact == 'stochastic':

    state = jnp.expand_dims(state, axis=0)
    net_outputs = jax.vmap(model.target, in_axes=(0))(state).q_values
    #print('net_outputs:',net_outputs.shape,net_outputs)
    net_outputs = jnp.squeeze(net_outputs)

    #print('net_outputs:',net_outputs.shape,net_outputs)

    policy_logits =  stable_scaled_log_softmax(net_outputs, tau, axis=0)
    #print('policy_logits:',policy_logits.shape,policy_logits)
    
    key = jax.random.PRNGKey(seed=0)
    stochastic_action = jax.random.categorical(key, policy_logits, axis=0, shape=None)
    #print('stochastic_action:',stochastic_action.shape,stochastic_action)
    #print('interact: stochastic')
    selected_action = stochastic_action

  elif interact == 'greedy':
    selected_action = jnp.argmax(network(state).q_values, axis=1)[0]
  else:
    print('error interact')

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        selected_action)

@gin.configurable
class JaxDQNAgentNew(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,

               tau,
               alpha=1,
               clip_value_min=-10,
               interact = 'greedy',

               net_conf = None,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               prioritized=False,
               noisy = False,
               dueling = False,
               target_opt=0,
               mse_inf=False,
               network=networks.NatureDQNNetwork,
               optimizer='adam',
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

    self._net_conf = net_conf
    self._env = env 
    self._normalize_obs = normalize_obs
    self._hidden_layer = hidden_layer
    self._neurons=neurons 
    self._noisy = noisy
    self._dueling = dueling
    self._target_opt = target_opt
    self._mse_inf = mse_inf
    self._tau = tau
    self._alpha = alpha
    self._clip_value_min = clip_value_min
    self._interact = interact

    super(JaxDQNAgentNew, self).__init__(
        num_actions= num_actions,
        network=network.partial(num_actions=num_actions,
                                net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling),
        optimizer=optimizer,
        epsilon_fn=dqn_agent.identity_epsilon if self._noisy == True else epsilon_fn)

    self._prioritized=prioritized
    self._rng = jax.random.PRNGKey(0)
    state_shape = self.observation_shape + (self.stack_size,)
    self.state = onp.zeros(state_shape)
    self._replay = self._build_replay_buffer_prioritized() if self._prioritized == True else self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._build_networks_and_optimizer()


  def _build_replay_buffer_prioritized(self):
    """Creates the prioritized replay buffer used by the agent."""
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
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        self.optimizer, loss = train(self.target_network,
                                     self.optimizer,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['next_state'],
                                     self.replay_elements['reward'],
                                     self.replay_elements['terminal'],
                                     self.cumulative_gamma,
                                     self._target_opt,
                                     self._mse_inf,
                                     self._tau,
                                     self._alpha,
                                     self._clip_value_min)
        
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition.

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    if self._prioritized==True:
      priority = self._replay.sum_tree.max_recorded_priority
      self._replay.add(last_observation, action, reward, is_terminal, priority)
    else:
      self._replay.add(last_observation, action, reward, is_terminal)

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



