import os
from typing import List

from .ppo_replay_buffers import PPOReplayBuffers

# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

from .enums.policy_type import PolicyType

from .environment import Environment
from .ppo_actor_critic import PPOActorCritic, strategy

# T << episode length pg. 5.
T = 20
EPOCHS = 3
MINIBATCH_SIZE = 256
GAMMA = 0.999

SUCCESSFUL_BUFFER_SIZE = 2 ** 19
UNSUCCESSFUL_BUFFER_SIZE = 2 ** 19

# This value (epsilon) is based on best performing clipping strategy
# in Table 1, pg. 7.
PROBABILITY_RATIO_CLIP_THRESHOLD = 0.2

STARTING_RHO = 0.45
STARTING_PHI = 0.05

# M <= NT from Algorithm 1 in pg. 5, where M is minibatch size.
# N = 1 as there are no parallel actors.
#assert(MINIBATCH_SIZE <= T)

# Assert some chance for actor interaction with environments.
assert(STARTING_RHO + STARTING_PHI <= 0.8)

class PPO:
    timestep = 0

    rho = STARTING_RHO
    phi = STARTING_PHI

    actor_critic: PPOActorCritic
    buffers: PPOReplayBuffers
    environments: List[Environment]

    def __init__(
        self,
        actor_critic: PPOActorCritic,
        environments: List[Environment],
        demonstration_environment: Environment,
        demonstration_actions: tf.Tensor
    ):
        assert(actor_critic.batch_size % MINIBATCH_SIZE == 0)
        assert(T <= len(demonstration_actions))
        assert(len(environments) == actor_critic.batch_size)
        assert(len(environments) == len(set(environments)))
        assert(demonstration_environment not in environments)

        self.actor_critic = actor_critic
        self.environments = environments

        states = []
        actions = []
        rewards = []

        for i in range(T):
            action = demonstration_actions[i]

            state, reward, _ = demonstration_environment.perform_action(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        self.buffers = PPOReplayBuffers(
            state_size=demonstration_environment.state_size,
            action_size=demonstration_environment.action_size,
            successful_buffer_size=SUCCESSFUL_BUFFER_SIZE,
            unsuccessful_buffer_size=UNSUCCESSFUL_BUFFER_SIZE,
            demonstrated_successful_states=states,
            demonstrated_successful_actions=actions,
            demonstrated_successful_rewards=rewards,
        )

    def __create_empty_states(self):
        return tf.convert_to_tensor([
            self.environments[batch_index].create_empty_state()
                for batch_index in range(self.actor_critic.batch_size)
        ])

    @tf.function
    def calculate_advantages_batch(
        self,
        initial_timestep,
        terminal_timestep,
        first_states,
        last_states,
        rewards
    ):
        tf.Assert(tf.equal(rewards.shape[0], T), [rewards])
        tf.Assert(tf.greater(terminal_timestep, initial_timestep), [terminal_timestep, initial_timestep])

        advantages = -tf.squeeze(
            self.actor_critic.critic_model(
                first_states,
                training=True
            ))
        
        timestep_window = terminal_timestep - initial_timestep
        for t in range(initial_timestep, terminal_timestep):
            advantages += tf.multiply(
                    tf.pow(
                        GAMMA,
                        timestep_window + t - terminal_timestep
                    ),
                    tf.squeeze(rewards[t - initial_timestep]),
                )

        advantages += tf.multiply(
                tf.pow(GAMMA, timestep_window),
                tf.squeeze(
                    self.actor_critic.critic_model(
                        last_states,
                        training=True
                    )
                ),
            )

        return advantages

    @tf.function
    def calculate_clipped_probability_ratios(self, probability_ratios, advantages):
        tf.Assert(tf.equal(probability_ratios.shape[0], T), [probability_ratios])
        tf.Assert(tf.equal(advantages.shape[0], T), [advantages])

        clipped = tf.clip_by_value(
            probability_ratios,
            1.0 - PROBABILITY_RATIO_CLIP_THRESHOLD,
            1.0 + PROBABILITY_RATIO_CLIP_THRESHOLD,
        )

        return tf.minimum(
            tf.multiply(probability_ratios, advantages),
            tf.multiply(clipped, advantages),
        )

    @tf.function
    def clipped_surrogate_loss(
        self,
        y,
        y_old,
        first_states,
        last_states,
        rewards
    ):
        tf.Assert(tf.equal(y.shape[0], T), [y])
        tf.Assert(tf.equal(y_old.shape[0], T), [y_old])

        probability_ratios = tf.math.divide_no_nan(y, y_old)

        advantages = [
            self.calculate_advantages_batch(
                t,
                self.timestep + T,
                first_states,
                last_states,
                rewards
            ) for t in range(self.timestep, self.timestep + T)
        ]

        advantages = tf.cast(advantages, dtype=tf.float64)
        advantages = tf.squeeze(advantages)
        advantages = tf.expand_dims(advantages, axis=-1)

        minimums = self.calculate_clipped_probability_ratios(probability_ratios, advantages)

        # The goal of the policy is to maximise the surrogate objective (pg. 3, para. 1),
        # and so our loss should negative this value.
        return -tf.reduce_mean(minimums)

    @tf.function
    def mse(self, y, rewards):
        tf.Assert(tf.equal(y.shape[0], T), [y])
        tf.Assert(tf.equal(rewards.shape[0], T), [rewards])

        # Expected value from paper pg. 5 ((V^targ)_t) can be defined as (R(s_t,a_t)), as
        # demonstrated by pi-tau from: https://ai.stackexchange.com/a/41896
        y_error = tf.cast(y, dtype=tf.float32) - rewards

        return 0.5 * tf.math.reduce_mean(tf.math.square(y_error))
    
    # Modified solution for shuffling along non-first axis by Faris Hijazi from:
    # https://github.com/tensorflow/swift/issues/394#issuecomment-779729550
    def __tf_shuffle_axis(self, value, axis, seed):
        tf.random.set_seed(seed)

        perm = list(range(tf.rank(value)))
        perm[axis], perm[0] = perm[0], perm[axis]
        value = tf.random.shuffle(tf.transpose(value, perm=perm), seed=seed)
        value = tf.transpose(value, perm=perm)
        return value

    def train_actor(
            self,
            states_minibatch,
            actions_old_minibatch,
            y_old_minibatch,
            rewards_minibatch
    ):
        actor_model = self.actor_critic.actor_model
        actor_optimizer = self.actor_critic.actor_optimizer
        
        with tf.GradientTape() as tape:
            y = [
                self.actor_critic.policy(
                    states_minibatch[i],
                    PolicyType.NORMAL.value,
                    batch_size=MINIBATCH_SIZE,
                    actions_reference=actions_old_minibatch[i],
                    use_actions_reference=True,
                )[1] for i in range(T)
            ]

            y = tf.convert_to_tensor(y)

            actor_loss = self.clipped_surrogate_loss(
                y,
                y_old_minibatch,
                states_minibatch[0],
                states_minibatch[T-1],
                rewards_minibatch
            )

            #print(f'Unscaled actor loss: {actor_loss}')
            actor_loss = actor_optimizer.scale_loss(actor_loss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    @tf.function
    def train_critic(self, states_minibatch, rewards_minibatch):
        critic_model = self.actor_critic.critic_model
        critic_optimizer = self.actor_critic.critic_optimizer
    
        with tf.GradientTape() as tape:
            values = tf.map_fn(
                lambda states: self.actor_critic.critic_model(states, training=True),
                states_minibatch
            )

            values = tf.squeeze(values)

            critic_loss = self.mse(values, rewards_minibatch)
            #print(f'Unscaled critic loss: {critic_loss}')
            critic_loss = self.actor_critic.critic_optimizer.scale_loss(critic_loss)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

    def learn(self, actions_old, y_old, states, rewards):
        tf.Assert(tf.equal(states.shape[0], T), [states])
        tf.Assert(tf.equal(rewards.shape[0], T), [rewards])        

        minibatches = self.actor_critic.batch_size // MINIBATCH_SIZE
        for minibatch in range(1, minibatches + 1):
            #print(f'Minibatch {minibatch}/{minibatches}...')

            from_batch_index = (minibatch - 1) * MINIBATCH_SIZE
            to_batch_position = minibatch * MINIBATCH_SIZE

            actions_old_minibatch = actions_old[:, from_batch_index: to_batch_position]
            y_old_minibatch = y_old[:, from_batch_index: to_batch_position]
            states_minibatch = states[:, from_batch_index: to_batch_position]
            rewards_minibatch = rewards[:, from_batch_index: to_batch_position]

            strategy.run(self.train_actor, [
                states_minibatch,
                actions_old_minibatch,
                y_old_minibatch,
                rewards_minibatch,
            ])

            strategy.run(self.train_critic, [
                states_minibatch,
                rewards_minibatch
            ])

    def __anneal_probabilities(self):
        probability_update = STARTING_PHI / self.buffers.max_unsuccessful_buffer_size

        self.rho += probability_update
        self.phi -= probability_update

        # Account for floating point precision.
        if self.rho > 0.99999:
            self.rho = 1.0
        
        # Account for floating point precision.
        #
        # This one is particularly important to be set to zero if expected, as
        # the unsuccessful buffer should be empty if this probability is to reach
        # zero.
        if self.phi < 0.00001:
            self.phi = 0.0

    def run(self):
        states = [self.__create_empty_states()]

        # TODO: Record total epsiodic rewards for individual environments.
        '''
        episodes = [
            1 for _ in range(self.actor_critic.batch_size)
        ]    
        episodic_rewards = [
            
        ]
        '''

        while True:
            rewards = []
            done_flags = []

            actions_old = []
            action_probabilities_old = []

            # Demonstrations should give a probability of 1 as they should come from a
            # separate policy which is definitionally constructed as such:
            #
            # Single demonstration PPO paper (pg. 3):
            # 
            # Policy = one of the below:
            #
            # πDR , if sampled from DR
            # πDV , if sampled from DV
            # πθ , if sampled from Env
            rand = np.random.rand()
            policy_type = PolicyType.SUCCESSFUL_DEMONSTRATIONS \
                if rand < self.rho \
                else PolicyType.UNSUCCESSFUL_DEMONSTRATIONS \
                if rand < self.rho + self.phi and not self.buffers.is_unsuccessful_buffer_empty() \
                else PolicyType.OLD

            trajectories = self.buffers.sample_successful_trajectories(
                    batch_size=self.actor_critic.batch_size
                ) \
                if policy_type == PolicyType.SUCCESSFUL_DEMONSTRATIONS \
                else self.buffers.sample_unsuccessful_trajectories(
                    batch_size=self.actor_critic.batch_size
                ) \
                if policy_type == PolicyType.UNSUCCESSFUL_DEMONSTRATIONS \
                else None

            for i in range(T):
                if trajectories == None:
                    action_batch, probabilities_batch = self.actor_critic.policy(
                        states[i],
                        PolicyType.OLD.value,
                        batch_size=self.actor_critic.batch_size,
                        actions_reference=tf.fill([self.actor_critic.batch_size, self.actor_critic.action_size,], -1),
                        use_actions_reference=False,
                    )
                else:
                    action_batch, probabilities_batch = trajectories[1][:,i], tf.ones(
                        [self.actor_critic.batch_size, 1],
                        dtype=tf.float64,
                    )
                    
                actions_old.append(action_batch)
                action_probabilities_old.append(probabilities_batch)

                if trajectories == None:
                    env_tuples = [
                        self.environments[batch_index].perform_action(action_batch[batch_index])
                            for batch_index in range(self.actor_critic.batch_size)
                    ]

                    states.append(tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples]))
                    rewards.append([env_tuple[1] for env_tuple in env_tuples])
                    done_flags.append([env_tuple[2] for env_tuple in env_tuples])
                else:
                    states.append(trajectories[0][:, i])
                    rewards.append(trajectories[2][:, i])

            states = tf.convert_to_tensor(states)
            actions_old = tf.convert_to_tensor(actions_old)
            rewards = tf.convert_to_tensor(rewards)

            if policy_type == PolicyType.OLD:
                for i in range(self.actor_critic.batch_size):
                    trajectory_reward = np.sum(rewards[:, i])

                    if trajectory_reward > 0.0:
                        self.buffers.record_successful_transitions(
                            states[:-1, i],
                            actions_old[:, i],
                            rewards[:, i],
                        )

                        if self.phi > 0.0:
                            self.__anneal_probabilities()
                    else:
                        # TODO: Ensure nothing added to unsuccessful transitions buffer when phi = 0
                        # as unsuccessful buffer unused at this point.
                        self.buffers.record_unsuccessful_transitions(
                            states[:-1, i],
                            actions_old[:, i],
                            rewards[:, i],
                            value_model=self.actor_critic.critic_model,
                        )

            action_probabilities_old = tf.convert_to_tensor(action_probabilities_old)

            mean_batch_rollout_reward = np.mean(rewards)

            for _ in range(1, EPOCHS + 1):
                seed = np.random.randint(0, 9999999)

                actions_old_shuffled = self.__tf_shuffle_axis(actions_old, axis=1, seed=seed)
                action_probabilities_old_shuffed = self.__tf_shuffle_axis(action_probabilities_old, axis=1, seed=seed)
                states_shuffled = self.__tf_shuffle_axis(states[:-1], axis=1, seed=seed)
                rewards_shuffled = self.__tf_shuffle_axis(rewards, axis=1, seed=seed)

                self.learn(
                    actions_old=actions_old_shuffled,
                    y_old=action_probabilities_old_shuffed,
                    states=states_shuffled,
                    rewards=rewards_shuffled,
                )

            self.actor_critic.update_old_actor_weights()

            if policy_type == PolicyType.OLD:
                self.timestep += T

            if policy_type == PolicyType.OLD:
                print(f'Timestep: {self.timestep}, Mean rollout reward: {mean_batch_rollout_reward}, Policy Type: {policy_type.name}')
            else:
                print(f'Timestep: {self.timestep}, Mean trajectory rollout playback reward: {mean_batch_rollout_reward}, Policy Type: {policy_type.name}')

            states = [states[-1]]
