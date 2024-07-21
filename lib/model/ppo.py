import os
import time

# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

from .enums.policy_type import PolicyType

from .environment import Environment
from .ppo_actor_critic import PPOActorCritic, strategy

# T << episode length pg. 5.
T = 16
EPOCHS = 3
MINIBATCH_SIZE = 16

# M <= NT from Algorithm 1 in pg. 5, where M is minibatch size.
# N = 1 as there are no parallel actors.
assert(MINIBATCH_SIZE <= T)

class PPO:
    timestep = 0
    gamma = 0.999

    # This value (epsilon) is based on best performing clipping strategy
    # in Table 1, pg. 7.
    probability_ratio_clip_threshold = 0.2

    actor_critic: PPOActorCritic
    env: Environment
    demonstration_actions: tf.Tensor

    def __init__(
        self,
        actor_critic: PPOActorCritic,
        env: Environment,
        demonstration_actions: tf.Tensor
    ):
        assert(actor_critic.batch_size % MINIBATCH_SIZE == 0)

        self.actor_critic = actor_critic
        self.env = env
        self.demonstration_actions = demonstration_actions

    def __create_empty_states(self):
        states = [
            self.env.create_empty_state(index=i)
                for i in range(self.actor_critic.batch_size)
        ]

        return tf.convert_to_tensor(states)

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
                        self.gamma,
                        timestep_window + t - terminal_timestep
                    ),
                    tf.squeeze(rewards[t - initial_timestep]),
                )

        advantages += tf.multiply(
                tf.pow(self.gamma, timestep_window),
                tf.squeeze(
                    self.actor_critic.critic_model(
                        last_states,
                        training=True
                    )
                ),
            )

        return advantages

    def calculate_clipped_probability_ratios(self, probability_ratios, advantages):
        tf.Assert(tf.equal(probability_ratios.shape[0], T), [probability_ratios])
        tf.Assert(tf.equal(advantages.shape[0], T), [advantages])

        clipped = tf.clip_by_value(
            probability_ratios,
            1.0 - self.probability_ratio_clip_threshold,
            1.0 + self.probability_ratio_clip_threshold
        )

        return tf.minimum(
            tf.multiply(probability_ratios, advantages),
            tf.multiply(clipped, advantages),
        )

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

        probability_ratios = tf.math.divide_no_nan(
            tf.cast(y, dtype=tf.float32),
            tf.cast(y_old, dtype=tf.float32),
        )

        advantages = [
            self.calculate_advantages_batch(
                t,
                self.timestep + T,
                first_states,
                last_states,
                rewards
            ) for t in range(self.timestep, self.timestep + T)
        ]

        advantages = tf.squeeze(advantages)
        advantages = tf.expand_dims(advantages, axis=-1)

        minimums = self.calculate_clipped_probability_ratios(probability_ratios, advantages)

        # The goal of the policy is to maximise the surrogate objective (pg. 3, para. 1),
        # and so our loss should negative this value.
        return -tf.reduce_mean(minimums)

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
                    training=True,
                    actions_reference=actions_old_minibatch[i]
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
            actor_loss = actor_optimizer.get_scaled_loss(actor_loss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_grad = actor_optimizer.get_unscaled_gradients(actor_grad)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))
        
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
            critic_loss = self.actor_critic.critic_optimizer.get_scaled_loss(critic_loss)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_grad = critic_optimizer.get_unscaled_gradients(critic_grad)
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

    def run(self):
        epsilon = 1.0

        for episode in range(1, 501):
            states = [self.__create_empty_states()]
            
            episodic_reward = 0.0

            #epsiode_started = time.time()

            while True:
                rewards = []
                done_flags = []

                actions_old = []
                action_probabilities_old = []

                demonstrating = np.random.rand() < epsilon

                for i in range(T):
                    actions_reference = tf.random.shuffle(self.demonstration_actions)[:self.actor_critic.batch_size] \
                        if demonstrating \
                        else tf.constant([])

                    action_batch, probabilities_batch = self.actor_critic.policy(
                        states[i],
                        PolicyType.OLD.value,
                        batch_size=self.actor_critic.batch_size,
                        training=False,
                        actions_reference=actions_reference,
                    )

                    actions_old.append(action_batch)
                    action_probabilities_old.append(probabilities_batch)

                    env_tuples = [
                        self.env.perform_action(action_batch[i], i, demonstrating=demonstrating)
                            for i in range(self.actor_critic.batch_size)
                    ]

                    states.append(tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples]))
                    rewards.append([env_tuple[1] for env_tuple in env_tuples])
                    done_flags.append([env_tuple[2] for env_tuple in env_tuples])

                action_probabilities_old = tf.convert_to_tensor(action_probabilities_old)
                states = tf.convert_to_tensor(states)
                rewards = tf.convert_to_tensor(rewards)

                # Nested list entry check solution by Pavel Anossov from:
                # https://stackoverflow.com/a/15057380
                done = any(True in lst for lst in done_flags)

                episodic_reward += np.sum(rewards)
                print(f'Episode {episode}, Total episodic reward: {episodic_reward}, Demonstrating: {demonstrating}')
 
                for epoch in range(1, EPOCHS + 1):
                    #print(f'Epoch {epoch}/{EPOCHS}...')

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

                self.timestep += T

                if done:
                    break

                states = [tf.convert_to_tensor(states[len(states) - 1])]

            #episode_ended = time.time()
            
            #print(f'Episode length (seconds): {episode_ended - epsiode_started}\n')

            # Base epsilon dynamically on accuracy of non-demonstration actions?
            epsilon = max(epsilon*0.975, 0.2)
