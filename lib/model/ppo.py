import tensorflow as tf
import numpy as np

from .enums.policy_type import PolicyType

from .environment import Environment
from .ppo_actor_critic import PPOActorCritic

# T << episode length pg. 5.
T = 10
PARALLEL_ACTORS = 1

class PPO:
    timestep = 0
    gamma = 0.999

    # This value (epsilon) is based on best performing clipping strategy
    # in Table 1, pg. 7.
    probability_ratio_clip_threshold = 0.2

    actor_critic: PPOActorCritic
    env: Environment

    def __init__(self, actor_critic: PPOActorCritic, env: Environment):
        # M <= NT from Algorithm 1 in pg. 5, where M is minibatch size.
        assert(actor_critic.batch_size <= PARALLEL_ACTORS * T)

        self.actor_critic = actor_critic
        self.env = env

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
        assert(len(rewards) == T)
        assert(terminal_timestep > initial_timestep)

        advantages = -tf.squeeze(
            self.actor_critic.critic_model(
                first_states,
                training=True
            ))
        
        timestep_window = terminal_timestep - initial_timestep
        for t in range(initial_timestep, terminal_timestep):
            advantages += tf.multiply(
                    pow(
                        self.gamma,
                        timestep_window + t - terminal_timestep
                    ),
                    tf.squeeze(rewards[t - initial_timestep]),
                )

        advantages += tf.multiply(
                pow(self.gamma, timestep_window),
                tf.squeeze(
                    self.actor_critic.critic_model(
                        last_states,
                        training=True
                    )
                ),
            )

        return advantages

    def calculate_clipped_probability_ratios(self, probability_ratios, advantages):
        assert(len(probability_ratios) == T)
        assert(len(advantages) == T)

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
        assert(len(y_old) == T)
        assert(len(y) == T)

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

        advantages = tf.convert_to_tensor(advantages)
        advantages = tf.squeeze(advantages)
        advantages = tf.expand_dims(advantages, axis=-1)

        minimums = self.calculate_clipped_probability_ratios(probability_ratios, advantages)

        return tf.reduce_mean(minimums)

    def mse(self, y, rewards):
        assert(len(y) == T)
        assert(len(rewards) == T)

        # Expected value from paper pg. 5 ((V^targ)_t) can be defined as (R(s_t,a_t)), as
        # demonstrated by pi-tau from: https://ai.stackexchange.com/a/41896
        y_error = tf.cast(y, dtype=tf.float32) - rewards

        return 0.5 * tf.math.reduce_mean(tf.math.square(y_error))
    
    def __learn(self, y, y_old, states, rewards):
        assert(len(states) == T)
        assert(len(rewards) == T)

        actor_model = self.actor_critic.actor_model
        critic_model = self.actor_critic.critic_model

        actor_optimizer = self.actor_critic.actor_optimizer
        critic_optimizer = self.actor_critic.critic_optimizer

        with tf.GradientTape() as tape:
            actor_loss = self.clipped_surrogate_loss(
                y,
                y_old,
                states[0],
                states[T-1],
                rewards
            )
            actor_loss += tf.add_n(actor_model.losses)
            actor_loss = actor_optimizer.get_scaled_loss(actor_loss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables, unconnected_gradients='zero')
        actor_grad = actor_optimizer.get_unscaled_gradients(actor_grad)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

        with tf.GradientTape() as tape:
            values = tf.convert_to_tensor([
                self.actor_critic.critic_model(states, training=True)
                    for states in states
            ])

            values = tf.squeeze(values)

            critic_loss = self.mse(values, rewards)
            critic_loss += tf.add_n(critic_model.losses)
            critic_loss = critic_optimizer.get_scaled_loss(critic_loss)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_grad = critic_optimizer.get_unscaled_gradients(critic_grad)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

    def run(self):
        for episode in range(1, 501):
            states = [self.__create_empty_states()]

            while True:
                rewards = []
                done_flags = []

                actions_old = []

                for i in range(T):
                    action_batch = self.actor_critic.policy(
                        states[i],
                        PolicyType.OLD.value,
                        training=False,
                    )

                    actions_old.append(action_batch)

                    env_tuples = [
                        self.env.perform_action(action_batch[i], i)
                            for i in range(self.actor_critic.batch_size)
                    ]

                    states.append([env_tuple[0] for env_tuple in env_tuples])
                    rewards.append([env_tuple[1] for env_tuple in env_tuples])
                    done_flags.append([env_tuple[2] for env_tuple in env_tuples])

                actions = [
                    self.actor_critic.policy(
                        states[i],
                        PolicyType.NORMAL.value,
                        training=False,
                    ) for i in range(T)
                ]

                states = tf.convert_to_tensor(states)
                rewards = tf.convert_to_tensor(rewards)
                actions_old = tf.convert_to_tensor(actions_old)
                actions = tf.convert_to_tensor(actions)

                # Nested list entry check solution by Pavel Anossov from:
                # https://stackoverflow.com/a/15057380
                done = any(True in lst for lst in done_flags)

                print(f'Episode: {episode}, Average reward: {np.mean(rewards)}, Episode ended: {done}')

                self.actor_critic.update_old_actor_weights()
                
                self.__learn(
                    actions,
                    actions_old,
                    states[:-1],
                    rewards,
                )

                self.timestep += T

                if done:
                    break

                states = [tf.convert_to_tensor(states[len(states) - 1])]
