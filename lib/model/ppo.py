import tensorflow as tf
import numpy as np

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
        self.actor_critic = actor_critic
        self.env = env

    def __create_empty_states(self):
        states = [
            self.env.create_empty_state(index=i)
                for i in range(self.actor_critic.batch_size)
        ]

        return tf.convert_to_tensor(states)

    def calculate_advantage(
        self,
        initial_timestep,
        terminal_timestep,
        first_state,
        last_state,
        rewards
    ):
        assert(len(rewards) == T)
        assert(terminal_timestep > initial_timestep)

        advantage = -self.actor_critic.critic_model(first_state)

        timestep_window = terminal_timestep - initial_timestep
        for t in range(initial_timestep, terminal_timestep):
            advantage += pow(
                    self.gamma,
                    timestep_window + t - terminal_timestep
                ) * rewards[t - initial_timestep]

        advantage += pow(self.gamma, timestep_window) * self.actor_critic.critic_model(last_state)

        return tf.convert_to_tensor(advantage)

    def calculate_clipped_probability_ratios(self, probability_ratios, advantages):
        assert(len(probability_ratios) == T)
        assert(len(advantages) == T)

        clipped = tf.clip_by_value(
            probability_ratios,
            1.0 - self.probability_ratio_clip_threshold,
            1.0 + self.probability_ratio_clip_threshold
        )

        return tf.minimum(probability_ratios * advantages, clipped * advantages)

    def clipped_surrogate_loss(
        self,
        y,
        y_old,
        first_state,
        last_state,
        rewards
    ):
        assert(len(y_old) == T)
        assert(len(y) == T)

        probability_ratios = y / y_old
        advantages = [
            self.calculate_advantage(
                self.timestep + t,
                self.timestep + T,
                first_state,
                last_state,
                rewards
            ) for t in range(self.timestep, self.timestep + T)
        ]
        advantages = tf.convert_to_tensor(advantages)

        minimums = self.calculate_clipped_probability_ratios(probability_ratios, advantages)

        return tf.reduce_mean(minimums)

    def mse(self, y, rewards):
        assert(len(y) == T)
        assert(len(rewards) == T)

        # Expected value from paper pg. 5 ((V^targ)_t) can be defined as (R(s_t,a_t)), as
        # demonstrated by pi-tau from: https://ai.stackexchange.com/a/41896
        y_error = y - rewards

        return 0.5 * tf.math.reduce_mean(tf.math.square(y_error))
    
    def __learn(self, y, y_old, first_state, last_state, rewards):
        actor_model = self.actor_critic.actor_model
        critic_model = self.actor_critic.critic_model

        actor_optimizer = self.actor_critic.actor_optimizer
        critic_optimizer = self.actor_critic.critic_optimizer

        with tf.GradientTape() as tape:
            actor_loss = self.clipped_surrogate_loss(
                y,
                y_old,
                first_state,
                last_state,
                rewards
            )
            actor_loss += tf.add_n(actor_model.losses)
            actor_loss = actor_optimizer.get_scaled_loss(actor_loss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables, unconnected_gradients='zero')
        actor_grad = actor_optimizer.get_unscaled_gradients(actor_grad)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

        with tf.GradientTape() as tape:
            critic_loss = self.mse(y, rewards)
            critic_loss += tf.add_n(critic_model.losses)
            critic_loss = critic_optimizer.get_scaled_loss(critic_loss)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_grad = critic_optimizer.get_unscaled_gradients(critic_grad)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

    def interact(self):
        states = self.__create_empty_states()

        for ep in range(500):
            while True:
                actions = [
                    self.actor_critic.actor_model(states)
                        for _ in range(T)
                ]

                actions_old = [
                    self.actor_critic.actor_model_old(states)
                        for _ in range(T)
                ]

                actions = tf.convert_to_tensor(actions)
                actions_old = tf.convert_to_tensor(actions_old)

                env_tuples = [
                    self.env.perform_action(actions_old[i], i)
                        for i in range(T)
                ]

                states = [env_tuple[0] for env_tuple in env_tuples]
                rewards = [env_tuple[1] for env_tuple in env_tuples]
                done_flags = [env_tuple[2] for env_tuple in env_tuples]

                done = True in done_flags

                print(f'Average reward: ${np.mean(rewards)}, Episode ended: ${done}')
                
                self.__learn(
                    actions,
                    actions_old,
                    states[0],
                    states[T-1],
                    rewards,
                )

                self.timestep += 1

                if done:
                    break