import os
import time
import numpy as np
from .policy_type import PolicyType
from .environment import Environment
from .ppo_actor_critic import PPOActorCritic
from typing import List
from . import state_factory
from .ppo_episodic_reporter import PPOEpisodicReporter
from .ppo_reporter import PPOReporter
from .ppo_running_statistics import PPORunningStatistics
from .ppo_replay_buffer import PPOReplayBuffer
from ..hyperparameters import STATE_SIZE, ACTION_SIZE, BATCH_SIZE, MINIBATCH_SIZE, PPO_SUCCESSFUL_BATCH_SIZE, \
    ENVIRONMENT_BATCH_SIZE, T, EPOCHS, GAMMA, PPO_PROBABILITY_RATIO_CLIP_THRESHOLD, PPO_SUCCESSFUL_BUFFER_SIZE
    

# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

class PPO:
    timestep = 0

    actor_critic: PPOActorCritic
    buffer: PPOReplayBuffer
    environments: List[Environment]

    def __init__(
        self,
        actor_critic: PPOActorCritic,
        environments: List[Environment],
    ):
        assert(len(environments) == ENVIRONMENT_BATCH_SIZE)
        assert(len(environments) == len(set(environments)))

        self.actor_critic = actor_critic
        self.environments = environments

    def __init_buffer(
        self,
        demonstration_environment: Environment,
        demonstration_actions: tf.Tensor,
    ):
        assert(demonstration_environment not in self.environments)
        
        assert(len(demonstration_actions) > 0)
        assert(len(demonstration_actions) % T == 0)
        
        assert(PPO_SUCCESSFUL_BUFFER_SIZE > len(demonstration_actions) % T)

        states = []
        actions = []
        rewards = []

        for i in range(T):
            action = demonstration_actions[i]

            state, reward = demonstration_environment.perform_demonstration_action(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        states = np.reshape(states, [-1, T, STATE_SIZE,])
        actions = np.reshape(actions, [-1, T, ACTION_SIZE,])
        rewards = np.reshape(rewards, [-1, T,])

        self.buffer = PPOReplayBuffer(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            successful_buffer_size=PPO_SUCCESSFUL_BUFFER_SIZE,
            demonstrated_successful_states=states,
            demonstrated_successful_actions=actions,
            demonstrated_successful_rewards=rewards,
        )

    def __create_empty_states(self):
        return tf.convert_to_tensor([
            state_factory.create_empty_state(state_size=STATE_SIZE)
                for _ in range(ENVIRONMENT_BATCH_SIZE)
        ])

    def calculate_advantages_batch(
        self,
        initial_timestep: int,
        terminal_timestep: int,
        first_states: tf.Tensor,
        last_states: tf.Tensor,
        rewards: tf.Tensor,
    ):
        tf.Assert(tf.equal(rewards.shape[0], T), [rewards])
        tf.Assert(tf.greater(terminal_timestep, initial_timestep), [terminal_timestep, initial_timestep])
        
        gamma = tf.constant(GAMMA, dtype=tf.float64)

        advantages = -tf.squeeze(
            self.actor_critic.critic_model(
                first_states,
                training=False,
            ))
        
        timestep_window = terminal_timestep - initial_timestep
        for t in range(initial_timestep, terminal_timestep):
            advantages += tf.multiply(
                    tf.pow(
                        gamma,
                        timestep_window + t - terminal_timestep
                    ),
                    tf.squeeze(rewards[t - initial_timestep]),
                )

        advantages += tf.multiply(
                tf.pow(gamma, timestep_window),
                tf.squeeze(
                    self.actor_critic.critic_model(
                        last_states,
                        training=False,
                    )
                ),
            )

        return advantages

    def calculate_clipped_probability_ratios(self, probability_ratios, advantages):
        tf.Assert(tf.equal(probability_ratios.shape[0], T), [probability_ratios])
        tf.Assert(tf.equal(advantages.shape[0], T), [advantages])

        clipped = tf.clip_by_value(
            probability_ratios,
            1.0 - PPO_PROBABILITY_RATIO_CLIP_THRESHOLD,
            1.0 + PPO_PROBABILITY_RATIO_CLIP_THRESHOLD,
        )

        return tf.minimum(
            tf.multiply(probability_ratios, advantages),
            tf.multiply(clipped, advantages),
        )

    def clipped_surrogate_loss(
        self,
        y: tf.Tensor,
        y_old: tf.Tensor,
        first_states: tf.Tensor,
        last_states: tf.Tensor,
        rewards: tf.Tensor,
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
        
        advantages = tf.squeeze(advantages)
        advantages = tf.expand_dims(advantages, axis=-1)

        minimums = self.calculate_clipped_probability_ratios(probability_ratios, advantages)
        
        # The goal of the policy is to maximise the surrogate objective (pg. 3, para. 1),
        # and so our loss should negate this value.
        return -tf.reduce_mean(minimums)

    def mse(self, y: tf.Tensor, rewards: tf.Tensor):
        tf.Assert(tf.equal(y.shape[0], T), [y])
        tf.Assert(tf.equal(rewards.shape[0], T), [rewards])

        # Expected value from paper pg. 5 ((V^targ)_t) can be defined as (R(s_t,a_t)), as
        # demonstrated by pi-tau from: https://ai.stackexchange.com/a/41896
        y_error = tf.cast(y, dtype=tf.float64) - rewards

        return 0.5 * tf.math.reduce_mean(tf.math.square(y_error))
    
    # Modified solution for shuffling along non-first axis by Faris Hijazi from:
    # https://github.com/tensorflow/swift/issues/394#issuecomment-779729550
    def __tf_shuffle_axis(self, value: tf.Tensor, axis: int, seed: int):
        tf.random.set_seed(seed)

        perm = list(range(tf.rank(value)))
        perm[axis], perm[0] = perm[0], perm[axis]
        value = tf.random.shuffle(tf.transpose(value, perm=perm), seed=seed)
        value = tf.transpose(value, perm=perm)
        return value

    def train_actor(
        self,
        states_minibatch: tf.Tensor,
        actions_old_minibatch: tf.Tensor,
        y_old_minibatch: tf.Tensor,
        rewards_minibatch: tf.Tensor,
    ):
        '''
        Returns actor loss.
        '''
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
        
        return actor_loss

    def train_critic(
        self,
        states_minibatch: tf.Tensor,
        rewards_minibatch: tf.Tensor,
    ):
        '''
        Returns critic loss.
        '''
        
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

        return critic_loss

    def __learn(
        self,
        actions_old: tf.Tensor,
        y_old: tf.Tensor,
        states: tf.Tensor,
        rewards: tf.Tensor,
    ):
        '''
        Returns `(actor losses, critic losses)` across minibatch trainings.
        '''
        assert(states.shape[0] == T)
        assert(rewards.shape[0] == T)    

        actor_losses, critic_losses = [], []
        
        minibatches = BATCH_SIZE // MINIBATCH_SIZE
        
        for minibatch in range(1, minibatches + 1):
            #print(f'Minibatch {minibatch}/{minibatches}...')

            from_batch_index = (minibatch - 1) * MINIBATCH_SIZE
            to_batch_position = minibatch * MINIBATCH_SIZE

            actions_old_minibatch = actions_old[:, from_batch_index: to_batch_position]
            y_old_minibatch = y_old[:, from_batch_index: to_batch_position]
            states_minibatch = states[:, from_batch_index: to_batch_position]
            rewards_minibatch = rewards[:, from_batch_index: to_batch_position]

            critic_loss = self.train_critic(
                states_minibatch,
                rewards_minibatch
            )

            actor_loss = self.train_actor(
                states_minibatch,
                actions_old_minibatch,
                y_old_minibatch,
                rewards_minibatch,
            )
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
        return actor_losses, critic_losses

    def __explore(
        self,
        states: List[tf.Tensor],
        reporter: PPOReporter,
        episodic_rewards_reporter: PPOEpisodicReporter,
    ):
        actions = []
        probabilities = []
        rewards = []

        trajectories = self.buffer.sample_successful_trajectories(
            batch_size=PPO_SUCCESSFUL_BATCH_SIZE,
        )
        
        # Demonstration initial states are appended first as samples
        # trajectories do not contain the states after the final action
        # is taken, i.e., no state for rollout index T.
        states = [
            tf.concat(
                values=[
                    states[0],
                    trajectories[0][:, 0],
                ],
                axis=0,
            )
        ]

        for i in range(T):
            # Interact with environments.
            # ===================================
            actions_env, probabilities_env = self.actor_critic.policy(
                states[i][:ENVIRONMENT_BATCH_SIZE],
                PolicyType.OLD.value,
                batch_size=ENVIRONMENT_BATCH_SIZE,
                actions_reference=tf.fill([ENVIRONMENT_BATCH_SIZE, ACTION_SIZE,], -1),
                use_actions_reference=False,
            )

            env_tuples = []

            for batch_index in range(ENVIRONMENT_BATCH_SIZE):
                environment = self.environments[batch_index]

                # Retrieve epsiode before its potential update.
                #
                # Otherwise, if the episode is updated, the reward returned
                # will match the previous episode.
                episode = environment.episode
                
                state, reward = environment.perform_action(
                    actions_env[batch_index],
                    timestep=self.timestep + 1,
                    reporter=reporter,
                )
                
                episodic_rewards_reporter.record_episodic_statistics(
                    episode=episode,
                    reward=reward,
                    batch_index=batch_index,
                )
                    
                env_tuples.append((state, reward,))

            states_env = [env_tuple[0] for env_tuple in env_tuples]
            rewards_env = [env_tuple[1] for env_tuple in env_tuples]
            # ===================================


            # Get trajectory from demonstration samples.
            # ===================================
            actions_demo = trajectories[1][:,i]

            # Demonstrations actions are set with a probability of 100%,
            # as outlined in the PPO using a Single Demonstration paper
            # (p.3).
            probabilities_demo =  tf.ones(
                [PPO_SUCCESSFUL_BATCH_SIZE, 1,],
                dtype=tf.float64,
            )

            # Demonstration states are offset by one from first initial
            # states. They do not contain states after the last rollout
            # action at index T.
            states_demo = trajectories[0][:, i + 1] if i < T - 1 else []

            rewards_demo = trajectories[2][:, i]
            # ===================================


            actions.append(
                tf.concat(
                    values=[
                        actions_env,
                        actions_demo,
                    ],
                    axis=0,
                ))

            probabilities.append(
                tf.concat(
                    values=[
                        probabilities_env,
                        probabilities_demo,
                    ],
                    axis=0,
                ))

            if len(states_demo) > 0:
                states.append(
                    tf.concat(
                        values=[
                            states_env,
                            states_demo,
                        ],
                        axis=0,
                    ))
            else:
                states.append(states_env)

            rewards.append(
                tf.concat(
                    values=[
                        rewards_env,
                        rewards_demo,
                    ],
                    axis=0,
                ))

        # Convert last states separately as last index of states
        # does not contain demonstration states. A tensor requires
        # all elements to conform to the same shape.
        next_states = tf.convert_to_tensor(states[-1], dtype=tf.float64)
        states = tf.convert_to_tensor(states[:-1], dtype=tf.float64)

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        probabilities = tf.convert_to_tensor(probabilities, dtype=tf.float64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)

        # Record successful on-policy transitions from environment.
        for i in range(PPO_SUCCESSFUL_BATCH_SIZE):
            trajectory_reward = np.sum(rewards[:, i])

            if trajectory_reward > 0.0:
                self.buffer.record_successful_transitions(
                    states[:, i],
                    actions[:, i],
                    rewards[:, i],
                )

        return states, next_states, actions, probabilities, rewards
    
    def __learn_sgd(
        self,
        states: List[tf.Tensor],
        actions_old: tf.Tensor,
        action_probabilities_old: tf.Tensor,
        rewards: tf.Tensor,
    ):
        '''
        Returns `(mean actor loss, mean critic loss)` across every optimisation step.
        '''
        actor_losses, critic_losses = [], []
        
        for _ in range(1, EPOCHS + 1):
            seed = np.random.randint(0, 9999999)

            actions_old_shuffled = self.__tf_shuffle_axis(actions_old, axis=1, seed=seed)
            action_probabilities_old_shuffed = self.__tf_shuffle_axis(action_probabilities_old, axis=1, seed=seed)
            states_shuffled = self.__tf_shuffle_axis(states, axis=1, seed=seed)
            rewards_shuffled = self.__tf_shuffle_axis(rewards, axis=1, seed=seed)

            epoch_actor_losses, epoch_critic_losses = self.__learn(
                actions_old=actions_old_shuffled,
                y_old=action_probabilities_old_shuffed,
                states=states_shuffled,
                rewards=rewards_shuffled,
            )
            
            actor_losses.extend(epoch_actor_losses)
            critic_losses.extend(epoch_critic_losses)

        self.actor_critic.update_old_actor_weights()
        
        return np.mean(actor_losses), np.mean(critic_losses)
            
    def __run_training_step(
        self,
        states: List[tf.Tensor],
        reporter: PPOReporter,
        episodic_rewards_reporter: PPOEpisodicReporter,
    ):        
        total_seconds = time.time()
        
        # Explore.
        # ===================================
        exploration_seconds = time.time()
        
        states, next_states, actions_old, action_probabilities_old, rewards = self.__explore(
            states=states,
            reporter=reporter,
            episodic_rewards_reporter=episodic_rewards_reporter,
        )
        
        exploration_seconds = time.time() - exploration_seconds
        # ===================================
        
        
        # Learn.
        # ===================================
        learning_seconds = time.time()

        mean_actor_loss, mean_critic_loss = self.__learn_sgd(
            states=states,
            actions_old=actions_old,
            action_probabilities_old=action_probabilities_old,
            rewards=rewards,
        )
        
        learning_seconds = time.time() - learning_seconds
        # ===================================
        
        
        self.timestep += T
        
        total_seconds = time.time() - total_seconds
        
        mean_batch_rollout_reward = np.mean(rewards)

        running_stats = PPORunningStatistics(
            timestep=self.timestep,
            mean_batch_reward=mean_batch_rollout_reward,
            mean_actor_loss=mean_actor_loss,
            mean_critic_loss=mean_critic_loss,
            exploration_seconds=exploration_seconds,
            learning_seconds=learning_seconds,
            training_step_seconds=total_seconds,
        )

        reporter.record_running_statistics(running_stats)

        return [next_states]
        

    def run(
        self,
        demonstration_environment: Environment,
        demonstration_actions: tf.Tensor,
    ):
        self.__init_buffer(
            demonstration_environment=demonstration_environment,
            demonstration_actions=demonstration_actions,
        )
        
        reporter = PPOReporter()
        
        episodic_rewards_reporter = PPOEpisodicReporter(
            batch_size=ENVIRONMENT_BATCH_SIZE,
            reporter=reporter,
        )

        reporter.start()
        
        starting_states = [self.__create_empty_states()]

        while True:
            starting_states = self.__run_training_step(
                states=starting_states,
                reporter=reporter,
                episodic_rewards_reporter=episodic_rewards_reporter,
            )
