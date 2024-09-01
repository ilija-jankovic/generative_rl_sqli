import os
import time
from typing import List

from .ppo_episodic_rewards_reporter import PPOEpisodicRewardsReporter

from ..hyperparameters import STATE_SIZE, ACTION_SIZE, BATCH_SIZE, T, EPOCHS, GAMMA, \
    PPO_PROBABILITY_RATIO_CLIP_THRESHOLD, PPO_SUCCESSFUL_BUFFER_SIZE, \
    PPO_SUCCESSFUL_POLICY_PROBABILITY, MINIBATCH_SIZE

from .ppo_reporter import PPOReporter
from .ppo_running_statistics import PPORunningStatistics
from .ppo_replay_buffer import PPOReplayBuffer

# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

from .enums.policy_type import PolicyType

from .environment import Environment
from .ppo_actor_critic import PPOActorCritic

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
        assert(len(environments) == BATCH_SIZE)
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
            self.environments[batch_index].create_empty_state()
                for batch_index in range(BATCH_SIZE)
        ])

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
                training=False,
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

    def train_critic(self, states_minibatch, rewards_minibatch):
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

    def __learn(self, actions_old, y_old, states, rewards):
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
            
    def __sample_policy_type(self):
        rand = np.random.rand()
        
        return PolicyType.SUCCESSFUL_DEMONSTRATIONS \
            if rand < PPO_SUCCESSFUL_POLICY_PROBABILITY \
            else PolicyType.OLD
            
    def __explore(
        self,
        policy_type: PolicyType,
        states: List[tf.Tensor],
        reporter: PPOReporter,
        episodic_rewards_reporter: PPOEpisodicRewardsReporter,
    ):
        actions_old = []
        action_probabilities_old = []
        rewards = []

        trajectories = self.buffer.sample_successful_trajectories(
                batch_size=BATCH_SIZE
            ) \
            if policy_type == PolicyType.SUCCESSFUL_DEMONSTRATIONS \
            else None

        for i in range(T):
            if policy_type == PolicyType.OLD:
                action_batch, probabilities_batch = self.actor_critic.policy(
                    states[i],
                    PolicyType.OLD.value,
                    batch_size=BATCH_SIZE,
                    actions_reference=tf.fill([BATCH_SIZE, ACTION_SIZE,], -1),
                    use_actions_reference=False,
                )
            elif policy_type == PolicyType.SUCCESSFUL_DEMONSTRATIONS:
                action_batch, probabilities_batch = self.actor_critic.policy(
                    states[i],
                    PolicyType.OLD.value,
                    batch_size=BATCH_SIZE,
                    actions_reference=trajectories[1][:,i],
                    use_actions_reference=True,
                )
            else:
                raise Exception(f'Policy type {policy_type} is invalid for exploration.')
                
            actions_old.append(action_batch)
            action_probabilities_old.append(probabilities_batch)

            if trajectories == None:
                env_tuples = []
                
                for batch_index in range(BATCH_SIZE):
                    environment = self.environments[batch_index]

                    # Retrieve epsiode before its potential update.
                    #
                    # Otherwise, if the episode is updated, the reward returned
                    # will match the previous episode.
                    episode = environment.episode
                    
                    state, reward = environment.perform_action(
                        action_batch[batch_index],
                        timestep=self.timestep + batch_index + 1,
                        reporter=reporter,
                    )
                    
                    episodic_rewards_reporter.record_reward(
                        reward=reward,
                        batch_index=batch_index,
                        episode=episode,
                    )
                        
                    env_tuples.append((state, reward,))


                states.append(tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples]))
                rewards.append([env_tuple[1] for env_tuple in env_tuples])
            else:
                states.append(trajectories[0][:, i])
                rewards.append(trajectories[2][:, i])

        states = tf.convert_to_tensor(states)
        actions_old = tf.convert_to_tensor(actions_old)
        action_probabilities_old = tf.convert_to_tensor(action_probabilities_old)
        rewards = tf.convert_to_tensor(rewards)

        if policy_type == PolicyType.OLD:
            for i in range(BATCH_SIZE):
                trajectory_reward = np.sum(rewards[:, i])

                if trajectory_reward > 0.0:
                    self.buffer.record_successful_transitions(
                        states[:-1, i],
                        actions_old[:, i],
                        rewards[:, i],
                    )
                    
        return states, actions_old, action_probabilities_old, rewards
    
    def __learn_sgd(
        self,
        states: tf.Tensor,
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
            states_shuffled = self.__tf_shuffle_axis(states[:-1], axis=1, seed=seed)
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
        episodic_rewards_reporter: PPOEpisodicRewardsReporter,
    ):        
        total_seconds = time.time()
        
        # Explore
        # ===================================
        exploration_seconds = time.time()
        
        policy_type = self.__sample_policy_type()
        
        states, actions_old, action_probabilities_old, rewards = self.__explore(
            policy_type=policy_type,
            states=states,
            reporter=reporter,
            episodic_rewards_reporter=episodic_rewards_reporter,
        )
        
        exploration_seconds = time.time() - exploration_seconds
        # ===================================
        
        
        # Learn
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
        
        
        # Update timestep (replayed demonstrations do not count as interaction
        # with the environment).
        if policy_type == PolicyType.OLD:
            self.timestep += T
        
        total_seconds = time.time() - total_seconds
        
        # Report running statistics and print reward if not from replayed
        # demonstrations.
        if policy_type == PolicyType.OLD:
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
        
        # Last states of this training step returned as expected to be beginning
        # of next training step.
        return [states[-1]]
        

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
        
        episodic_rewards_reporter = PPOEpisodicRewardsReporter(
            batch_size=BATCH_SIZE,
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
