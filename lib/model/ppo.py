import os
import time
import numpy as np

from lib.pretrain_actor_type import PretrainActorType
from .policy_type import PolicyType
from .environment import Environment
from .ppo_actor_critic import PPOActorCritic
from typing import List
from .state_factory import StateFactory
from .ppo_episodic_reporter import PPOEpisodicReporter
from .ppo_reporter import PPOReporter
from .ppo_running_statistics import PPORunningStatistics
from ..hyperparameters import PRETRAIN_ACTOR_TYPE, PRETRAINING_STEPS, STATE_SIZE, ACTION_SIZE, BATCH_SIZE, MINIBATCH_SIZE, \
    T, EPOCHS, GAMMA, PPO_PROBABILITY_RATIO_CLIP_THRESHOLD

# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


class PPO:
    actor_critic: PPOActorCritic
    environments: List[Environment]
    timestep = 1


    def __init__(
        self,
        actor_critic: PPOActorCritic,
        environments: List[Environment],
    ):
        assert(len(environments) == BATCH_SIZE)
        assert(len(environments) == len(set(environments)))

        self.actor_critic = actor_critic
        self.environments = environments


    def __create_empty_states(self):
        return tf.convert_to_tensor([
            StateFactory.create_empty_state(state_size=STATE_SIZE)
                for _ in range(BATCH_SIZE)
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


    def __run_exploration_policy(self, states: tf.Tensor):
        return self.actor_critic.policy(
            states,
            PolicyType.OLD.value,
            batch_size=BATCH_SIZE,
            actions_reference=tf.fill([BATCH_SIZE, ACTION_SIZE,], -1),
            use_actions_reference=False,
        )


    def __explore(
        self,
        states: List[tf.Tensor],
        reporter: PPOReporter,
        episodic_rewards_reporter: PPOEpisodicReporter,
    ):
        actions = []
        probabilities = []
        rewards = []
        
        for i in range(T):
            actions_env, probabilities_env = self.__run_exploration_policy(
                states=states[i],
            )
            
            env_tuples = []

            for batch_index in range(BATCH_SIZE):
                environment = self.environments[batch_index]

                # Retrieve epsiode before its potential update.
                #
                # Otherwise, if the episode is updated, the reward returned
                # will match the previous episode.
                episode = environment.episode
                
                state, reward = environment.perform_action(
                    action=actions_env[batch_index],
                    timestep=self.timestep,
                    reporter=reporter,
                )
                
                episodic_rewards_reporter.record_episodic_statistics(
                    episode=episode,
                    reward=reward,
                    batch_index=batch_index,
                )
                    
                env_tuples.append((state, reward,))

                next_states_env = [env_tuple[0] for env_tuple in env_tuples]
                rewards_env = [env_tuple[1] for env_tuple in env_tuples]

            actions.append(tf.convert_to_tensor(actions_env, dtype=tf.int32))
            probabilities.append(tf.convert_to_tensor(probabilities_env, dtype=tf.float64))
            rewards.append(tf.convert_to_tensor(rewards_env, dtype=tf.float64))
            states.append(tf.convert_to_tensor(next_states_env, dtype=tf.float64))
            
            self.timestep += 1

        next_states = tf.convert_to_tensor(states[-1], dtype=tf.float64)
        states = tf.convert_to_tensor(states[:-1], dtype=tf.float64)

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        probabilities = tf.convert_to_tensor(probabilities, dtype=tf.float64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)

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
        

        total_seconds = time.time() - total_seconds

        running_stats = PPORunningStatistics(
            timestep=self.timestep - 1,
            mean_batch_reward=np.mean(rewards),
            mean_actor_loss=mean_actor_loss,
            mean_critic_loss=mean_critic_loss,
            exploration_seconds=exploration_seconds,
            learning_seconds=learning_seconds,
            training_step_seconds=total_seconds,
        )

        reporter.record_running_statistics(running_stats)

        return [next_states]
    

    def __pretrain_actor(
        self,
        demonstration_environment: Environment,
        demonstration_actions: tf.Tensor,
    ):
        demonstration_actions_count = len(demonstration_actions)
        
        states = [StateFactory.create_empty_state(state_size=STATE_SIZE)]
        actions = []

        for i in range(demonstration_actions_count):
            action = demonstration_actions[i]

            state, _ = demonstration_environment.perform_demonstration_action(action)

            states.append(state)
            actions.append(action)
            
        states = np.array(states[:-1])
        actions = np.array(actions)
        
        for i in range(PRETRAINING_STEPS):
            start = time.time()
 
            transition_indices = np.random.choice(
                demonstration_actions_count,
                size=MINIBATCH_SIZE,
            )
            
            states_batch = states[transition_indices]
            actions_batch = actions[transition_indices]
            
            actor_model = self.actor_critic.actor_model
            actor_optimizer = self.actor_critic.actor_pretraining_optimizer
            
            actor_model.save(f'{os.path.dirname(__file__)}/../../pretrained_actor.keras')
        
            with tf.GradientTape() as tape:
                y = [
                    self.actor_critic.policy(
                        states_batch,
                        PolicyType.NORMAL.value,
                        batch_size=MINIBATCH_SIZE,
                        actions_reference=actions_batch,
                        use_actions_reference=True,
                    )[1] for _ in range(T)
                ]

                y = tf.convert_to_tensor(y)

                unscaled_actor_loss = -tf.reduce_sum(tf.clip_by_value(tf.math.log(y), -99999.0, 0.0))
                actor_loss = actor_optimizer.scale_loss(unscaled_actor_loss)
                

            actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))
            
            print(f'[{i+1}/{PRETRAINING_STEPS}] Unscaled demonstration loss: {unscaled_actor_loss}, Seconds: {time.time() - start}')
        
        self.actor_critic.update_old_actor_weights()
        
        print('Behavioural cloning pretraining completed.')
        

    def run(
        self,
        demonstration_environment: Environment,
        demonstration_actions: tf.Tensor,
    ):
        reporter = PPOReporter()
        reporter.start()
        
        if(PRETRAIN_ACTOR_TYPE == PretrainActorType.PRETRAIN):
            self.__pretrain_actor(
                demonstration_environment=demonstration_environment,
                demonstration_actions=demonstration_actions,
            )
        
        episodic_rewards_reporter = PPOEpisodicReporter(
            batch_size=BATCH_SIZE,
            reporter=reporter,
        )
        
        starting_states = [self.__create_empty_states()]

        while True:
            starting_states = self.__run_training_step(
                states=starting_states,
                reporter=reporter,
                episodic_rewards_reporter=episodic_rewards_reporter,
            )
