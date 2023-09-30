import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.layers.recurrent import LSTM
from typing import Callable, Tuple
from .models.epsilon_model import EpsilonModel
from .models.rl_hyperparameters_model import RLHyperparametersModel

# DQN (Adapted from https://keras.io/examples/rl/deep_q_network_breakout/
# and https://github.com/ilija-jankovic/sqli_rl/blob/main/sqli_rl.ipynb)
class DQN:

    available_actions_range: range
    
    __hyperparameters: RLHyperparametersModel
    __epsilon_config: EpsilonModel
    __perform_action_callback: Callable[[int], Tuple[np.ndarray, float, bool]]
    __pre_training_completed_callback: Callable[[], Tuple[None]]

    def __init__(
            self,
            hyperparameters: RLHyperparametersModel,
            epsilon_config: EpsilonModel,
            available_actions_range: range,
            perform_action_callback: Callable[[int], Tuple[np.ndarray, float, bool]],
            pre_training_completed_callback: Callable[[], Tuple[None]]
        ):
        self.available_actions_range = available_actions_range
        self.__hyperparameters = hyperparameters
        self.__epsilon_config = epsilon_config
        self.__perform_action_callback = perform_action_callback
        self.__pre_training_completed_callback = pre_training_completed_callback

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time.
    #
    # NOTE: Uses legacy Adam optimizer for running on M1/M2 devices.
    #
    # TODO: Automatically use up-to-date Adam optimizer if no M1/M2 detected.
    def __create_optimizer(self):
        return tf.keras.optimizers.legacy.Adam(learning_rate=self.__hyperparameters.learning_rate, clipnorm=1.0)

    def __create_q_model(self, features: int, actions: int, batch_size: int):
        model = keras.Sequential([
            layers.SimpleRNN(512, activation='relu', batch_input_shape=(batch_size, features, 1),
                 return_sequences=True, stateful=True),
            layers.Dense(features),
            layers.Dense(1024),
            layers.Dense(1024),
            layers.Flatten(),
            layers.Dense(actions, activation='softmax')
        ])

        optimizer = self.__create_optimizer()
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        model.build(input_shape=(batch_size, features, 1))

        return model

    def create_model(self):
        features = self.__hyperparameters.feature_count
        actions = self.__hyperparameters.action_count
        batch_size = self.__hyperparameters.batch_size

        # The first model makes the predictions for Q-values which are used to
        # make an action.
        model = self.__create_q_model(features, actions, batch_size)
        
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        model_target = self.__create_q_model(features, actions, batch_size)

        return model, model_target
    
    # Workaround for using LSTMs with only 1 row after training with a batch
    # size greater than 1.
    #
    # Needed as LSTMs must have their batch sizes defined upfront, i.e., they
    # cannot have variable batch sizes like other layers.
    # 
    # Adapated from solution by Jason Brownlee from
    # https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/.
    def __create_one_step_model(self, model: keras.Sequential):
        features = self.__hyperparameters.feature_count
        actions = self.__hyperparameters.action_count

        new_model = self.__create_q_model(features, actions, batch_size=1)

        # Copy weights.
        old_weights = model.get_weights()
        new_model.set_weights(old_weights)

        return new_model
    
    def create_empty_state(self):
        return np.array([-1] * self.__hyperparameters.feature_count, dtype='float32')
    
    def run(self, model: keras.Sequential, model_target: keras.Sequential):
        optimizer = self.__create_optimizer()

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0

        epsilon = self.__epsilon_config.epsilon
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 1000000
        # Train the model after 4 actions
        update_after_actions = 4
        # How often to update the target network
        update_target_network = 100
        # Using huber loss for stability
        loss_function = keras.losses.Huber()
        training = True

        while True:  # Run until solved
            state = self.create_empty_state()
            episode_reward = 0

            for _ in range(1, self.__hyperparameters.max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < self.__epsilon_config.random_frame_count or epsilon > np.random.rand(1)[0]:
                    action = np.random.randint(
                        self.available_actions_range.start, self.available_actions_range.stop)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)

                    one_step_model = self.__create_one_step_model(model)
                    action_probs = one_step_model(state_tensor, training=False)[0]

                    # Mask all unavailable actions.
                    masked_probs = [action_probs[i] if i in self.available_actions_range 
                                    else float('-inf') for i in range(len(action_probs))]

                    # Take the best action.
                    action = tf.argmax(masked_probs).numpy()

                # Decay probability of taking random action
                epsilon -= self.__epsilon_config.epsilon_interval / self.__epsilon_config.greedy_frame_count
                epsilon = max(epsilon, self.__epsilon_config.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done = self.__perform_action_callback(action)

                episode_reward += reward

                # If the reward is positive, a context has been solved.
                # Keep track of this.
                if reward > 0:
                    # Log details
                    template = 'Running reward: {:.2f}\t Episode {}\t Frame count: {}\t Epsilon: {:.2f}%'
                    print(template.format(running_reward, episode_count + 1, frame_count, epsilon * 100.0))

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over its defined size
                if training and frame_count % update_after_actions == 0 and len(done_history) > self.__hyperparameters.batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=self.__hyperparameters.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices]) \
                        .reshape(self.__hyperparameters.batch_size, self.__hyperparameters.feature_count, 1)
                    state_next_sample = np.array([state_next_history[i] for i in indices]) \
                        .reshape(self.__hyperparameters.batch_size, self.__hyperparameters.feature_count, 1)
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target.predict(state_next_sample, batch_size=self.__hyperparameters.batch_size , verbose=0)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.__hyperparameters.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.__hyperparameters.action_count)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample, training=True)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            # Stop training after training episodes.
            if episode_count == self.__hyperparameters.training_episodes:
                training = False
                self.__pre_training_completed_callback()

            episode_count += 1

            if episode_count >= self.__hyperparameters.episodes:
                print(f'DQN terminated at episode {episode_count} with a running reward of {running_reward}!')
                break



