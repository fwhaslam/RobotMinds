#
#   From:
#       https://tf.wiki/en/basic/models.html#deep-reinforcement-learning-drl
#
#   See Also:
#       https://www.gymlibrary.dev/
#
#   Notes:
#       As currently implemented, it does not learn at all.  It tends to stick to one action.
#       Next step is to experiment with different models + loss algorithms.
#
#   This code is exactly per the source page except:
#       Removed japanese comments
#       'pip install gym'  ( gym version 0.26.2 )
#       converted newer gym response to older gym values
#       model.predict() returns array of integral logits for action space
#

import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-3
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01


class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)
        self.scale = tf.keras.layers.Softmax( axis=-1 )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        r"""Takes 'state' as inputs, produces integral logits for action space.
        This is used to take an action at each step."""
        q_values = self(inputs)
        # print('q_values=',q_values)
        # debatch = remove one dimension
        q_values = tf.squeeze(q_values)
        # print('debatch=',q_values)
        # softmax values into [0,1]
        q_values = self.scale( q_values )
        # transform [0,+1] into [0/1]
        q_values = tf.cast( tf.round(q_values), tf.int32 )
        # print('round(0/1)=',q_values)
        return q_values


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=10000)
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):

        # state = env.reset()     # older version
        state, info = env.reset()     # v0.26.2

        epsilon = max(
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)
        for t in range(max_len_episode):
            env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()
                action = action[0]

            # invoke the game environment
            # next_state, reward, done, info = env.step(action)   # older version
            next_state, reward, terminated, truncated, info = env.step(action)     # v0.26.2
            if terminated or truncated:
                next_state, info = env.reset()
            done = terminated or truncated

            # Game Over.
            reward = -10. if done else reward
            # (state, action, reward, next_state, 1/0 for done)
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # state
            state = next_state

            if done:
                print("episode %4d, epsilon %.4f, score %4d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                    map(np.array, zip(*random.sample(replay_buffer, batch_size)))

                q_value = model(batch_next_state)
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))