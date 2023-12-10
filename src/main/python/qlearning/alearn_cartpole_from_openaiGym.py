#
#   From:
#       _other_tutorials/tfwiki_qlearning_DRL_cartpole.py
#
#   See Also:
#       https://www.gymlibrary.dev/
#
#   Notes:
#       Attempting to improve learning.
#
#   Lessons:
#       I think that q-learning 'overtrains' the models.
#           That means that the model may display some skill, but then forget it.
#           I believe this is due to the sensitivity of the cartpole system,
#           vs the 'gross learning' of q-learning algorithm ( as opposed to 'fine learning' )
#       Modifying the reward seemed to have the biggest impact on overall performance.
#           Mostly the reward is smaller, but gets bigger after many time_steps
#
#   This code is exactly per the source page except:
#       Using multiple alternate models
#       using function to modify reward based on time_step
#       keeping a copy of the best_play and adding it to the replay queue
#       Added statistics at end to better evaluate performance over episodes.
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
final_epsilon = 0       # 0.01

REPLAY_MEMORY = 3_000 # 10_000
STATE_SHAPE = (5,)      # cartpole observation/state is 4 floats, and I add one basis value

VERSION = 3     # selects from model_function and reward_function

# model_function[VERSION](shape)
model_function = [
    None,
    lambda s: create_model_v1(s),
    lambda s: create_model_v2(s),
    lambda s: create_model_v3(s),
    lambda s: create_model_v4(s),
]

# reward_function[VERSION](time_step,reward)
reward_function = [
    None,
    lambda t,r: r,
    lambda t,r: t / 200.,
    lambda t,r: 1. if t>80 else t / 80.,
    lambda t,r: t / 200.,
]

########################################################################################################################

def create_model_v1(shape):
    r"""Performance is pretty random.  Usually gets better around 100-200, then declines towards 10.

Scores over time:
scores[0:100] max=54.0,med=11.0 avg=13.78 std=8.27
scores[100:200] max=186.0,med=18.5 avg=25.19 std=23.29
scores[200:300] max=45.0,med=9.0 avg=11.48 std=6.95
scores[300:400] max=230.0,med=25.0 avg=37.01 std=36.29
scores[400:500] max=233.0,med=10.0 avg=26.28 std=37.21

    """

    inputs = x = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Dense(units=24, activation='ReLU')(x)
    x = tf.keras.layers.Dense(units=24, activation='ReLU')(x)
    x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    outputs = x = tf.keras.layers.Softmax( axis=-1 )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs,name='model_v1')


def create_model_v2(shape):
    r"""The thought here is that allowing more emphasis on negative sums would improve learning.
    I actually put 96 units in the first layer before I realized that the input only has 4 values.
    The modified reward function is probably having a greater impact

Scores over time:
scores[0:100] max=233.0,med=31.0 avg=40.12 std=32.78
scores[100:200] max=347.0,med=21.0 avg=41.53 std=61.97
scores[200:300] max=10.0,med=8.0 avg=8.38 std=0.78
scores[300:400] max=10.0,med=8.0 avg=8.27 std=0.72
scores[400:500] max=23.0,med=9.0 avg=10.90 std=4.36
    """

    inputs = x = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Dense(units=24, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(units=24, activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    outputs = x = tf.keras.layers.Softmax( axis=-1 )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs,name='model_v2')

def create_model_v3(shape):
    r"""Fewer units with more layers, and using SeLU which is supposed to self-normalize.

Scores over time:
scores[0:100] max=46.0,med=11.0 avg=14.00 std=7.38
scores[100:200] max=115.0,med=29.0 avg=33.82 std=23.56
scores[200:300] max=74.0,med=26.0 avg=26.65 std=15.18
scores[300:400] max=63.0,med=33.0 avg=34.50 std=9.77
scores[400:500] max=56.0,med=34.0 avg=33.40 std=6.77
    """

    inputs = x = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Dense(units=8, activation='selu')(x)
    x = tf.keras.layers.Dense(units=8, activation='selu')(x)
    x = tf.keras.layers.Dense(units=8, activation='selu')(x)
    x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    outputs = x = tf.keras.layers.Softmax( axis=-1 )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs,name='model_v3')

def create_model_v4(shape):
    r"""

Scores over time:
scores[0:100] max=272.0,med=50.0 avg=61.93 std=49.62
scores[100:200] max=499.0,med=10.0 avg=52.36 std=116.84
scores[200:300] max=499.0,med=11.0 avg=54.49 std=121.37
scores[300:400] max=10.0,med=8.0 avg=8.24 std=0.79
scores[400:500] max=10.0,med=8.0 avg=8.32 std=0.76
    """

    inputs = x = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Dense(units=10, activation='ReLU')(x)
    x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    outputs = x = tf.keras.layers.Softmax( axis=-1 )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs,name='model_v4')

########################################################################################################################

def observation_to_state( observation ):
    r"""Observation is 4 float values.  State includes a BASIS value."""
    return np.append( observation, 1. )

FIRST = tf.constant( [1,0] )
SECOND = tf.constant( [0,1] )

def predict( model, inputs ):
    r"""Takes 'state' as inputs, produces integral logits for action space.
    This is used to take an action at each step."""
    q_values = model(inputs)
    # debatch = remove one dimension
    q_values = tf.squeeze(q_values)

    # random choice based on first value = did not provide much value, performance was lamer
    # if tf.random.uniform( (), maxval=1 ) < q_values[0]:
    #     return FIRST
    # else:
    #     return SECOND

    # transform softmax(0,+1) into [0/1]
    q_values = tf.cast( tf.round(q_values), tf.int32 )
    return q_values


def train_model( model, replay_buffer ):

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
        # print('loss=',loss)
        # loss = loss * tf.cast( slow_learning_rate(time_step), tf.float32 )
        # print('lossX=',loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


########################################################################################################################
#
#   Note: observation space ( eg. state ) for CartPole-v1 is an array of 4 floating point numbers
#   See: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L40
#
if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    model = model_function[VERSION]( STATE_SHAPE )
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = []  #deque(maxlen=REPLAY_MEMORY)
    epsilon = initial_epsilon
    scores = np.empty( num_episodes )

    score = 0
    best_score = 0
    best_play = []

    for episode_id in range(num_episodes):

        # state = env.reset()     # older version
        observation, info = env.reset()     # v0.26.2
        state = observation_to_state( observation )

        epsilon = max(
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)

        for time_step in range(max_len_episode):
            env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = predict( model, np.expand_dims(state, axis=0) ).numpy()
                action = action[0]

            # invoke the game environment
            # next_state, reward, done, info = env.step(action)   # older version
            next_observation, reward, terminated, truncated, info = env.step(action)     # v0.26.2
            reward = reward_function[VERSION]( time_step, reward )

            if terminated or truncated:
                next_observation, info = env.reset()
                scores[ episode_id ] = time_step
            done = terminated or truncated          # conversion to older values: done, next_state
            next_state = observation_to_state( next_observation )

            # Game Over?
            reward = -10. if done else reward
            # (state, action, reward, next_state, 1/0 for done)
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # state
            state = next_state

            if done:
                score = time_step
                if score>best_score:
                    best_score = score
                    best_play = replay_buffer[ -time_step: ]
                print("episode=%4d, epsilon=%.4f, score=%4d best=%4d" % (episode_id, epsilon, score,best_score))
                break

            # originally we trained after each step in an episode
            # this is absolutely correct to implement q-learning,
            #   HOWEVER it means stepping away from good solutions when training cart-pole
            replay_buffer = replay_buffer[ -REPLAY_MEMORY: ]
            train_model( model, best_play + replay_buffer )

        # continue episode

    # end of q-learning


########################################################################################################################

    # show score statistics after all episodes
    print('\n\nScores over time:')
    for ix in range(0,5):
        start = ix*100
        end = start+100
        score100 = scores[start:end]
        print('scores[{}:{}] max={},med={} avg={:.2f} std={:.2f}'.format(
            start,end,max(score100),
            np.median(score100),
            np.average(score100),np.std(score100))
        )
