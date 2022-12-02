#
#   From:
#       alearn_cartpole_from_openaiGym.py
#
#   See Also:
#       https://www.gymlibrary.dev/
#
#   Notes:
#       Switching from q-learning to random learning model.
#       Only keeping new values if performance improves.
#       CartPole will 'truncate' the game at 500 steps ( default setting )
#
#       WOW!  Adding basis to the observation dramatically improved results.
#           It will almost always improve to score of 100 minimum,
#           and often hits score of 500 minimum.
#       This also demonstrates that q-learning is not ideal for cartpole problem.
#
#   This code is exactly per the source page except:
#       Removed q-learning and tf.Models
#       Using random model + jiggle
#       only keeping improved model if it's minimum improves after a number of tries
#

import numpy as np
import gym
import random

LIMIT_TRIES_WITHOUT_IMPROVEMENT = 250
STATS_REPLAY = 10
MODEL_SHAPE = (5,)
SCORE_LIMIT = 500       # this is a 'gym' limitation

def jiggle( drift ):
    weights = np.zeros( MODEL_SHAPE )
    rfunc = np.vectorize( lambda x: random.uniform(-1,+1) )
    weights = rfunc( weights )
    return weights * drift

def pick_action( env, observation, model ):
    r"""Observation is ndarray[4]."""

    # append a basis to the observation
    state = np.append( observation, 1. )
    weight = np.sum( state * model )

    # print("env.action_space=",env.action_space)
    actions_count = env.action_space.n - env.action_space.start
    # print('len=',actions_count)

    action_weight = np.clip( weight, 0., .99999 )
    # print('clipped=',action_weight)

    action = int( np.floor(actions_count * action_weight ) )
    # print('action=',action)

    # shift into action_space
    action += env.action_space.start

    return action

def print_stats( key, stats ):
    print( key + ' scores max={},med={} min={} avg={:.2f} std={:.2f}'.format(
        max(stats), min(stats),
        np.median(stats), np.average(stats), np.std(stats)
    ))
    return

def is_better( old_stats, new_stats ):
    r"""Stats are considered 'better' depending on multiple values."""
    min0 = min(old_stats)
    min1 = min(new_stats)
    if min1 > min0: return True
    if min1 < min0: return False
    med0 = np.median( old_stats )
    med1 = np.median( new_stats )
    return med1 > med0

########################################################################################################################
#
#   Note: observation space for CartPole-v1 is an array of 4 floating point numbers
#   See: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L40
#
if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    model = jiggle(1)
    best_model = model.copy()
    print("model=",model)

    last_score = 0.
    best_score = 0.
    best_stats = np.zeros(STATS_REPLAY)

    no_change = 0
    model_count = 0

    # observation is ndarray(4) of float
    observation, info = env.reset()

    # try another model
    while no_change < LIMIT_TRIES_WITHOUT_IMPROVEMENT:

        # play five times, evaluate statistics
        stats = np.empty(STATS_REPLAY)
        for play in range(STATS_REPLAY):

            # play an episode
            time_step = 0
            while True:

                time_step += 1

                action = pick_action( env, observation, model )
                # print('action=',action,' ts=',time_step)

                # invoke the game environment
                observation, reward, terminated, truncated, info = env.step( action )

                if terminated or truncated:
                    observation, info = env.reset()
                    # print('end episode at time_step =',time_step )
                    break

            # end of play
            stats[play] = time_step

        # print_stats( '10 play round', stats )

        # evaluate model
        last_score = np.min( stats )
        if is_better( best_stats, stats ):
            best_score = last_score
            best_stats = stats
            best_model = model.copy()
            model_count += 1
            no_change = 0
            print('IMPROVING best=',best_score)
            # print('best_model=',best_model)
        else:
            no_change += 1
            # print('no_change=',no_change)

        if best_score>=SCORE_LIMIT:
            break

        # new model to test
        drift = 1. if best_score<10 else 10./best_score
        model = best_model + jiggle( drift )

    # end of episodes
    print('\nmodel count = ',model_count )
    print('best model score = ',best_score )
    print_stats( 'best', best_stats )
