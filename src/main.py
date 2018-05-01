"""

IFT6135 Representation Learning - Final Project

Alexandre Beaulne I.D. 20087309
Amina Madzhun I.D. 20052277

References:

    * https://openreview.net/forum?id=HyiAuyb0b

Credits - Borrowed from:

    * https://github.com/Shmuma/ptan/blob/master/samples/dqn_speedup/

"""

# stdlib imports
import argparse

# third party imports
import gym

# local imports
import wrappers

def main():

    games = {'pong': 'PongNoFrameskip-v4',
             'seaquest': 'SeaquestNoFrameskip-v4',
             'spaceinvaders': 'SpaceInvadersNoFrameskip-v4',
             'frostbite': 'FrostbiteNoFrameskip-v4',
             'beamrider': 'BeamRiderNoFrameskip-v4'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, required=True, choices=games.keys())
    args = parser.parse_args()

    game = games[args.game]

    env = wrappers.wrap(gym.make(game))

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.close()

if __name__ == '__main__':
    main()

