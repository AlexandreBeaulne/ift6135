
import argparse
import gym

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

    env = gym.make(game)
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.close()

if __name__ == '__main__':
    main()

