"""
Algorithm 1 from Mnih et al ICML 2016 (except single-threaded)
"""

# stdlib imports
import collections
from datetime import datetime
from random import random
from statistics import mean

# third party imports
import gym
import torch

# local imports
from qnetwork import QNetwork
from utils import Epsilon
from utils import FloatTensor
from utils import GAMES
import wrappers

def train(game, num_steps=60000000, lr=0.00025, gamma=0.99, C=10000, batch_size=32):

    env = wrappers.wrap(gym.make(GAMES[game]))

    num_actions = env.action_space.n

    Q1 = QNetwork(num_actions)
    Q2 = QNetwork(num_actions)
    Q2.load_state_dict(Q1.state_dict())

    if torch.cuda.is_available():
        Q1.cuda()
        Q2.cuda()

    epsilon = Epsilon(1, 0.05, 1000000)
    optimizer = torch.optim.Adam(Q1.parameters(), lr=lr)

    state1 = env.reset()

    t, last_t, loss, episode, score = 0, 0, 0, 0, 0
    last_ts, scores = datetime.now(), collections.deque(maxlen=100)

    while t < num_steps:

        qvalues = Q1(state1)
        if random() < epsilon(t):
            action = env.action_space.sample()
        else:
            action = qvalues.data.max(dim=1)[1][0]

        q = qvalues[0][action]

        state2, reward, done, _info = env.step(action)
        score += reward

        if not done:
            y = gamma * Q2(state2).detach().max(dim=1)[0][0] + reward
            state1 = state2
        else:
            reward = FloatTensor([reward])
            y = torch.autograd.Variable(reward, requires_grad=False)
            state1 = env.reset()
            scores.append(score)
            score = 0
            episode += 1

        loss += torch.nn.functional.smooth_l1_loss(q, y)

        t += 1

        if t % batch_size == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

        if t % C == 0:
            Q2.load_state_dict(Q1.state_dict())
            torch.save(Q1.state_dict(), 'qlearning_{}.pt'.format(game))

        if t % 1000 == 0:
            ts = datetime.now()
            datestr = ts.strftime('%Y-%m-%dT%H:%M:%S.%f')
            avg = mean(scores) if scores else float('nan')
            steps_per_sec = (t - last_t) / (ts - last_ts).total_seconds()
            l = '{} step {} episode {} avg last 100 scores: {:.2f} ε: {:.2f}, steps/s: {:.0f}'
            print(l.format(datestr, t, episode, avg, epsilon(t), steps_per_sec))
            last_t, last_ts = t, ts

