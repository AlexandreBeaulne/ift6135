"""
Algortithm 1 from Mnih et al ICML 2016
"""

# stdlib imports
from random import random

# third party imports
import torch

# local imports
from qnetwork import QNetwork

def worker():
    t = 0



num_actions = 5

Q1 = QNetwork(num_actions)
Q2 = QNetwork(num_actions)
Q2.load_state_dict(Q1.state_dict())
T = 0


def train(env, epsilon=0.01, Tmax=5, lr=1e-4, gamma=0.99, Itarget=2, Iasync=2):
    global T

    t = 0

    optimizer = torch.optim.Adam(Q1.parameters(), lr=lr)

    state1 = env.reset()

    loss = 0

    while T < Tmax:

        qvalues = Q1(state1)
        if random() < epsilon:
            action = env.action_space.sample()
        else:
            action = qvalues.data.max(dim=1)[1][0]

        q = qvalues[0][action]

        state2, reward, done, _info = env.step(action)

        if not done:
            y = gamma * Q2(state2).detach().max(dim=1)[0][0] + reward
        else:
            y = reward

        loss += torch.nn.functional.smooth_l1_loss(q, y)

        state1 = state2

        T += 1
        t += 1

        if T % Itarget == 0:
            Q2.load_state_dict(Q1.state_dict())

        if t % Iasync == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

