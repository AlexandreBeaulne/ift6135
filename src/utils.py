
# third party imports
import torch

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

GAMES = {'pong': 'PongNoFrameskip-v4',
         'seaquest': 'SeaquestNoFrameskip-v4',
         'spaceinvaders': 'SpaceInvadersNoFrameskip-v4',
         'frostbite': 'FrostbiteNoFrameskip-v4',
         'beamrider': 'BeamRiderNoFrameskip-v4'}

class Epsilon(object):

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, t):
        return self.end + max(0, 1 - t / self.decay) * (self.start - self.end)

