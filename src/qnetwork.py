
import torch

class QNetwork(torch.nn.Module):
    """ As in Mnih et al 2015 """

    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.l1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.l2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.l3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.l4 = torch.nn.Linear(3136, 512)
        self.l5 = torch.nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        x = torch.nn.functional.relu(self.l4(x.view(x.size(0), -1)))
        return self.l5(x)

