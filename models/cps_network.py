import copy
import torch.nn as nn

class CPSNetwork(nn.Module):
    def __init__(self, s_model=None, t_model=None):
        super(CPSNetwork, self).__init__()
        self.branch1 = s_model
        self.branch2 = t_model

    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)