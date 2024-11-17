import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, filename, hidden_dim=512):
        super(DQN, self).__init__()

        C, H, W = n_observations

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(n_observations)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        #Save filename for model
        self.filename = filename

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    
    # Save a model
    def save_model(self):
        torch.save(self.state_dict(), f'./models/{time.gmtime(0)}' + self.filename + 'py.pth')
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
