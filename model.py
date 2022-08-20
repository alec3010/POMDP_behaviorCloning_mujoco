import torch
import torch.nn as nn
import torch.nn.functional as F

class pyTorchModel(torch.nn.Module):
    def __init__(self, belief_dim, obs_dim, actor_hidden_dim, action_dim, ch=2):
        super(pyTorchModel,self).__init__()
        self.policy = torch.nn.Sequential(
            nn.Linear(belief_dim , actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim)
        )
        
        
    def forward(self,x):
        x = self.policy(x)
        
        return x

