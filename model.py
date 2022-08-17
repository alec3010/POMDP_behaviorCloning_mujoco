import torch

class pyTorchModel(torch.nn.Module):
    def __init__(self, belief_dim, obs_dim, actor_hidden_dim, ch=2):
        super(pyTorchModel,self).__init__()
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(belief_dim + obs_dim, actor_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(actor_hidden_dim, actor_hidden_dim),
            torch.nn.ReLU()
        )
        
    def forward(self,x):
        x = self.policy(x)
        return x
