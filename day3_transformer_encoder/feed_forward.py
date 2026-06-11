import torch
import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self,d_model,hidden_dim):
    super().__init__()
    self.fc1=nn.Linear(
        d_model,
        hidden_dim
    )
    self.relu=nn.ReLU()
    self.fc2=nn.Linear(
        hidden_dim,
        d_model
    )
  def forward(self,x):
    x=self.fc1(x)
    x=self.relu(x)
    x=self.fc2(x)
    return x
