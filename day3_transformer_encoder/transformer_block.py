import math
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self,d_model,num_heads,hidden_dim,FeedForward):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.d_k=d_model//num_heads
    self.hidden_dim=hidden_dim
    self.ffn=FeedForward(
        self.d_model,
        self.hidden_dim
    )

    self.W_Q=nn.Linear(
        d_model,
        d_model
    )
    self.W_K=nn.Linear(
        d_model,
        d_model
    )
    self.W_V=nn.Linear(
        d_model,
        d_model
    )
    self.W_O=nn.Linear(
        d_model,
        d_model
    )
  def attention(self,Q,K,V):
    attenstion_score=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
    softmax=nn.Softmax(dim=-1)
    return torch.matmul(softmax(attenstion_score),V)
  def layer_norm(self,x,eps=1e-5):
    mean=torch.mean(x,dim=-1,keepdim=True)
    var=torch.var(x,dim=-1,keepdim=True)

    x_norm=(x-mean)/torch.sqrt(var+eps)
    return x_norm
  def forward(self,x):
    Q=self.W_Q(x)
    K=self.W_K(x)
    V=self.W_V(x)

    attn=self.attention(Q,K,V)
    attn=self.W_O(attn)
    x=x+attn 

    x=self.layer_norm(x)
    ffn_out=self.ffn(x)
    x=x+ffn_out
    x=self.layer_norm(x)

    return x