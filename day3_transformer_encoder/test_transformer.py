import torch
from transformer_block import TransformerBlock
from feed_forward import FeedForward
x = torch.randn(32,100,512)

block = TransformerBlock(
    512,
    8,
    2048,
    FeedForward
)

out = block(x)

print(out.shape)