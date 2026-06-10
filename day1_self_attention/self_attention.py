import numpy as np
def softmax(x):
  ep=np.exp(x-np.max(x,axis=-1,keepdims=True))
  return ep/np.sum(ep,axis=-1,keepdims=True)
def self_attention(Q,K,V):
  score=np.dot(Q,K.T)
  score=score/np.sqrt(K.shape[-1])
  weight=softmax(score)
  output=weight @ V
  return output,weight

Q = np.array([
    [1,0],
    [0,1]
])

K = np.array([
    [1,0],
    [0,1]
])

V = np.array([
    [10,0],
    [0,20]
])

output, weights = self_attention(Q,K,V)

print(weights)
print(output)