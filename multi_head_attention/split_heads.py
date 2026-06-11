def split_heads(x,num_heads):
  batch_size,seq_len,d_model=x.shape
  d_k=d_model//num_heads
  x=x.reshape(
      batch_size,
      seq_len,
      num_heads,
      d_k
  )
  x=x.transpose(
      0,2,1,3
  )
  return x