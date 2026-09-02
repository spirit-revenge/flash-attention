import flash_attention
import numpy as np

batch_size = 2
head_size = 3
seq_len = 5
dim = 4

q = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)
k = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)
v = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)

out = flash_attention.flash_attention_py(q, k, v, block_size=2)
print(np.asarray(out).shape)
print(np.asarray(out).reshape(-1)[:8])
