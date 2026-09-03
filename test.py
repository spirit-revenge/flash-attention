# 这是一个最小化的 Python 端 smoke test，
# 用于验证 Rust 编译的扩展模块是否可以被 Python 正常导入，
# 并且 Q/K/V 形状与接口调用是否符合预期。
import flash_attention
import numpy as np

# 设定一个小规模测试形状：
# [batch_size, head_size, seq_len, dim]
batch_size = 2
head_size = 3
seq_len = 5
dim = 4

# 构造随机 Q/K/V 张量，统一使用 float32，
# 这种数据类型和 Rust 中的 `f32` 计算保持一致，
# 能降低跨语言数值差异带来的干扰。
q = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)
k = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)
v = np.random.randn(batch_size, head_size, seq_len, dim).astype(np.float32)

# 通过 Python 绑定调用 Rust 实现的 flash attention。
# 这里使用 `block_size=2` 来模拟分块计算，
# 方便验证函数在 batched / multi-head 的情况下也能正确工作。
out = flash_attention.flash_attention_py(q, k, v, block_size=2)

# 输出结果保留原始 shape，并转换为 NumPy 数组便于观察。
print(np.asarray(out).shape)
print(np.asarray(out).reshape(-1)[:8])
