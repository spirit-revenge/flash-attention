# Flash Attention

中文 | [English](README.md)

一个使用 Rust 实现的 CPU 版 Flash Attention 和 Scaled Dot-Product Attention，支持可选的 Python 和 NumPy 接口。

## 项目特性

- 纯 Rust CPU 实现
- 基于分块和在线 softmax 的 Attention 算法
- 支持多 batch、多 head
- 支持 causal mask
- 支持自定义 scale 和 block size
- 可选 PyO3 Python 扩展

## 安装

### Rust

构建并运行测试：

```bash
cargo build --release
cargo test
```

### Python

安装构建依赖：

```bash
python3 -m pip install numpy maturin
```

在项目根目录构建并安装 Python 扩展：

```bash
maturin build --release --features python
python3 -m pip install --force-reinstall target/wheels/*.whl
```

## 使用

### Python

输入形状通常为 `[batch, heads, seq, dim]`，也支持 `[seq, dim]` 等低维输入。

```python
import numpy as np
import flash_attention

q = np.random.randn(2, 3, 5, 4).astype(np.float32)
k = np.random.randn(2, 3, 5, 4).astype(np.float32)
v = np.random.randn(2, 3, 5, 4).astype(np.float32)

output = flash_attention.flash_attention_py(
    q,
    k,
    v,
    causal=False,
    block_size=2,
)

print(np.asarray(output).shape)  # (2, 3, 5, 4)
```

也可以使用标准 SDPA 命名的接口：

```python
output = flash_attention.scaled_dot_product_attention_py(
    q, k, v, causal=True
)
```

运行项目自带的 Python demo：

```bash
python3 test.py
```

> [!WARNING]
> 这是一个个人学习项目，请勿用于正式项目。