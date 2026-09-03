pub mod attention;

#[cfg(feature = "python")]
// PyO3 负责把 Rust 代码暴露成 Python 可调用模块。
// 这里引入 Python 异常类型和基础对象类型，便于在 Rust 中处理 Python 侧的输入输出。
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

// 公开导出由 attention 模块提供的核心 API，方便外部 Rust 用户直接使用。
pub use attention::{
    benchmark_attention_shapes, flash_attention as flash_attention_impl,
    flash_attention_cpu, make_qkv_example, naive_attention, scaled_dot_product_attention,
    BenchmarkResult, Tensor4,
};

#[cfg(feature = "python")]
/// 规范化 Python 侧的 shape 表达。
///
/// Python 传入的多维数组可能是：
/// - 1D: [dim]
/// - 2D: [seq, dim]
/// - 3D: [batch, seq, dim]
/// - 4D: [batch, heads, seq, dim]
///
/// 这个函数把它们统一成标准的 `(batch, heads, seq, dim)` 形式，
/// 让后续的 Rust `Tensor4` 构造逻辑更统一。
fn normalize_shape(shape: &[usize]) -> (usize, usize, usize, usize) {
    match shape {
        [] => (1, 1, 1, 0),
        [d] => (1, 1, 1, *d),
        [seq, dim] => (1, 1, *seq, *dim),
        [batch, seq, dim] => (*batch, 1, *seq, *dim),
        [batch, heads, seq, dim] => (*batch, *heads, *seq, *dim),
        _ => panic!("attention arrays must have rank 1..=4; got shape {:?}", shape),
    }
}

#[cfg(feature = "python")]
/// 将 Python 中的嵌套数组转换成一维 `Vec<f32>` 和对应的 shape。
///
/// 这样 Rust 中的 `Tensor4::from_vec` 可以直接接收它，
/// 同时可以在输出时根据原始 shape 恢复为多层 Python list。
fn nested_to_flat_and_shape(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f32>)> {
    // 1D 输入：例如 [1.0, 2.0, 3.0]
    if let Ok(values) = obj.extract::<Vec<f32>>() {
        return Ok((vec![values.len()], values));
    }
    // 2D 输入：例如 [[...], [...]]
    if let Ok(values) = obj.extract::<Vec<Vec<f32>>>() {
        let rows = values.len();
        let cols = values.first().map_or(0, |row| row.len());
        let mut flat = Vec::with_capacity(rows * cols);
        for row in values {
            flat.extend(row);
        }
        return Ok((vec![rows, cols], flat));
    }
    // 3D 输入：例如 [batch][seq][dim]
    if let Ok(values) = obj.extract::<Vec<Vec<Vec<f32>>>>() {
        let batch = values.len();
        let seq = values.first().map_or(0, |row| row.len());
        let dim = values.first().and_then(|row| row.first()).map_or(0, |x| x.len());
        let mut flat = Vec::with_capacity(batch * seq * dim);
        for b in values {
            for s in b {
                flat.extend(s);
            }
        }
        return Ok((vec![batch, seq, dim], flat));
    }
    // 4D 输入：例如 [batch][heads][seq][dim]
    if let Ok(values) = obj.extract::<Vec<Vec<Vec<Vec<f32>>>>>() {
        let batch = values.len();
        let heads = values.first().map_or(0, |row| row.len());
        let seq = values.first().and_then(|row| row.first()).map_or(0, |x| x.len());
        let dim = values
            .first()
            .and_then(|row| row.first())
            .and_then(|row| row.first())
            .map_or(0, |x| x.len());
        let mut flat = Vec::with_capacity(batch * heads * seq * dim);
        for b in values {
            for h in b {
                for s in h {
                    flat.extend(s);
                }
            }
        }
        return Ok((vec![batch, heads, seq, dim], flat));
    }
    // 若输入对象本身有 `.tolist()`，例如 NumPy array，则递归转换为 Python list 再继续处理。
    if let Ok(list) = obj.call_method0("tolist") {
        return nested_to_flat_and_shape(&list);
    }

    Err(PyErr::new::<PyTypeError, _>(format!(
        "Expected q/k/v to be a float tensor or nested sequence, got {}",
        obj.get_type().name()?
    )))
}

#[cfg(feature = "python")]
/// 把 Rust 端的 `Tensor4` 输出恢复成 Python 中的嵌套 list 结构。
///
/// 也就是说，输出会保留输入的 shape 语义：
/// - `[dim]`
/// - `[seq, dim]`
/// - `[batch, seq, dim]`
/// - `[batch, heads, seq, dim]`
///
/// 这样 Python 侧不需要手动再重构 shape，调用体验更接近 NumPy。
fn tensor4_to_python_nested(py: Python<'_>, out: &Tensor4, shape: &[usize]) -> PyResult<PyObject> {
    match shape {
        [dim] => Ok(PyList::new_bound(py, out.data[..*dim].iter().copied()).into_any().unbind()),
        [seq, dim] => {
            let mut rows = Vec::with_capacity(*seq);
            for s in 0..*seq {
                let row: Vec<f32> = (0..*dim).map(|d| out.data[s * *dim + d]).collect();
                rows.push(PyList::new_bound(py, row).into_any());
            }
            Ok(PyList::new_bound(py, rows).into_any().unbind())
        }
        [batch, seq, dim] => {
            let mut batches = Vec::with_capacity(*batch);
            for b in 0..*batch {
                let mut rows = Vec::with_capacity(*seq);
                for s in 0..*seq {
                    let row: Vec<f32> = (0..*dim)
                        .map(|d| out.data[(b * *seq + s) * *dim + d])
                        .collect();
                    rows.push(PyList::new_bound(py, row).into_any());
                }
                batches.push(PyList::new_bound(py, rows).into_any());
            }
            Ok(PyList::new_bound(py, batches).into_any().unbind())
        }
        [batch, heads, seq, dim] => {
            let mut batches = Vec::with_capacity(*batch);
            for b in 0..*batch {
                let mut heads_rows = Vec::with_capacity(*heads);
                for h in 0..*heads {
                    let mut rows = Vec::with_capacity(*seq);
                    for s in 0..*seq {
                        let row: Vec<f32> = (0..*dim)
                            .map(|d| out.data[((b * *heads + h) * *seq + s) * *dim + d])
                            .collect();
                        rows.push(PyList::new_bound(py, row).into_any());
                    }
                    heads_rows.push(PyList::new_bound(py, rows).into_any());
                }
                batches.push(PyList::new_bound(py, heads_rows).into_any());
            }
            Ok(PyList::new_bound(py, batches).into_any().unbind())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "output rank {} is not supported",
            shape.len()
        ))),
    }
}

#[cfg(feature = "python")]
/// Rust 侧暴露给 Python 的主入口：`flash_attention_py`。
///
/// 它接收 Q/K/V，并自动完成：
/// 1. 读取 Python 对象；
/// 2. 扁平化为一维 `Vec<f32>`；
/// 3. 判断形状是否一致；
/// 4. 组装成 `Tensor4`；
/// 5. 调用 `scaled_dot_product_attention`；
/// 6. 按原始输入 shape 输出 Python list。
#[pyfunction]
#[pyo3(signature = (q, k, v, *, scale=None, causal=false, block_size=64))]
fn flash_attention_py(
    py: Python<'_>,
    q: Bound<'_, PyAny>,
    k: Bound<'_, PyAny>,
    v: Bound<'_, PyAny>,
    scale: Option<f32>,
    causal: bool,
    block_size: usize,
) -> PyResult<PyObject> {
    let (q_shape, q_flat) = nested_to_flat_and_shape(&q)?;
    let (k_shape, k_flat) = nested_to_flat_and_shape(&k)?;
    let (v_shape, v_flat) = nested_to_flat_and_shape(&v)?;

    let (b, h, s, d) = normalize_shape(&q_shape);
    let (kb, kh, ks, kd) = normalize_shape(&k_shape);
    let (vb, vh, vs, vd) = normalize_shape(&v_shape);

    if b != kb || h != kh || s != ks || d != kd || b != vb || h != vh || s != vs || d != vd {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "q, k, and v must share the same batch/head/sequence/dim shape",
        ));
    }

    let q_t = Tensor4::from_vec(b, h, s, d, q_flat);
    let k_t = Tensor4::from_vec(kb, kh, ks, kd, k_flat);
    let v_t = Tensor4::from_vec(vb, vh, vs, vd, v_flat);
    let out = scaled_dot_product_attention(&q_t, &k_t, &v_t, scale, causal, block_size);
    tensor4_to_python_nested(py, &out, &q_shape)
}

#[cfg(feature = "python")]
/// 这是 `flash_attention_py` 的别名，用来更贴近标准 SDPA API 的命名。
///
/// 它们在语义上是等价的：都是对 Q/K/V 做 scaled dot-product attention。
#[pyfunction]
#[pyo3(signature = (q, k, v, *, scale=None, causal=false, block_size=64))]
fn scaled_dot_product_attention_py(
    py: Python<'_>,
    q: Bound<'_, PyAny>,
    k: Bound<'_, PyAny>,
    v: Bound<'_, PyAny>,
    scale: Option<f32>,
    causal: bool,
    block_size: usize,
) -> PyResult<PyObject> {
    flash_attention_py(py, q, k, v, scale, causal, block_size)
}

#[cfg(feature = "python")]
/// 这是纯 Python 侧的 naive attention 入口，主要用于对照验证。
///
/// 与前面的函数不同，这个接口接受的是已经展开的 flat `Vec<f32>`，
/// 因为它更偏“低层测试/调试”的使用方式。
#[pyfunction]
fn naive_attention_py(
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> PyResult<Vec<f32>> {
    let q_t = Tensor4::from_vec(batch, heads, seq, dim, q);
    let k_t = Tensor4::from_vec(batch, heads, seq, dim, k);
    let v_t = Tensor4::from_vec(batch, heads, seq, dim, v);
    let out = naive_attention(&q_t, &k_t, &v_t);
    Ok(out.data)
}

#[cfg(feature = "python")]
/// 这是 Python 模块的导出入口。
///
/// 在 `maturin` 构建扩展模块时，`#[pymodule]` 会生成 `flash_attention` 这个 Python 模块，
/// 其中包含 `flash_attention_py`、`scaled_dot_product_attention_py` 和 `naive_attention_py` 三个函数。
#[pymodule]
fn flash_attention(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flash_attention_py, m)?)?;
    m.add_function(wrap_pyfunction!(scaled_dot_product_attention_py, m)?)?;
    m.add_function(wrap_pyfunction!(naive_attention_py, m)?)?;
    Ok(())
}
