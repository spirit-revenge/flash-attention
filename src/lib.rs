pub mod attention;

#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

pub use attention::{
    benchmark_attention_shapes, flash_attention as flash_attention_impl,
    flash_attention_cpu, make_qkv_example, naive_attention, scaled_dot_product_attention,
    BenchmarkResult, Tensor4,
};

#[cfg(feature = "python")]
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
fn nested_to_flat_and_shape(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f32>)> {
    if let Ok(values) = obj.extract::<Vec<f32>>() {
        return Ok((vec![values.len()], values));
    }
    if let Ok(values) = obj.extract::<Vec<Vec<f32>>>() {
        let rows = values.len();
        let cols = values.first().map_or(0, |row| row.len());
        let mut flat = Vec::with_capacity(rows * cols);
        for row in values {
            flat.extend(row);
        }
        return Ok((vec![rows, cols], flat));
    }
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
    if let Ok(list) = obj.call_method0("tolist") {
        return nested_to_flat_and_shape(&list);
    }

    Err(PyErr::new::<PyTypeError, _>(format!(
        "Expected q/k/v to be a float tensor or nested sequence, got {}",
        obj.get_type().name()?
    )))
}

#[cfg(feature = "python")]
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
#[pymodule]
fn flash_attention(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flash_attention_py, m)?)?;
    m.add_function(wrap_pyfunction!(scaled_dot_product_attention_py, m)?)?;
    m.add_function(wrap_pyfunction!(naive_attention_py, m)?)?;
    Ok(())
}
