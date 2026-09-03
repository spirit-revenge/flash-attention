use flash_attention::{flash_attention_cpu, make_qkv_example, naive_attention};

/// `main` 函数提供了一个非常简单的运行示例，用来验证：
/// - 生成真实 Q/K/V 样本；
/// - 计算 flash attention 输出；
/// - 计算参考 naive attention 输出；
/// - 打印出前几项结果，方便快速比较。
fn main() {
    // 生成一个小规模但包含 batch / head / seq / dim 的样例张量。
    // 这里使用 `make_qkv_example` 来制造更接近真实场景的数据分布。
    let (q, k, v) = make_qkv_example(2, 3, 5, 4, 1234);

    // 使用分块式 flash attention 计算输出。
    let out = flash_attention_cpu(&q, &k, &v, 2);

    // 用朴素实现作为参考值，便于比对结果是否一致。
    let ref_out = naive_attention(&q, &k, &v);

    // 打印一下张量形状，确认 batch/head/seq/dim 的布局符合预期。
    println!("batch={}, heads={}, seq={}, dim={}", q.batch, q.heads, q.seq, q.dim);

    // 查看前几个元素，确认输出值是正常的浮点数，而不是空值或者无效值。
    println!("flash attention result sample: {:?}", &out.data[..8]);
    println!("reference result sample: {:?}", &ref_out.data[..8]);
}
