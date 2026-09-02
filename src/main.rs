use flash_attention::{flash_attention_cpu, make_qkv_example, naive_attention};

fn main() {
    let (q, k, v) = make_qkv_example(2, 3, 5, 4, 1234);
    let out = flash_attention_cpu(&q, &k, &v, 2);
    let ref_out = naive_attention(&q, &k, &v);

    println!("batch={}, heads={}, seq={}, dim={}", q.batch, q.heads, q.seq, q.dim);
    println!("flash attention result sample: {:?}", &out.data[..8]);
    println!("reference result sample: {:?}", &ref_out.data[..8]);
}
