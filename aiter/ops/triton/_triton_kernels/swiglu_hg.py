import triton
import triton.language as tl

@triton.jit
def _swiglu_kernel(
    input_ptr,      # [M, 2K]
    output_ptr,     # [M, N]
    w_ptr, v_ptr,   # [K, N]
    b, c,           # [N]
    beta,           # scalar
    M, K, N,        # matrix dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.float32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize offsets and masks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
 
    # Initialize accumulators (BLOCK_SIZE_M x BLOCK_SIZE_N) for both matmuls
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Reduction over K dimension in BLOCK_SIZE_K
    for k_start in tl.range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        # Load x1 and x2: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # row stride for input matrix = 2 * K
        offs_mk = offs_m[:, None] * (2 * K) + offs_k[None, :]
        offs_m2k = offs_m[:, None] * (2 * K) + (offs_k[None, :] + K)
        mask_mk = mask_m[:, None] & mask_k[None, :]
        x1 = tl.load(input_ptr + offs_mk, mask=mask_mk)
        x2 = tl.load(input_ptr + offs_m2k, mask=mask_mk)

        # Load W and V: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        offs_kn = offs_k[:, None] * N + offs_n[None, :]
        mask_kn = mask_k[:, None] & mask_n[None, :]
        w = tl.load(w_ptr + offs_kn, mask=mask_kn)
        v = tl.load(v_ptr + offs_kn, mask=mask_kn)

        # Accumulate products
        acc1 += tl.dot(x1, w)
        acc2 += tl.dot(x2, v)

    # Load bias: shape (BLOCK_SIZE_N,)
    b_bias = tl.load(b + offs_n, mask=mask_n)
    c_bias = tl.load(c + offs_n, mask=mask_n)

    # Add biases (broadcast along M tile dimension)
    acc1 = acc1.to(OUT_DTYPE)
    acc2 = acc2.to(OUT_DTYPE)
    x1_bias = acc1 + b_bias[None, :]
    x2_bias = acc2 + c_bias[None, :]

    # swish(x1_bias) = x1_bias * sigmoid(beta * x1_bias)
    beta_x1 = beta * x1_bias
    sigmoid_x1 = 1.0 / (1.0 + tl.exp(-beta_x1))
    swish_x1 = x1_bias * sigmoid_x1
    # swiglu(x) = swish(x1_bias) * x2_bias
    swiglu_x = swish_x1 * x2_bias

    # Store result
    offs_mn = offs_m[:, None] * N + offs_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptr + offs_mn, swiglu_x, mask=mask_mn)
    return
