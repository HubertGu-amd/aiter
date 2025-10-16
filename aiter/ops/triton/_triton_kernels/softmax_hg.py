import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    input_ptr,                  # [M, N]
    output_ptr,                 # [M, N]
    M, N,                       # matrix dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Initialize offsets and masks for m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    # Initialize accumulators (BLOCK_SIZE_M x 1) for both
    row_max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Get exponential sum and max for each row
    for col_start in tl.range(0, N, BLOCK_SIZE_N):
        offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
        offs_mn = offs_m[:, None] * N + offs_n[None, :]
        mask_n = offs_n < N
        mask_mn = mask_m[:, None] & mask_n[None, :]

        # Load x: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
        x = tl.load(input_ptr + offs_mn, mask=mask_mn)

        # Get max of the block
        x_max = tl.max(x, axis=1)
        new_max = tl.maximum(row_max, x_max)

        # Get sum of exponentials
        # new_sum = exp(old_x-new_max) + exp(x-new_max) for the block
        # exp(old_x-new_max) = exp(old_x-old_max+old_max-new_max) = exp(old_x-old_max)*exp(old_max-new_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max[:, None]), axis=1)
        row_max = new_max

    # Compute and write softmax output
    for col_start in tl.range(0, N, BLOCK_SIZE_N):
        offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
        offs_mn = offs_m[:, None] * N + offs_n[None, :]
        mask_n = offs_n < N
        mask_mn = mask_m[:, None] & mask_n[None, :]

        # Load x: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
        x = tl.load(input_ptr + offs_mn, mask=mask_mn)

        # Compute softmax
        softmax_x = tl.exp(x - row_max[:, None]) / row_sum[:, None]

        # Store result
        tl.store(output_ptr + offs_mn, softmax_x, mask=mask_mn)
    return
