import math
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu_hg import _swiglu_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Hardware shared memory limit (bytes) – ROCm MI300
_SM_LIMIT = 163840


def _choose_block_sizes(M, K, N, dtype_size):
    # Conservative candidate sets (largest first). Reduce if OOR.
    for bm in [triton.next_power_of_2(M), 256, 128, 64]:
        for bn in [triton.next_power_of_2(N), 256, 128, 64]:
            for bk in [triton.next_power_of_2(K), 64, 32, 16]:
                # Approx shared memory use (double-buffered if num_stages>1):
                # A_tile: bm * bk * dtype_size
                # W_tile: bk * bn * dtype_size
                shm = (bm * bk + bk * bn) * dtype_size
                # Allow room for staging (×2) if num_stages=2; keep a safety margin
                if 2 * shm < _SM_LIMIT * 0.9:
                    return bm, bn, bk, 2
                if shm < _SM_LIMIT * 0.9:
                    return bm, bn, bk, 1
    # Fallback smallest
    return 64, 64, 16, 1


def swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    beta: float = 1.0
):
    """Compute SwiGLU activation of a 2D input tensor.

    Key parameters:
        x (torch.Tensor): Input tensor of shape (M, 2K).
        w (torch.Tensor): Weight matrix W of shape (K, N).
        v (torch.Tensor): Weight matrix V of shape (K, N).
        b (torch.Tensor): Bias for W projection of shape (N,).
        c (torch.Tensor): Bias for V projection of shape (N,).
        beta (float): Scaling for the swish activation (default 1.0).

    Returns:
        torch.Tensor: A tensor of the shape (M, N), where SwiGLU activation has been
        applied with: 
            y = ( (x1 @ W + b) * sigmoid(beta * (x1 @ W + b)) * (x2 @ V + c) )
    """
    # Validation
    assert x.dtype == w.dtype == v.dtype == b.dtype == c.dtype, "Dtypes must match"
    M, twoK = x.shape
    K_w, N = w.shape
    K_v, N_v = v.shape
    if K_w != K_v or N != N_v:
        raise ValueError(
            f"w ({w.shape}) and v ({v.shape}) dimension mismatch for (K,N)"
        )
    if twoK % 2 != 0:
        raise ValueError(
            f"Input second dimension must be even (2K); got {twoK} which is not divisible by 2"
        )
    K = twoK // 2
    if K != K_w:
        raise ValueError(
            f"Derived K from x ({K}) does not match weight K ({K_w})"
        )

    if b is None:
        b = torch.zeros(N)
    if c is None:
        c = torch.zeros(N)
    if b.shape != (N,) or c.shape != (N,):
        raise ValueError(
            f"Bias shapes must be (N,) = ({N},); got b={tuple(b.shape)} c={tuple(c.shape)}"
        )

    _LOGGER.info(f"SWIGLU: x={(M, 2*K)} w={(K, N)} v={(K, N)} beta={beta}")

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages = _choose_block_sizes(M, K, N, x.element_size())
    waves_per_eu = 2
    num_warps = 8
    # Map torch dtype -> triton dtype for OUT_DTYPE
    if x.dtype == torch.float16:
        out_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif x.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")

    y = torch.empty((M, N), dtype=x.dtype, device="cuda")

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M), 
        triton.cdiv(N, BLOCK_SIZE_N)
    )

    _swiglu_kernel[grid](
        x,
        y,
        w,
        v,
        b,
        c,
        beta,
        M,
        K,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        OUT_DTYPE=out_dtype,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
