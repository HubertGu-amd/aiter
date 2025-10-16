import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.softmax_hg import _softmax_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def softmax(x):
    """Compute row-wise softmax using Triton kernel.

    Parameters
    ----------
    x : torch.Tensor (M, N)
        Input tensor.

    Returns
    -------
    torch.Tensor (M, N)
        Softmax result.
    """
    assert x.dim() == 2, "x must be 2D"

    _LOGGER.info(f"SOFTMAX: x={tuple(x.shape)}")
    M, N = x.shape

    MAX_FUSED_SIZE = 256
    BLOCK_SIZE_M = min(MAX_FUSED_SIZE, triton.next_power_of_2(M))
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    y = torch.empty((M, N), dtype=x.dtype, device="cuda")

    # 1D over tiles of M
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    _softmax_kernel[grid](
        x,
        y,
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return y
