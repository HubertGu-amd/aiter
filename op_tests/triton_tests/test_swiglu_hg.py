import torch
import pytest
from aiter.ops.triton.swiglu_hg import swiglu
from aiter.ops.triton.utils.types import str_to_torch_dtype


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
	"M,K,N",
	[
		(1, 1, 1),
		(8, 16, 32),
		(32, 64, 64),
		(127, 31, 97),
		(256, 128, 256),
		(513, 37, 129), 
	],
)
@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 2.0])


def test_swiglu(M, K, N, dtype, beta):
	dtype = str_to_torch_dtype[dtype]
	device = "cuda"
	torch.manual_seed(0)

	x = torch.randn(M, 2 * K, device=device, dtype=dtype)
	w = torch.randn(K, N, device=device, dtype=dtype)
	v = torch.randn(K, N, device=device, dtype=dtype)
	b = torch.randn(N, device=device, dtype=dtype)
	c = torch.randn(N, device=device, dtype=dtype)

	# Triton implementation 
	y_triton = swiglu(x, w, v, b, c, beta=beta)

	# PyTorch implementation
	x1, x2 = x[:, :K], x[:, K:]
	proj1 = x1 @ w + b
	proj2 = x2 @ v + c
	swish1 = proj1 * torch.sigmoid(beta * proj1)
	y_torch = swish1 * proj2

	if dtype in (torch.float16, torch.bfloat16):
		atol, rtol = 1e-2, 1e-2
	else:
        # float32 typically can be tighter
		atol, rtol = 1e-5, 1e-5
	
	max_abs = (y_triton - y_torch).abs().max().item()
	print(f"[swiglu test] dtype={dtype} shape=({M},{N}) max_abs={max_abs:.3e} atol={atol} rtol={rtol}")

	torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
	print("Swiglu has been asserted")

test_swiglu(8192, 8192, 8192, "fp16", 1.5)
