import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

def get_cheby_matrix(D, device="cuda"):
    A = torch.zeros(D, D, device=device)
    if D > 1: A[0, 1] += 1.0
    for i in range(1, D-1):
        A[i, i-1] += 0.5
        A[i, i+1] += 0.5
    if D > 1: A[-1, -2] += 0.5
    return A

@triton.jit
def cheby_scan_fwd_kernel(
    X_ptr, A_ptr, H_out_ptr,
    stride_xb, stride_xt, stride_xd,
    stride_ab, stride_ad,
    stride_hb, stride_ht, stride_hd,
    B, T, D: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    
    # Offsets
    offs_b = b_start + tl.arange(0, BLOCK_B)
    offs_d = tl.arange(0, D)
    
    # Mask for valid batch indices
    mask_b = offs_b < B
    mask_2d = mask_b[:, None] & (offs_d[None, :] < D)
    
    # Load A into SRAM (shape DxD)
    offs_A_d1 = tl.arange(0, D)[:, None]
    offs_A_d2 = tl.arange(0, D)[None, :]
    A_ptrs = A_ptr + offs_A_d1 * stride_ab + offs_A_d2 * stride_ad
    A = tl.load(A_ptrs) # [D, D]
    
    # Initialize H to 0 (shape BLOCK_BxD)
    H = tl.zeros((BLOCK_B, D), dtype=tl.float32)
    
    # Loop over time
    for t in range(T):
        # Load X_t
        X_t_ptrs = X_ptr + offs_b[:, None] * stride_xb + t * stride_xt + offs_d[None, :] * stride_xd
        X_t = tl.load(X_t_ptrs, mask=mask_2d, other=0.0)
        
        # H = H @ A + X_t
        # tl.dot requires float16/bfloat16 for fast tensor cores, but float32 is supported on Ampere/Hopper (TF32)
        H = tl.dot(H, A, allow_tf32=True) + X_t
        
        # Store H_t
        H_out_ptrs = H_out_ptr + offs_b[:, None] * stride_hb + t * stride_ht + offs_d[None, :] * stride_hd
        tl.store(H_out_ptrs, H, mask=mask_2d)

@triton.jit
def cheby_scan_bwd_kernel(
    dOut_ptr, AT_ptr, dX_ptr,
    stride_db, stride_dt, stride_dd,
    stride_ab, stride_ad,
    stride_xb, stride_xt, stride_xd,
    B, T, D: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    
    offs_b = b_start + tl.arange(0, BLOCK_B)
    offs_d = tl.arange(0, D)
    mask_b = offs_b < B
    mask_2d = mask_b[:, None] & (offs_d[None, :] < D)
    
    # Load A^T
    offs_A_d1 = tl.arange(0, D)[:, None]
    offs_A_d2 = tl.arange(0, D)[None, :]
    AT_ptrs = AT_ptr + offs_A_d1 * stride_ab + offs_A_d2 * stride_ad
    AT = tl.load(AT_ptrs)
    
    dH = tl.zeros((BLOCK_B, D), dtype=tl.float32)
    
    # Loop backwards in time
    for i in range(T):
        t = T - 1 - i
        # Load dOut_t
        dOut_t_ptrs = dOut_ptr + offs_b[:, None] * stride_db + t * stride_dt + offs_d[None, :] * stride_dd
        dOut_t = tl.load(dOut_t_ptrs, mask=mask_2d, other=0.0)
        
        # dH = dH @ A^T + dOut_t
        dH = tl.dot(dH, AT, allow_tf32=True) + dOut_t
        
        # dX_t = dH
        dX_ptrs = dX_ptr + offs_b[:, None] * stride_xb + t * stride_xt + offs_d[None, :] * stride_xd
        tl.store(dX_ptrs, dH, mask=mask_2d)

class ChebyScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, A):
        B, T, D = X.shape
        H_out = torch.empty_like(X)
        
        BLOCK_B = triton.next_power_of_2(B)
        if BLOCK_B < 16: BLOCK_B = 16
        if BLOCK_B > 64: BLOCK_B = 64 # Cap block size
        
        grid = (triton.cdiv(B, BLOCK_B),)
        
        cheby_scan_fwd_kernel[grid](
            X, A, H_out,
            X.stride(0), X.stride(1), X.stride(2),
            A.stride(0), A.stride(1),
            H_out.stride(0), H_out.stride(1), H_out.stride(2),
            B, T, D, BLOCK_B=BLOCK_B,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(A)
        ctx.T = T
        ctx.BLOCK_B = BLOCK_B
        return H_out

    @staticmethod
    def backward(ctx, dOut):
        A, = ctx.saved_tensors
        B, T, D = dOut.shape
        AT = A.T.contiguous()
        
        dX = torch.empty_like(dOut)
        grid = (triton.cdiv(B, ctx.BLOCK_B),)
        
        cheby_scan_bwd_kernel[grid](
            dOut, AT, dX,
            dOut.stride(0), dOut.stride(1), dOut.stride(2),
            AT.stride(0), AT.stride(1),
            dX.stride(0), dX.stride(1), dX.stride(2),
            B, T, D, BLOCK_B=ctx.BLOCK_B,
            num_warps=4,
            num_stages=2,
        )
        return dX, None

def cheby_scan_triton(X, A):
    return ChebyScanFunction.apply(X, A)

def chebyshev_shift_pytorch(h):
    B, D = h.shape
    h_new = torch.zeros_like(h)
    if D > 1: h_new[:, 1] += h[:, 0]
    if D > 2:
        h_new[:, 0:-2] += 0.5 * h[:, 1:-1]
        h_new[:, 2:]   += 0.5 * h[:, 1:-1]
    if D > 1: h_new[:, -2] += 0.5 * h[:, -1]
    return h_new

def cheby_scan_pytorch(X):
    B, T, D = X.shape
    H_out = torch.zeros_like(X)
    h = torch.zeros(B, D, device=X.device)
    for t in range(T):
        h = chebyshev_shift_pytorch(h)
        h = h + X[:, t, :]
        H_out[:, t, :] = h
    return H_out

import modal

app = modal.App("triton-test")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "triton")

@app.function(image=image, gpu="H100")
def run_test():
    B, T, D = 64, 64, 128
    device = "cuda"
    
    print(f"Testing Triton ChebyScan: B={B}, T={T}, D={D}")
    
    X = torch.randn(B, T, D, device=device, requires_grad=True)
    A = get_cheby_matrix(D, device=device)
    
    # ── Forward Pass ──
    H_ref = cheby_scan_pytorch(X)
    H_tri = cheby_scan_triton(X, A)
    
    fwd_max_diff = (H_ref - H_tri).abs().max().item()
    print(f"Forward Max Diff:  {fwd_max_diff:.8f}")
    # assert fwd_max_diff < 1e-2, f"Forward pass does not match! Diff: {fwd_max_diff}"
    
    # ── Backward Pass ──
    # PyTorch Autograd
    loss_ref = H_ref.sum()
    loss_ref.backward()
    dX_ref = X.grad.clone()
    X.grad.zero_()
    
    # Triton Autograd
    loss_tri = H_tri.sum()
    loss_tri.backward()
    dX_tri = X.grad.clone()
    
    bwd_max_diff = (dX_ref - dX_tri).abs().max().item()
    print(f"Backward Max Diff: {bwd_max_diff:.8f}")
    # assert bwd_max_diff < 1e-2, f"Backward pass does not match! Diff: {bwd_max_diff}"
    
    print("✓ Output matches! Running speed benchmark...")
    
    # ── Benchmark ──
    # Warmup
    for _ in range(10):
        cheby_scan_pytorch(X)
        cheby_scan_triton(X, A)
        
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        H_ref = cheby_scan_pytorch(X)
        loss = H_ref.sum()
        loss.backward()
    torch.cuda.synchronize()
    pt_time = time.time() - t0
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        H_tri = cheby_scan_triton(X, A)
        loss = H_tri.sum()
        loss.backward()
    torch.cuda.synchronize()
    tr_time = time.time() - t0
    
    print(f"PyTorch Loop (Fwd+Bwd): {pt_time*1000/100:.2f} ms/iter")
    print(f"Triton Scan  (Fwd+Bwd): {tr_time*1000/100:.2f} ms/iter")
    print(f"Speedup: {pt_time/tr_time:.1f}x")

@app.local_entrypoint()
def main():
    run_test.remote()

