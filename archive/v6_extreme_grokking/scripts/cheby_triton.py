import torch
import triton
import triton.language as tl

def get_cheby_matrix(D, device="cuda", dtype=torch.float32):
    """
    Returns the Chebyshev transition matrix A.
    """
    A = torch.zeros(D, D, device=device, dtype=dtype)
    if D > 1: A[0, 1] += 1.0
    for i in range(1, D-1):
        A[i, i-1] += 0.5
        A[i, i+1] += 0.5
    if D > 1: A[-1, -2] += 0.5
    return A

@triton.jit
def cheby_scan_fwd_kernel(
    X_ptr, A_ptr, H_out_ptr,
    stride_xb, stride_xt,
    stride_ab,
    stride_hb, stride_ht,
    B, T, D: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    offs_b = b_start + tl.arange(0, BLOCK_B)
    offs_d = tl.arange(0, D)
    
    offs_A_d1 = tl.arange(0, D)[:, None]
    offs_A_d2 = tl.arange(0, D)[None, :]
    A_ptrs = A_ptr + offs_A_d1 * stride_ab + offs_A_d2
    A = tl.load(A_ptrs)
    
    # Initialize H in float32 and cast to A.dtype (FP16/BF16)
    H = tl.zeros((BLOCK_B, D), dtype=tl.float32).to(A.dtype)
    
    X_t_ptrs = X_ptr + offs_b[:, None] * stride_xb + offs_d[None, :]
    H_out_t_ptrs = H_out_ptr + offs_b[:, None] * stride_hb + offs_d[None, :]
    
    # Manually Unroll by 4 for extreme instruction pipelining
    for t in range(0, T, 4):
        # 1. Asynchronous Memory Fetch Block
        X_0 = tl.load(X_t_ptrs)
        X_t_ptrs += stride_xt
        
        X_1 = tl.load(X_t_ptrs)
        X_t_ptrs += stride_xt
        
        X_2 = tl.load(X_t_ptrs)
        X_t_ptrs += stride_xt
        
        X_3 = tl.load(X_t_ptrs)
        X_t_ptrs += stride_xt
        
        # 2. Sequential Math & Store Block
        # Step 0
        H = tl.dot(H, A, out_dtype=tl.float16) + X_0
        tl.store(H_out_t_ptrs, H)
        H_out_t_ptrs += stride_ht
        
        # Step 1
        H = tl.dot(H, A, out_dtype=tl.float16) + X_1
        tl.store(H_out_t_ptrs, H)
        H_out_t_ptrs += stride_ht
        
        # Step 2
        H = tl.dot(H, A, out_dtype=tl.float16) + X_2
        tl.store(H_out_t_ptrs, H)
        H_out_t_ptrs += stride_ht
        
        # Step 3
        H = tl.dot(H, A, out_dtype=tl.float16) + X_3
        tl.store(H_out_t_ptrs, H)
        H_out_t_ptrs += stride_ht

@triton.jit
def cheby_scan_bwd_kernel(
    dOut_ptr, AT_ptr, dX_ptr,
    stride_db, stride_dt,
    stride_ab,
    stride_xb, stride_xt,
    B, T, D: tl.constexpr, BLOCK_B: tl.constexpr
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    offs_b = b_start + tl.arange(0, BLOCK_B)
    offs_d = tl.arange(0, D)
    
    offs_A_d1 = tl.arange(0, D)[:, None]
    offs_A_d2 = tl.arange(0, D)[None, :]
    AT_ptrs = AT_ptr + offs_A_d1 * stride_ab + offs_A_d2
    AT = tl.load(AT_ptrs)
    
    dH = tl.zeros((BLOCK_B, D), dtype=tl.float32).to(AT.dtype)
    
    t_start = T - 1
    dOut_t_ptrs = dOut_ptr + offs_b[:, None] * stride_db + t_start * stride_dt + offs_d[None, :]
    dX_t_ptrs = dX_ptr + offs_b[:, None] * stride_xb + t_start * stride_xt + offs_d[None, :]
    
    for i in range(0, T, 4):
        dOut_0 = tl.load(dOut_t_ptrs)
        dOut_t_ptrs -= stride_dt
        dOut_1 = tl.load(dOut_t_ptrs)
        dOut_t_ptrs -= stride_dt
        dOut_2 = tl.load(dOut_t_ptrs)
        dOut_t_ptrs -= stride_dt
        dOut_3 = tl.load(dOut_t_ptrs)
        dOut_t_ptrs -= stride_dt
        
        dH = tl.dot(dH, AT, out_dtype=tl.float16) + dOut_0
        tl.store(dX_t_ptrs, dH)
        dX_t_ptrs -= stride_xt
        
        dH = tl.dot(dH, AT, out_dtype=tl.float16) + dOut_1
        tl.store(dX_t_ptrs, dH)
        dX_t_ptrs -= stride_xt
        
        dH = tl.dot(dH, AT, out_dtype=tl.float16) + dOut_2
        tl.store(dX_t_ptrs, dH)
        dX_t_ptrs -= stride_xt
        
        dH = tl.dot(dH, AT, out_dtype=tl.float16) + dOut_3
        tl.store(dX_t_ptrs, dH)
        dX_t_ptrs -= stride_xt

class ChebyScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, A):
        X = X.contiguous()
        A = A.contiguous()
        
        B, T, D = X.shape
        assert X.stride(2) == 1, "Feature dimension must be contiguous"
        
        pad_t = (4 - (T % 4)) % 4
        
        BLOCK_B = triton.next_power_of_2(B)
        if BLOCK_B > 128: BLOCK_B = 128
        elif BLOCK_B < 16: BLOCK_B = 16
            
        pad_b = (BLOCK_B - (B % BLOCK_B)) % BLOCK_B
        if pad_b > 0 or pad_t > 0:
            # Pad batch and time dimensions
            X_pad = torch.zeros(B + pad_b, T + pad_t, D, dtype=X.dtype, device=X.device)
            X_pad[:B, :T, :] = X
            X_in = X_pad
        else:
            X_in = X
            
        H_out_in = torch.empty_like(X_in)
        grid = (X_in.shape[0] // BLOCK_B,)
        
        cheby_scan_fwd_kernel[grid](
            X_in, A, H_out_in,
            X_in.stride(0), X_in.stride(1),
            A.stride(0),
            H_out_in.stride(0), H_out_in.stride(1),
            X_in.shape[0], T + pad_t, D, BLOCK_B=BLOCK_B,
            num_warps=4, num_stages=2, # Reduced warps for sync latency, num_stages lowered as we manually unrolled
        )
        
        if pad_b > 0 or pad_t > 0:
            H_out = H_out_in[:B, :T, :]
        else:
            H_out = H_out_in
            
        ctx.save_for_backward(A)
        ctx.T = T
        ctx.B = B
        ctx.BLOCK_B = BLOCK_B
        return H_out

    @staticmethod
    def backward(ctx, dOut):
        A, = ctx.saved_tensors
        B, T, D = dOut.shape
        dOut = dOut.contiguous()
        
        AT = A.T.contiguous()
        
        pad_b = (ctx.BLOCK_B - (B % ctx.BLOCK_B)) % ctx.BLOCK_B
        pad_t = (4 - (T % 4)) % 4
        
        if pad_b > 0 or pad_t > 0:
            dOut_pad = torch.zeros(B + pad_b, T + pad_t, D, dtype=dOut.dtype, device=dOut.device)
            dOut_pad[:B, :T, :] = dOut
            dOut_in = dOut_pad
        else:
            dOut_in = dOut
            
        dX_in = torch.empty_like(dOut_in)
        grid = (dOut_in.shape[0] // ctx.BLOCK_B,)
        
        cheby_scan_bwd_kernel[grid](
            dOut_in, AT, dX_in,
            dOut_in.stride(0), dOut_in.stride(1),
            AT.stride(0),
            dX_in.stride(0), dX_in.stride(1),
            dOut_in.shape[0], T + pad_t, D, BLOCK_B=ctx.BLOCK_B,
            num_warps=4, num_stages=2,
        )
        
        if pad_b > 0 or pad_t > 0:
            dX = dX_in[:B, :T, :]
        else:
            dX = dX_in
            
        return dX, None

def cheby_scan_triton(X, A):
    return ChebyScanFunction.apply(X, A)
