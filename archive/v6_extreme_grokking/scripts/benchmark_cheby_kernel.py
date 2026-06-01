import torch
import triton
import modal
from scripts.cheby_triton import cheby_scan_triton, get_cheby_matrix

app = modal.App("cheby-triton-benchmark")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "triton", "pandas", "matplotlib")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],  # Argument names to use as an x-axis for the plot.
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'pytorch'],  # Possible values for `line_arg`.
        line_names=['Triton Kernel', 'PyTorch For-Loop'],  # Label name for the lines.
        styles=[('blue', '-'), ('red', '--')],  # Line styles.
        ylabel='Execution Time (ms)',  # Label name for the y-axis.
        plot_name='cheby-scan-performance',  # Name for the plot.
        args={'B': 64, 'D': 128},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(B, T, D, provider):
    device = "cuda"
    X = torch.randn(B, T, D, device=device, dtype=torch.float16, requires_grad=True)
    A = get_cheby_matrix(D, device=device, dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        def pytorch_scan():
            H_out = torch.zeros_like(X)
            h = torch.zeros(B, D, device=X.device)
            for t in range(T):
                h_new = torch.zeros_like(h)
                if D > 1: h_new[:, 1] += h[:, 0]
                if D > 2:
                    h_new[:, 0:-2] += 0.5 * h[:, 1:-1]
                    h_new[:, 2:]   += 0.5 * h[:, 1:-1]
                if D > 1: h_new[:, -2] += 0.5 * h[:, -1]
                h = h_new + X[:, t, :]
                H_out[:, t, :] = h
            return H_out
        
        if T > 1024:
            return 0.0, 0.0, 0.0
            
        ms, min_ms, max_ms = triton.testing.do_bench(pytorch_scan, quantiles=quantiles)
        
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cheby_scan_triton(X, A), quantiles=quantiles)
        
    return ms, max_ms, min_ms

@app.function(image=image, gpu="H100")
def run_benchmark():
    print("Running Triton Benchmark Sweep on H100...")
    benchmark.run(print_data=True, show_plots=False)
    print("Benchmark complete!")

@app.local_entrypoint()
def main():
    run_benchmark.remote()
