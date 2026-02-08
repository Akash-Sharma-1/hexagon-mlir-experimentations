# ===- reduction_kernel.py ----------------------------------------------------===
#
# Example: Optimized Reduction Kernel
# This kernel showcases:
# - Various reduction operations (sum, max, min)
# - Reduction optimizations and vectorization
# - Multi-threading for parallel reductions
# - Efficient reduction patterns for HVX
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# Test parameters
NUM_ELEMENTS = 131072  # Large size to showcase tiling
ATOL = 1e-5


@triton.jit
def reduction_sum_kernel(
    x_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
):
    """
    Sum reduction kernel.
    Computes: output = sum(x)
    
    This showcases:
    1. Reduction operations
    2. Tiling for large inputs
    3. Multi-threading
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum reduction
    block_sum = tl.sum(x, axis=0)
    
    # Store partial result
    tl.store(output_ptr + pid, block_sum)


@triton.jit
def reduction_max_kernel(
    x_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
):
    """
    Max reduction kernel.
    Computes: output = max(x)
    
    This showcases:
    1. Max reduction operations
    2. Efficient reduction patterns
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    
    # Compute max reduction
    block_max = tl.max(x, axis=0)
    
    # Store partial result
    tl.store(output_ptr + pid, block_max)


@triton.jit
def reduction_mean_kernel(
    x_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
):
    """
    Mean reduction kernel.
    Computes: output = mean(x)
    
    This showcases:
    1. Mean computation (sum + divide)
    2. Reduction optimizations
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum
    block_sum = tl.sum(x, axis=0)
    
    # Count valid elements in this block
    valid_count = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
    
    # Store sum and count for final reduction
    tl.store(output_ptr + pid * 2, block_sum)
    tl.store(output_ptr + pid * 2 + 1, valid_count)


def test_reduction_sum():
    """Test sum reduction kernel."""
    print("=" * 100)
    print("Testing Sum Reduction Kernel")
    print("=" * 100)
    
    x = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
    num_threads = 4
    block_size = NUM_ELEMENTS // num_threads
    
    # Partial results
    partial_sums = torch.zeros(num_threads, dtype=torch.float32)
    
    print(f"Input size: {NUM_ELEMENTS}")
    print(f"Number of threads: {num_threads}")
    print(f"Block size per thread: {block_size}")
    print("-" * 100)
    
    # Run kernel
    reduction_sum_kernel[(num_threads,)](
        x,
        partial_sums,
        BLOCK_SIZE=block_size,
        NUM_ELEMENTS=NUM_ELEMENTS,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    
    # Final reduction (can be done on host or in another kernel)
    result = partial_sums.sum().item()
    reference = x.sum().item()
    
    assert abs(result - reference) < ATOL * NUM_ELEMENTS, \
        f"Sum mismatch! Got {result}, expected {reference}"
    
    print(f"    ✓ Test passed!")
    print(f"    Result: {result:.6f}, Reference: {reference:.6f}")
    print(f"    Difference: {abs(result - reference):.6f}")


def test_reduction_max():
    """Test max reduction kernel."""
    print("\n" + "=" * 100)
    print("Testing Max Reduction Kernel")
    print("=" * 100)
    
    x = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
    num_threads = 4
    block_size = NUM_ELEMENTS // num_threads
    
    # Partial results
    partial_maxs = torch.zeros(num_threads, dtype=torch.float32)
    
    print(f"Input size: {NUM_ELEMENTS}")
    print(f"Number of threads: {num_threads}")
    print("-" * 100)
    
    # Run kernel
    reduction_max_kernel[(num_threads,)](
        x,
        partial_maxs,
        BLOCK_SIZE=block_size,
        NUM_ELEMENTS=NUM_ELEMENTS,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    
    # Final reduction
    result = partial_maxs.max().item()
    reference = x.max().item()
    
    assert abs(result - reference) < ATOL, \
        f"Max mismatch! Got {result}, expected {reference}"
    
    print(f"    ✓ Test passed!")
    print(f"    Result: {result:.6f}, Reference: {reference:.6f}")


def test_reduction_mean():
    """Test mean reduction kernel."""
    print("\n" + "=" * 100)
    print("Testing Mean Reduction Kernel")
    print("=" * 100)
    
    x = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
    num_threads = 4
    block_size = NUM_ELEMENTS // num_threads
    
    # Store sum and count
    partial_results = torch.zeros(num_threads * 2, dtype=torch.float32)
    
    print(f"Input size: {NUM_ELEMENTS}")
    print(f"Number of threads: {num_threads}")
    print("-" * 100)
    
    # Run kernel
    reduction_mean_kernel[(num_threads,)](
        x,
        partial_results,
        BLOCK_SIZE=block_size,
        NUM_ELEMENTS=NUM_ELEMENTS,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    
    # Final reduction
    total_sum = partial_results[::2].sum().item()
    total_count = partial_results[1::2].sum().item()
    result = total_sum / total_count if total_count > 0 else 0.0
    reference = x.mean().item()
    
    assert abs(result - reference) < ATOL, \
        f"Mean mismatch! Got {result}, expected {reference}"
    
    print(f"    ✓ Test passed!")
    print(f"    Result: {result:.6f}, Reference: {reference:.6f}")
    print(f"    Difference: {abs(result - reference):.6f}")


def test_all_reductions():
    """Run all reduction tests."""
    test_reduction_sum()
    test_reduction_max()
    test_reduction_mean()
    
    print("\n" + "=" * 100)
    print("Optimization Analysis:")
    print("=" * 100)
    print("""
This kernel demonstrates several Hexagon-MLIR optimizations:

1. **Reduction Optimizations**:
   - Efficient sum/max/min reduction patterns
   - Vectorized reduction operations
   - Tree reduction patterns for multi-threading

2. **Tiling for Large Inputs**:
   - Large inputs split into tiles
   - Each thread processes a tile
   - Partial results combined

3. **Multi-threading**:
   - Parallel reduction across threads
   - Efficient synchronization patterns
   - Load balancing

4. **VTCM Utilization**:
   - Tiles fit in VTCM for fast access
   - Reduces memory bandwidth

5. **Vectorization**:
   - Reduction operations vectorized for HVX
   - Efficient SIMD patterns

To analyze the optimizations:
  - Set MLIR_ENABLE_DUMP=1 to see reduction lowering
  - Look for linalg.reduce operations
  - Check vectorization patterns
  - Inspect async.execute for multi-threading
    """)


if __name__ == "__main__":
    test_all_reductions()
