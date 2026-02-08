# ===- layer_norm_optimized.py ------------------------------------------------===
#
# Example: Optimized Layer Normalization Kernel
# This kernel showcases:
# - Reduction optimizations (mean, variance)
# - Vectorization of element-wise operations
# - Multi-threading for row-wise parallelism
# - VTCM tiling for large tensors
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# Test parameters
NUM_ROWS = 128
NUM_COLS = 8192
EPSILON = 1e-5
ATOL = 1e-5


@triton.jit
def layer_norm_optimized_kernel(
    x_ptr,
    y_ptr,
    weights_ptr,
    biases_ptr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_ROWS: tl.constexpr,
):
    """
    Optimized Layer Normalization kernel.
    Computes: y = (x - mean) / sqrt(var + eps) * weights + biases
    
    This showcases:
    1. Reduction operations (sum for mean/variance)
    2. Row-wise parallelism
    3. Vectorized element-wise operations
    4. Efficient memory access patterns
    """
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < NUM_COLS
    
    # Process multiple rows per thread
    for row in range(pid, NUM_ROWS, programs):
        # Compute row pointers
        x_row_ptr = x_ptr + row * NUM_COLS
        y_row_ptr = y_ptr + row * NUM_COLS
        
        # Load row data
        x = tl.load(x_row_ptr + block, mask=mask, other=0.0)
        
        # Compute mean (reduction)
        mean = tl.sum(x, axis=0) / NUM_COLS
        
        # Compute variance (reduction)
        x_centered = tl.where(mask, x - mean, 0.0)
        x_squared = x_centered * x_centered
        var = tl.sum(x_squared, axis=0) / NUM_COLS
        
        # Compute rstd (reciprocal standard deviation)
        rstd = 1.0 / tl.sqrt(var + EPSILON)
        
        # Normalize and apply affine transformation
        x_normalized = x_centered * rstd
        
        # Load weights and biases
        w = tl.load(weights_ptr + block, mask=mask, other=0.0)
        b = tl.load(biases_ptr + block, mask=mask, other=0.0)
        
        # Apply affine transformation
        y = x_normalized * w + b
        
        # Store result
        tl.store(y_row_ptr + block, y, mask=mask)


def test_layer_norm_optimized():
    """Test the optimized layer norm kernel."""
    print("=" * 100)
    print("Testing Optimized Layer Normalization Kernel")
    print("=" * 100)
    
    # Create test data
    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=torch.float32)
    output = torch.empty_like(x, dtype=torch.float32)
    weights = torch.rand(NUM_COLS, dtype=torch.float32)
    biases = torch.rand(NUM_COLS, dtype=torch.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Block size: {triton.next_power_of_2(NUM_COLS)}")
    print("-" * 100)
    
    # Test with single thread
    print("\n[1] Running with single thread (no multi-threading):")
    block_size = triton.next_power_of_2(NUM_COLS)
    layer_norm_optimized_kernel[(1,)](
        x,
        output,
        weights,
        biases,
        BLOCK_SIZE=block_size,
        EPSILON=EPSILON,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        enableMultiThreading=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False,
        enableHexagonmemCopyToDMA=False,
    )
    
    reference = torch.nn.functional.layer_norm(
        x, (NUM_COLS,), weight=weights, bias=biases, eps=EPSILON
    )
    
    assert torch.allclose(output, reference, atol=ATOL), \
        f"Output mismatch! Max diff: {(output - reference).abs().max().item()}"
    
    print("    ✓ Test passed!")
    print(f"    Max difference: {(output - reference).abs().max().item():.6f}")
    
    # Test with multi-threading and optimizations
    print("\n[2] Running with optimizations enabled:")
    print("    - Multi-threading: ON")
    print("    - VTCM tiling: ON")
    print("    - HexagonMem conversion: ON")
    print("    - DMA transfers: ON")
    
    output_opt = torch.empty_like(x, dtype=torch.float32)
    num_threads = 4
    layer_norm_optimized_kernel[(num_threads,)](
        x,
        output_opt,
        weights,
        biases,
        BLOCK_SIZE=block_size,
        EPSILON=EPSILON,
        NUM_ROWS=NUM_ROWS,
        NUM_COLS=NUM_COLS,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    
    assert torch.allclose(output_opt, reference, atol=ATOL), \
        f"Optimized output mismatch!"
    
    print("    ✓ Optimized test passed!")
    
    print("\n" + "=" * 100)
    print("Optimization Analysis:")
    print("=" * 100)
    print("""
This kernel demonstrates several Hexagon-MLIR optimizations:

1. **Reduction Optimizations**: 
   - Sum reductions for mean and variance computation
   - Efficient reduction patterns for HVX

2. **Row-wise Parallelism**:
   - Multiple rows processed in parallel across threads
   - Each thread handles multiple rows

3. **Vectorization**:
   - Element-wise operations (subtract, multiply, add) vectorized
   - HVX vector operations for efficient computation

4. **Memory Access Patterns**:
   - Coalesced memory access for row-wise operations
   - Efficient use of TCM for frequently accessed data

5. **VTCM Tiling**:
   - Large rows tiled to fit in VTCM
   - Reduces memory bandwidth requirements

To analyze the optimizations:
  - Set MLIR_ENABLE_DUMP=1 to see reduction lowering
  - Check for vector.transfer_read/write operations
  - Look for async.execute patterns (multi-threading)
  - Inspect DMA operations for VTCM transfers
    """)


if __name__ == "__main__":
    test_layer_norm_optimized()
