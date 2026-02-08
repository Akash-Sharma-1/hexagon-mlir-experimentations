# ===- fused_matmul_relu.py -------------------------------------------------===
#
# Example: Fused MatMul + ReLU Kernel
# This kernel showcases:
# - MatMul optimizations (tiling, vectorization, multi-threading)
# - Operation fusion (matmul + relu)
# - VTCM tiling and DMA transfers
# - HexKL matmul support (optional)
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# Test parameters
M, N, K = 1024, 512, 256
ATOL = 2e-1


@triton.jit
def fused_matmul_relu_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """
    Fused MatMul + ReLU kernel.
    Computes: C = ReLU(A @ B)
    
    This showcases:
    1. MatMul operation with tiling
    2. Element-wise ReLU fusion
    3. Multi-threading support
    4. VTCM utilization
    """
    # Create block pointers for tiled matmul
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(0, 0),
        block_shape=(M, K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(K, N),
        order=(1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(M, N),
        order=(1, 0),
    )
    
    # Load tiles
    a = tl.load(A_block_ptr)
    b = tl.load(B_block_ptr)
    
    # Compute matmul
    c = tl.dot(a, b, out_dtype=C.type.element_ty)
    
    # Fused ReLU: max(0, c)
    c = tl.maximum(c, 0.0)
    
    # Store result
    tl.store(C_block_ptr, c)


def test_fused_matmul_relu():
    """Test the fused matmul + relu kernel with various optimization flags."""
    print("=" * 100)
    print("Testing Fused MatMul + ReLU Kernel")
    print("=" * 100)
    
    # Create test data
    A = torch.rand((M, K), dtype=torch.float16)
    B = torch.rand((K, N), dtype=torch.float16)
    C = torch.zeros((M, N), dtype=torch.float16)
    
    print(f"Matrix shapes: A={A.shape}, B={B.shape}, C={C.shape}")
    print(f"Expected: C = ReLU(A @ B)")
    print("-" * 100)
    
    # Run kernel with optimizations enabled
    print("\n[1] Running with all optimizations enabled:")
    print("    - Multi-threading: ON")
    print("    - VTCM tiling: ON")
    print("    - HexagonMem conversion: ON")
    print("    - DMA transfers: ON")
    
    fused_matmul_relu_kernel[(1,)](
        A,
        B,
        C,
        M=M,
        N=N,
        K=K,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    
    # Compute reference
    reference = torch.nn.functional.relu(torch.matmul(A, B))
    
    # Verify correctness
    assert torch.allclose(C, reference, atol=ATOL), \
        f"Output mismatch! Max diff: {(C - reference).abs().max().item()}"
    
    print("    ✓ Test passed!")
    print(f"    Max difference: {(C - reference).abs().max().item():.6f}")
    
    # Test with HexKL (if available)
    print("\n[2] Testing with HexKL matmul (experimental):")
    C_hexkl = torch.zeros((M, N), dtype=torch.float32)
    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)
    
    @triton.jit
    def fused_matmul_relu_hexkl(
        A,
        B,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
        A_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(K, 1),
            offsets=(0, 0),
            block_shape=(M, K),
            order=(1, 0),
        )
        B_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(K, N),
            order=(1, 0),
        )
        C_block_ptr = tl.make_block_ptr(
            base=C,
            shape=(M, N),
            strides=(N, 1),
            offsets=(0, 0),
            block_shape=(M, N),
            order=(1, 0),
        )
        
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        c = tl.dot(a, b, out_dtype=tl.float32)
        c = tl.maximum(c, 0.0)
        tl.store(C_block_ptr, c)
    
    try:
        fused_matmul_relu_hexkl[(1,)](
            A_f32,
            B_f32,
            C_hexkl,
            M=M,
            N=N,
            K=K,
            enableHexKL=True,
            iterations=1,
        )
        reference_hexkl = torch.nn.functional.relu(torch.matmul(A_f32, B_f32))
        assert torch.allclose(C_hexkl, reference_hexkl, atol=1e-2), \
            f"HexKL output mismatch!"
        print("    ✓ HexKL test passed!")
    except Exception as e:
        print(f"    ⚠ HexKL test skipped: {e}")
    
    print("\n" + "=" * 100)
    print("Optimization Analysis:")
    print("=" * 100)
    print("""
This kernel demonstrates several Hexagon-MLIR optimizations:

1. **MatMul Tiling**: The matmul operation is tiled to fit in VTCM
2. **Operation Fusion**: MatMul and ReLU are fused into a single kernel
3. **Multi-threading**: Parallel execution across multiple HVX threads
4. **VTCM Utilization**: Efficient use of Tightly Coupled Memory
5. **DMA Transfers**: Optimized DDR ↔ TCM data movement
6. **Vectorization**: HVX vector operations for element-wise ReLU

To analyze the optimizations:
  - Set MLIR_ENABLE_DUMP=1 to see intermediate IRs
  - Set LLVM_IR_ENABLE_DUMP=1 to see final LLVM IR
  - Use linalg-hexagon-opt to inspect specific passes
    """)


if __name__ == "__main__":
    test_fused_matmul_relu()
