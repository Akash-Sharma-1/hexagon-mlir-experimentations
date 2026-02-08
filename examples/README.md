# Hexagon-MLIR Optimization Examples

This directory contains example Triton kernels that showcase the optimization capabilities of the Hexagon-MLIR dialect. These examples demonstrate how various optimization passes transform high-level Triton kernels into optimized Hexagon NPU code.

## Overview

The examples in this directory are designed to:

1. **Showcase Optimization Passes**: Demonstrate how Hexagon-MLIR optimizes different types of operations
2. **Enable IR Analysis**: Provide tools to inspect intermediate representations (IR) at various compilation stages
3. **Compare Optimization Strategies**: Allow comparison between different optimization configurations

## Examples

### 1. Fused MatMul + ReLU (`kernels/fused_matmul_relu.py`)

**Optimizations Demonstrated:**
- MatMul tiling and vectorization
- Operation fusion (matmul + relu)
- Multi-threading support
- VTCM tiling and DMA transfers
- Optional HexKL matmul support

**Key Features:**
- Shows how matmul operations are optimized for Hexagon hardware
- Demonstrates fusion of element-wise operations with matmul
- Compares standard HVX-based lowering vs HexKL-based lowering

**Run:**
```bash
python3 kernels/fused_matmul_relu.py
```

### 2. Optimized Layer Normalization (`kernels/layer_norm_optimized.py`)

**Optimizations Demonstrated:**
- Reduction optimizations (sum for mean/variance)
- Row-wise parallelism
- Vectorized element-wise operations
- Efficient memory access patterns
- VTCM tiling for large tensors

**Key Features:**
- Shows reduction operation lowering and optimization
- Demonstrates multi-threading for row-wise parallelism
- Highlights vectorization of normalization operations

**Run:**
```bash
python3 kernels/layer_norm_optimized.py
```

### 3. Reduction Kernels (`kernels/reduction_kernel.py`)

**Optimizations Demonstrated:**
- Sum, max, and mean reduction optimizations
- Tiling for large inputs
- Multi-threading for parallel reductions
- Vectorized reduction operations
- Tree reduction patterns

**Key Features:**
- Multiple reduction types (sum, max, mean)
- Shows how large reductions are tiled and parallelized
- Demonstrates efficient reduction patterns for HVX

**Run:**
```bash
python3 kernels/reduction_kernel.py
```

### 4. Simplified Flash Attention (`kernels/flash_attention_simple.py`)

**Optimizations Demonstrated:**
- Fused matmul operations (QK^T, attention @ V)
- Softmax optimizations
- Operation fusion across attention stages
- Efficient memory access patterns
- Multi-threading across heads and batches

**Key Features:**
- Shows complex operation fusion
- Demonstrates attention computation optimization
- Highlights memory access pattern optimizations

**Run:**
```bash
python3 kernels/flash_attention_simple.py
```

## Analysis Tools

### IR Analysis Script (`analysis_scripts/analyze_ir.py`)

Analyzes MLIR IR dumps to identify optimizations and transformations.

**Usage:**
```bash
# Analyze an IR dump file
python3 analysis_scripts/analyze_ir.py <ir_dump_file> [output_report.txt]

# Example
python3 analysis_scripts/analyze_ir.py kernel_ir_dump.txt analysis_report.txt
```

**Features:**
- Extracts IR sections from dump files
- Counts operation types (linalg, vector, async, etc.)
- Detects optimization patterns (vectorization, multi-threading, DMA, etc.)
- Compares IR before and after optimization passes
- Generates comprehensive analysis reports

### Run with Analysis (`analysis_scripts/run_with_analysis.sh`)

Convenience script to run kernels and automatically capture IR dumps for analysis.

**Usage:**
```bash
# Run a kernel and capture IR dumps
bash analysis_scripts/run_with_analysis.sh <kernel_file.py> [output_dir]

# Examples
bash analysis_scripts/run_with_analysis.sh kernels/fused_matmul_relu.py
bash analysis_scripts/run_with_analysis.sh kernels/layer_norm_optimized.py ./my_analysis
```

**Features:**
- Automatically sets environment variables for IR dumping
- Captures MLIR and LLVM IR dumps
- Runs analysis scripts automatically
- Organizes output in timestamped directories

## Understanding the Optimizations

### Key Optimization Passes

1. **Multi-threading** (`enableMultiThreading=True`)
   - Parallel execution across multiple HVX threads
   - Uses `async.execute` operations
   - Lowers to LLVM coroutines

2. **VTCM Tiling** (`enableVTCMTiling=True`)
   - Tiles operations to fit in Tightly Coupled Memory
   - Reduces memory latency
   - Improves cache utilization

3. **HexagonMem Conversion** (`enableConvertToHexagonmem=True`)
   - Optimizes memory layout for Hexagon architecture
   - Enables Hexagon-specific memory optimizations

4. **DMA Transfers** (`enableHexagonmemCopyToDMA=True`)
   - Efficient DDR ↔ TCM data movement
   - Uses DMA engines for high-bandwidth transfers
   - Overlaps computation and communication

5. **HexKL MatMul** (`enableHexKL=True`)
   - Uses Hexagon Kernel Library for matrix operations
   - Leverages HMX units for matrix multiplication
   - Experimental feature

### IR Analysis Workflow

1. **Run kernel with IR dumping:**
   ```bash
   export MLIR_ENABLE_DUMP=1
   export LLVM_IR_ENABLE_DUMP=1
   python3 kernels/fused_matmul_relu.py
   ```

2. **Extract IR sections:**
   - IR dumps are typically written to `/tmp/` or current directory
   - Look for files with `.mlir` or `.ll` extensions
   - IR sections are also embedded in log output

3. **Analyze IR:**
   ```bash
   python3 analysis_scripts/analyze_ir.py ir_dump.txt report.txt
   ```

4. **Inspect specific passes:**
   ```bash
   # Use linalg-hexagon-opt to inspect specific passes
   linalg-hexagon-opt input.mlir --hexagon-tiling --mlir-print-ir-after-all
   ```

### What to Look For

**Initial IR (Triton → Linalg):**
- `linalg.generic` or `linalg.matmul` operations
- High-level tensor operations
- No vectorization or tiling yet

**After Tiling:**
- `scf.for` loops with step sizes
- Tiled `linalg` operations
- Memory operations for tile loading/storing

**After Vectorization:**
- `vector.transfer_read` and `vector.transfer_write`
- `vector<32xf32>` or similar vector types
- HVX-optimized vector operations

**After Multi-threading:**
- `async.execute` operations
- `async.create_group` and `async.add_to_group`
- Parallel execution patterns

**After DMA Optimization:**
- `memref.dma_start` and `memref.dma_wait`
- TCM memory space annotations
- Overlapped computation and communication

**Final LLVM IR:**
- LLVM function definitions
- HVX intrinsics
- Optimized memory access patterns

## Prerequisites

- Hexagon-MLIR compiler toolchain installed
- Triton with Hexagon backend configured
- Python 3.8+
- Access to Hexagon device (for execution) or `linalg-hexagon-opt` tool (for IR analysis)

## Environment Setup

```bash
# Set up Hexagon driver
export TRITON_ALWAYS_COMPILE=1

# Enable IR dumping for analysis
export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1

# Optional: Set output directory for IR dumps
export MLIR_DUMP_DIR=/tmp/mlir_dumps
```

## Troubleshooting

### No IR Dumps Generated

- Ensure `MLIR_ENABLE_DUMP=1` is set
- Check that the kernel actually compiles (set `TRITON_ALWAYS_COMPILE=1`)
- Look in `/tmp/` directory for dump files
- Check kernel output logs for embedded IR

### Analysis Script Fails

- Ensure Python 3.8+ is installed
- Check that the IR dump file format is correct
- Verify the file contains "// IR Dump After" markers

### Kernel Execution Fails

- Verify Hexagon device is accessible (if running on device)
- Check that all dependencies are installed
- Review kernel output logs for error messages
- Try running with optimizations disabled first

## Further Reading

- [Hexagon-MLIR Documentation](../hexagon-mlir/docs/)
- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)
- [MLIR Documentation](https://mlir.llvm.org/)

## Contributing

When adding new examples:

1. Follow the existing kernel structure
2. Include comprehensive docstrings explaining optimizations
3. Add test functions with reference implementations
4. Document expected optimization patterns
5. Update this README with example descriptions

## License

These examples follow the same license as the Hexagon-MLIR project.
