# Examples Summary

This document provides an overview of all the examples and tools created to showcase Hexagon-MLIR optimizations.

## Structure

```
examples/
├── kernels/                          # Example Triton kernels
│   ├── fused_matmul_relu.py         # MatMul + ReLU fusion
│   ├── layer_norm_optimized.py      # Layer normalization
│   ├── reduction_kernel.py          # Reduction operations
│   └── flash_attention_simple.py    # Simplified attention
├── analysis_scripts/                 # Analysis tools
│   ├── analyze_ir.py                # IR analysis script
│   ├── run_with_analysis.sh         # Automated analysis runner
│   └── compare_optimizations.py     # Optimization comparison
├── README.md                         # Comprehensive documentation
├── QUICKSTART.md                    # Quick start guide
└── SUMMARY.md                       # This file
```

## Kernels Overview

### 1. Fused MatMul + ReLU (`fused_matmul_relu.py`)

**Purpose**: Demonstrate matmul optimizations and operation fusion

**Key Optimizations**:
- MatMul tiling and vectorization
- Element-wise operation fusion
- Multi-threading
- VTCM utilization
- Optional HexKL support

**Best For**: Understanding matmul optimizations and fusion patterns

### 2. Layer Normalization (`layer_norm_optimized.py`)

**Purpose**: Showcase reduction optimizations

**Key Optimizations**:
- Sum reduction for mean/variance
- Row-wise parallelism
- Vectorized element-wise ops
- Efficient memory access

**Best For**: Learning about reduction operations and vectorization

### 3. Reduction Kernels (`reduction_kernel.py`)

**Purpose**: Demonstrate various reduction types

**Key Optimizations**:
- Sum, max, mean reductions
- Tiling for large inputs
- Multi-threaded reductions
- Tree reduction patterns

**Best For**: Understanding reduction optimizations

### 4. Flash Attention (`flash_attention_simple.py`)

**Purpose**: Show complex operation fusion

**Key Optimizations**:
- Fused matmul operations
- Softmax optimizations
- Multi-stage fusion
- Memory access patterns

**Best For**: Seeing advanced fusion patterns

## Analysis Tools

### `analyze_ir.py`

**Purpose**: Analyze MLIR IR dumps

**Features**:
- Extract IR sections
- Count operations
- Detect optimizations
- Compare before/after
- Generate reports

**Usage**:
```bash
python3 analyze_ir.py <ir_dump> [report.txt]
```

### `run_with_analysis.sh`

**Purpose**: Automated kernel execution and analysis

**Features**:
- Sets up environment
- Captures IR dumps
- Runs analysis automatically
- Organizes output

**Usage**:
```bash
bash run_with_analysis.sh <kernel.py> [output_dir]
```

### `compare_optimizations.py`

**Purpose**: Compare different optimization configurations

**Features**:
- Run kernels with different flags
- Compare performance
- Analyze differences

**Usage**:
```bash
python3 compare_optimizations.py <kernel.py>
```

## Optimization Passes Demonstrated

### Multi-threading
- **What**: Parallel execution across HVX threads
- **IR Pattern**: `async.execute`, `async.create_group`
- **Kernels**: All kernels support this

### VTCM Tiling
- **What**: Tile operations to fit in TCM
- **IR Pattern**: `scf.for` with step sizes, tiled `linalg` ops
- **Kernels**: All kernels support this

### Vectorization
- **What**: HVX vector operations
- **IR Pattern**: `vector.transfer_read/write`, `vector<32xf32>`
- **Kernels**: All kernels demonstrate this

### DMA Transfers
- **What**: Efficient DDR ↔ TCM transfers
- **IR Pattern**: `memref.dma_start`, `memref.dma_wait`
- **Kernels**: All kernels support this

### Operation Fusion
- **What**: Combine multiple operations
- **IR Pattern**: Fewer `linalg.generic` ops
- **Kernels**: `fused_matmul_relu.py`, `flash_attention_simple.py`

### Reduction Optimizations
- **What**: Efficient reduction patterns
- **IR Pattern**: Optimized `linalg.reduce` operations
- **Kernels**: `layer_norm_optimized.py`, `reduction_kernel.py`

### HexKL MatMul
- **What**: Hexagon Kernel Library for matmul
- **IR Pattern**: `hexkl.matmul` operations
- **Kernels**: `fused_matmul_relu.py` (optional)

## Quick Reference

### Running Examples

```bash
# Basic run
python3 kernels/<kernel_name>.py

# With IR dumping
MLIR_ENABLE_DUMP=1 python3 kernels/<kernel_name>.py

# With analysis
bash analysis_scripts/run_with_analysis.sh kernels/<kernel_name>.py
```

### Analyzing IR

```bash
# Manual analysis
python3 analysis_scripts/analyze_ir.py <ir_dump> report.txt

# Using linalg-hexagon-opt
linalg-hexagon-opt input.mlir --hexagon-tiling --mlir-print-ir-after-all
```

### Environment Variables

```bash
export TRITON_ALWAYS_COMPILE=1      # Always recompile
export MLIR_ENABLE_DUMP=1           # Dump MLIR IR
export LLVM_IR_ENABLE_DUMP=1        # Dump LLVM IR
```

## Expected Outcomes

### IR Evolution

1. **Initial**: High-level `linalg` operations
2. **After Tiling**: `scf.for` loops, tiled operations
3. **After Vectorization**: `vector` operations
4. **After Multi-threading**: `async` operations
5. **After DMA**: `memref.dma_*` operations
6. **Final**: LLVM IR with HVX intrinsics

### Performance Improvements

- **Multi-threading**: 2-4x speedup (depending on workload)
- **VTCM Tiling**: Reduced memory latency
- **Vectorization**: Better SIMD utilization
- **DMA**: Overlapped computation/communication
- **Fusion**: Reduced memory traffic

## Next Steps

1. **Run Examples**: Start with `fused_matmul_relu.py`
2. **Analyze IR**: Use `analyze_ir.py` to understand transformations
3. **Experiment**: Modify kernels and see how IR changes
4. **Compare**: Use different optimization flags
5. **Deep Dive**: Use `linalg-hexagon-opt` for specific passes

## Resources

- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [Hexagon-MLIR Docs](../hexagon-mlir/docs/) - Official docs

## Contributing

When adding new examples:
1. Follow existing kernel structure
2. Include comprehensive docstrings
3. Add test functions
4. Document optimizations
5. Update this summary
