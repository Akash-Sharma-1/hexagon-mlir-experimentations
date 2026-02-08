# Quick Start Guide

This guide will help you quickly get started with analyzing Hexagon-MLIR optimizations using the example kernels.

## Prerequisites

1. **Hexagon-MLIR Setup**: Ensure the Hexagon-MLIR compiler is installed and configured
2. **Triton Backend**: Make sure the Hexagon backend is set up for Triton
3. **Python Environment**: Python 3.8+ with required dependencies

## Quick Start

### 1. Run a Simple Example

Start with the fused matmul + ReLU kernel:

```bash
cd examples
python3 kernels/fused_matmul_relu.py
```

This will:
- Compile the kernel
- Run it on the Hexagon device (if available)
- Show basic output

### 2. Enable IR Dumping

To see the intermediate representations:

```bash
export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export TRITON_ALWAYS_COMPILE=1

python3 kernels/fused_matmul_relu.py
```

IR dumps will be written to `/tmp/` or the current directory.

### 3. Use the Analysis Script

For automated analysis:

```bash
bash analysis_scripts/run_with_analysis.sh kernels/fused_matmul_relu.py
```

This will:
- Run the kernel with IR dumping enabled
- Capture all IR files
- Generate an analysis report
- Save everything in a timestamped directory

### 4. Analyze IR Manually

If you have an IR dump file:

```bash
python3 analysis_scripts/analyze_ir.py <ir_dump_file> analysis_report.txt
```

## Understanding the Output

### Kernel Output

When you run a kernel, you'll see:
- Test information (kernel name, result, performance)
- Any errors or warnings
- Reference comparison results

### IR Analysis Report

The analysis report shows:
- **Operation Counts**: Number of each operation type (linalg, vector, async, etc.)
- **Optimization Detection**: Which optimizations are present
- **Changes**: How the IR changed between passes

### Key Indicators

Look for these in the IR:

**Vectorization:**
- `vector.transfer_read` / `vector.transfer_write`
- `vector<32xf32>` types

**Multi-threading:**
- `async.execute`
- `async.create_group`

**DMA Transfers:**
- `memref.dma_start`
- `memref.dma_wait`

**Tiling:**
- `scf.for` loops with step sizes
- Tiled `linalg` operations

## Example Workflow

### Workflow 1: Basic Analysis

```bash
# 1. Run kernel with IR dumping
export MLIR_ENABLE_DUMP=1
python3 kernels/layer_norm_optimized.py > output.log 2>&1

# 2. Extract IR from log
grep -A 1000 "// IR Dump After" output.log > ir_dump.txt

# 3. Analyze
python3 analysis_scripts/analyze_ir.py ir_dump.txt report.txt

# 4. View report
cat report.txt
```

### Workflow 2: Automated Analysis

```bash
# Run with automated analysis
bash analysis_scripts/run_with_analysis.sh kernels/reduction_kernel.py

# View results
ls -lh analysis_output/reduction_kernel_*/
cat analysis_output/reduction_kernel_*/analysis_report.txt
```

### Workflow 3: Compare Optimizations

```bash
# Run with different optimization flags
# (Modify kernel to accept flags, then:)

# No optimizations
python3 kernels/fused_matmul_relu.py  # with flags disabled

# All optimizations
python3 kernels/fused_matmul_relu.py  # with flags enabled

# Compare IR dumps
diff ir_dump_no_opt.txt ir_dump_all_opt.txt
```

## Common Issues

### Issue: No IR Dumps Generated

**Solution:**
- Ensure `MLIR_ENABLE_DUMP=1` is set
- Check that `TRITON_ALWAYS_COMPILE=1` is set
- Look in `/tmp/` directory
- Check kernel output logs for embedded IR

### Issue: Kernel Fails to Compile

**Solution:**
- Verify Hexagon-MLIR is properly installed
- Check that all dependencies are available
- Review error messages in output
- Try with optimizations disabled first

### Issue: Analysis Script Errors

**Solution:**
- Ensure Python 3.8+ is installed
- Check IR dump file format
- Verify file contains "// IR Dump After" markers
- Check file permissions

## Next Steps

1. **Explore Examples**: Try all the example kernels
2. **Modify Kernels**: Experiment with different kernel configurations
3. **Analyze IR**: Deep dive into the IR transformations
4. **Compare Passes**: Use `linalg-hexagon-opt` to inspect specific passes

## Useful Commands

```bash
# Find linalg-hexagon-opt tool
find . -name linalg-hexagon-opt

# List available passes
linalg-hexagon-opt --help | grep hexagon

# Run specific passes
linalg-hexagon-opt input.mlir --hexagon-tiling --hexagon-vectorization --mlir-print-ir-after-all

# View IR after each pass
linalg-hexagon-opt input.mlir --linalg-to-llvm --mlir-print-ir-after-all
```

## Resources

- [Main README](README.md) - Comprehensive documentation
- [Hexagon-MLIR Docs](../hexagon-mlir/docs/) - Official documentation
- [Triton Docs](https://triton-lang.org/) - Triton language reference

## Getting Help

If you encounter issues:
1. Check the [README.md](README.md) troubleshooting section
2. Review kernel output logs
3. Verify environment setup
4. Check Hexagon-MLIR documentation
