#!/bin/bash
# ===- run_with_analysis.sh ---------------------------------------------------===
#
# Script to run a Triton kernel and capture IR dumps for analysis
#
# ===------------------------------------------------------------------------===

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNELS_DIR="$EXAMPLES_DIR/kernels"
ANALYSIS_DIR="$EXAMPLES_DIR/analysis_scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <kernel_file.py> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 kernels/fused_matmul_relu.py"
    echo "  $0 kernels/layer_norm_optimized.py ./analysis_output"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

KERNEL_FILE="$1"
OUTPUT_DIR="${2:-./analysis_output}"

# Check if kernel file exists
if [ ! -f "$KERNEL_FILE" ]; then
    if [ -f "$KERNELS_DIR/$KERNEL_FILE" ]; then
        KERNEL_FILE="$KERNELS_DIR/$KERNEL_FILE"
    else
        echo -e "${RED}Error: Kernel file not found: $KERNEL_FILE${NC}"
        exit 1
    fi
fi

KERNEL_NAME=$(basename "$KERNEL_FILE" .py)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$OUTPUT_DIR/${KERNEL_NAME}_${TIMESTAMP}"

echo -e "${GREEN}Running kernel analysis for: $KERNEL_NAME${NC}"
echo -e "${YELLOW}Output directory: $OUTPUT_DIR${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for IR dumping
export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1

# Run the kernel and capture output
echo -e "${GREEN}Running kernel...${NC}"
python3 "$KERNEL_FILE" 2>&1 | tee "$OUTPUT_DIR/kernel_output.log"

# Check if MLIR dump was created (it might be in /tmp or current directory)
echo ""
echo -e "${GREEN}Searching for IR dump files...${NC}"

# Common locations for IR dumps
IR_DUMP_PATTERNS=(
    "/tmp/*${KERNEL_NAME}*.mlir"
    "/tmp/*${KERNEL_NAME}*.ll"
    "./*${KERNEL_NAME}*.mlir"
    "./*${KERNEL_NAME}*.ll"
    "$OUTPUT_DIR/*.mlir"
    "$OUTPUT_DIR/*.ll"
)

IR_FILES_FOUND=0

for pattern in "${IR_DUMP_PATTERNS[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}Found IR file: $file${NC}"
            cp "$file" "$OUTPUT_DIR/"
            IR_FILES_FOUND=1
        fi
    done
done

if [ $IR_FILES_FOUND -eq 0 ]; then
    echo -e "${YELLOW}Warning: No IR dump files found.${NC}"
    echo -e "${YELLOW}The IR might be embedded in the log file.${NC}"
fi

# Extract IR from log file if present
if [ -f "$OUTPUT_DIR/kernel_output.log" ]; then
    echo ""
    echo -e "${GREEN}Extracting IR sections from log file...${NC}"
    
    # Try to extract MLIR sections
    if grep -q "// IR Dump After" "$OUTPUT_DIR/kernel_output.log"; then
        echo "Found MLIR IR dumps in log file"
        # Extract IR sections
        awk '/\/\/ IR Dump After/,/^$/ {print}' "$OUTPUT_DIR/kernel_output.log" > "$OUTPUT_DIR/mlir_ir_dump.txt" || true
    fi
    
    # Try to extract LLVM IR sections
    if grep -q "define" "$OUTPUT_DIR/kernel_output.log"; then
        echo "Found LLVM IR in log file"
        grep -A 100 "define" "$OUTPUT_DIR/kernel_output.log" > "$OUTPUT_DIR/llvm_ir_dump.txt" || true
    fi
fi

# Run analysis script if IR dump found
if [ -f "$OUTPUT_DIR/mlir_ir_dump.txt" ] || [ -f "$OUTPUT_DIR/kernel_output.log" ]; then
    echo ""
    echo -e "${GREEN}Running IR analysis...${NC}"
    
    IR_FILE="$OUTPUT_DIR/mlir_ir_dump.txt"
    if [ ! -f "$IR_FILE" ]; then
        IR_FILE="$OUTPUT_DIR/kernel_output.log"
    fi
    
    if [ -f "$ANALYSIS_DIR/analyze_ir.py" ]; then
        python3 "$ANALYSIS_DIR/analyze_ir.py" "$IR_FILE" "$OUTPUT_DIR/analysis_report.txt"
    else
        echo -e "${YELLOW}Analysis script not found. Skipping analysis.${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Analysis complete!${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo ""
echo "To view the analysis report:"
echo "  cat $OUTPUT_DIR/analysis_report.txt"
echo ""
echo "To inspect IR dumps:"
echo "  ls -lh $OUTPUT_DIR/*.mlir $OUTPUT_DIR/*.ll 2>/dev/null || echo 'No IR files found'"
