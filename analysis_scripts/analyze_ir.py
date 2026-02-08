#!/usr/bin/env python3
# ===- analyze_ir.py ---------------------------------------------------------===
#
# Script to analyze MLIR IR dumps from Hexagon-MLIR compilation
# This script helps visualize the optimization passes and IR transformations
#
# ===------------------------------------------------------------------------===

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_ir_sections(content: str) -> Dict[str, List[str]]:
    """Extract IR sections from dump file."""
    sections = {}
    current_section = None
    current_lines = []
    
    # Pattern to match pass names: "// IR Dump After PassName"
    pass_pattern = r'// IR Dump After (.*)'
    
    for line in content.split('\n'):
        match = re.match(pass_pattern, line)
        if match:
            # Save previous section
            if current_section:
                sections[current_section] = current_lines.copy()
            # Start new section
            current_section = match.group(1)
            current_lines = []
        else:
            if current_section:
                current_lines.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = current_lines
    
    return sections


def count_operations(ir_content: str) -> Dict[str, int]:
    """Count different types of operations in IR."""
    counts = {}
    
    # Common MLIR operation patterns
    patterns = {
        'linalg.generic': r'linalg\.generic',
        'linalg.matmul': r'linalg\.matmul',
        'linalg.fill': r'linalg\.fill',
        'vector.transfer_read': r'vector\.transfer_read',
        'vector.transfer_write': r'vector\.transfer_write',
        'scf.for': r'scf\.for',
        'async.execute': r'async\.execute',
        'memref.dma_start': r'memref\.dma_start',
        'memref.dma_wait': r'memref\.dma_wait',
        'hexkl.matmul': r'hexkl\.matmul',
        'arith.addf': r'arith\.addf',
        'arith.mulf': r'arith\.mulf',
    }
    
    for op_name, pattern in patterns.items():
        matches = len(re.findall(pattern, ir_content))
        if matches > 0:
            counts[op_name] = matches
    
    return counts


def analyze_optimizations(ir_content: str) -> Dict[str, any]:
    """Analyze optimizations present in IR."""
    analysis = {
        'has_vectorization': 'vector.transfer_read' in ir_content or 'vector.transfer_write' in ir_content,
        'has_multithreading': 'async.execute' in ir_content or 'async.create_group' in ir_content,
        'has_dma': 'memref.dma_start' in ir_content or 'memref.dma_wait' in ir_content,
        'has_tiling': 'scf.for' in ir_content and 'step' in ir_content,
        'has_hexkl': 'hexkl.matmul' in ir_content or 'hexkl' in ir_content,
        'has_fusion': 'linalg.generic' in ir_content and ir_content.count('linalg.generic') < 5,
        'operation_counts': count_operations(ir_content),
    }
    
    return analysis


def compare_ir_sections(before: str, after: str) -> Dict[str, any]:
    """Compare two IR sections to see what changed."""
    before_analysis = analyze_optimizations(before)
    after_analysis = analyze_optimizations(after)
    
    comparison = {
        'before': before_analysis,
        'after': after_analysis,
        'changes': {},
    }
    
    # Compare operation counts
    before_counts = before_analysis['operation_counts']
    after_counts = after_analysis['operation_counts']
    
    all_ops = set(before_counts.keys()) | set(after_counts.keys())
    for op in all_ops:
        before_count = before_counts.get(op, 0)
        after_count = after_counts.get(op, 0)
        if before_count != after_count:
            comparison['changes'][op] = {
                'before': before_count,
                'after': after_count,
                'delta': after_count - before_count,
            }
    
    # Compare optimization flags
    for key in ['has_vectorization', 'has_multithreading', 'has_dma', 'has_tiling', 'has_hexkl', 'has_fusion']:
        if before_analysis[key] != after_analysis[key]:
            comparison['changes'][key] = {
                'before': before_analysis[key],
                'after': after_analysis[key],
            }
    
    return comparison


def print_analysis_report(sections: Dict[str, List[str]], output_file: str = None):
    """Print a comprehensive analysis report."""
    output_lines = []
    
    output_lines.append("=" * 100)
    output_lines.append("Hexagon-MLIR IR Analysis Report")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    # Analyze each section
    section_names = list(sections.keys())
    output_lines.append(f"Found {len(section_names)} IR sections:")
    for i, name in enumerate(section_names, 1):
        output_lines.append(f"  {i}. {name}")
    output_lines.append("")
    
    # Analyze first section (initial IR)
    if section_names:
        first_section = '\n'.join(sections[section_names[0]])
        first_analysis = analyze_optimizations(first_section)
        
        output_lines.append("=" * 100)
        output_lines.append("Initial IR Analysis")
        output_lines.append("=" * 100)
        output_lines.append(f"Operation counts:")
        for op, count in first_analysis['operation_counts'].items():
            output_lines.append(f"  {op}: {count}")
        output_lines.append("")
        output_lines.append("Optimizations detected:")
        output_lines.append(f"  Vectorization: {first_analysis['has_vectorization']}")
        output_lines.append(f"  Multi-threading: {first_analysis['has_multithreading']}")
        output_lines.append(f"  DMA transfers: {first_analysis['has_dma']}")
        output_lines.append(f"  Tiling: {first_analysis['has_tiling']}")
        output_lines.append(f"  HexKL: {first_analysis['has_hexkl']}")
        output_lines.append(f"  Fusion: {first_analysis['has_fusion']}")
        output_lines.append("")
    
    # Analyze last section (final IR)
    if len(section_names) > 1:
        last_section = '\n'.join(sections[section_names[-1]])
        last_analysis = analyze_optimizations(last_section)
        
        output_lines.append("=" * 100)
        output_lines.append("Final IR Analysis")
        output_lines.append("=" * 100)
        output_lines.append(f"Operation counts:")
        for op, count in last_analysis['operation_counts'].items():
            output_lines.append(f"  {op}: {count}")
        output_lines.append("")
        output_lines.append("Optimizations detected:")
        output_lines.append(f"  Vectorization: {last_analysis['has_vectorization']}")
        output_lines.append(f"  Multi-threading: {last_analysis['has_multithreading']}")
        output_lines.append(f"  DMA transfers: {last_analysis['has_dma']}")
        output_lines.append(f"  Tiling: {last_analysis['has_tiling']}")
        output_lines.append(f"  HexKL: {last_analysis['has_hexkl']}")
        output_lines.append(f"  Fusion: {last_analysis['has_fusion']}")
        output_lines.append("")
        
        # Compare first and last
        comparison = compare_ir_sections(first_section, last_section)
        
        output_lines.append("=" * 100)
        output_lines.append("Optimization Changes")
        output_lines.append("=" * 100)
        
        if comparison['changes']:
            for key, change in comparison['changes'].items():
                if 'delta' in change:
                    output_lines.append(f"{key}: {change['before']} → {change['after']} (Δ{change['delta']:+d})")
                else:
                    output_lines.append(f"{key}: {change['before']} → {change['after']}")
        else:
            output_lines.append("No significant changes detected.")
        output_lines.append("")
    
    # Print report
    report = '\n'.join(output_lines)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_ir.py <ir_dump_file> [output_report.txt]")
        print("\nExample:")
        print("  python analyze_ir.py kernel_ir_dump.txt analysis_report.txt")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Reading IR dump from: {input_file}")
    content = input_file.read_text()
    
    sections = extract_ir_sections(content)
    
    if not sections:
        print("Warning: No IR sections found. The file might not be a valid IR dump.")
        print("Make sure to run with MLIR_ENABLE_DUMP=1")
        sys.exit(1)
    
    print_analysis_report(sections, output_file)


if __name__ == "__main__":
    main()
