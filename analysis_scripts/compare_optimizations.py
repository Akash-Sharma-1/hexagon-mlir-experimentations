#!/usr/bin/env python3
# ===- compare_optimizations.py ----------------------------------------------===
#
# Script to compare kernel performance and IR with different optimization flags
# This helps understand the impact of each optimization pass
#
# ===------------------------------------------------------------------------===

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_kernel_with_flags(kernel_file: str, flags: Dict[str, bool], capture_ir: bool = True) -> Tuple[str, float]:
    """Run a kernel with specific optimization flags."""
    env = {
        'TRITON_ALWAYS_COMPILE': '1',
    }
    
    if capture_ir:
        env['MLIR_ENABLE_DUMP'] = '1'
        env['LLVM_IR_ENABLE_DUMP'] = '1'
    
    # Build Python code to run kernel with flags
    flag_str = ', '.join([f"{k}={v}" for k, v in flags.items()])
    
    # Note: This is a simplified version. In practice, you'd modify the kernel
    # to accept these flags or create wrapper scripts.
    print(f"  Running with flags: {flag_str}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, kernel_file],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr}")
            return result.stdout + result.stderr, elapsed
        
        return result.stdout, elapsed
    except subprocess.TimeoutExpired:
        print("    Timeout!")
        return "", float('inf')


def extract_performance_metrics(output: str) -> Dict[str, float]:
    """Extract performance metrics from kernel output."""
    metrics = {}
    
    # Look for performance information
    import re
    
    # Pattern for "Perf: xxxx Units:us"
    perf_pattern = r'Perf:\s*([\d.]+)\s*Units:(\w+)'
    match = re.search(perf_pattern, output)
    if match:
        metrics['performance'] = float(match.group(1))
        metrics['units'] = match.group(2)
    
    # Pattern for test result
    result_pattern = r'Result:(Pass|Fail)'
    match = re.search(result_pattern, output)
    if match:
        metrics['result'] = match.group(1)
    
    return metrics


def compare_configurations(kernel_file: str, configs: List[Dict[str, Dict[str, bool]]]):
    """Compare different optimization configurations."""
    print("=" * 100)
    print("Optimization Configuration Comparison")
    print("=" * 100)
    print(f"Kernel: {kernel_file}")
    print()
    
    results = []
    
    for i, config in enumerate(configs, 1):
        config_name = config.get('name', f'Config {i}')
        flags = config.get('flags', {})
        
        print(f"[{i}/{len(configs)}] {config_name}")
        print("-" * 100)
        
        output, elapsed = run_kernel_with_flags(kernel_file, flags, capture_ir=True)
        metrics = extract_performance_metrics(output)
        
        result = {
            'name': config_name,
            'flags': flags,
            'elapsed_time': elapsed,
            'metrics': metrics,
            'output': output,
        }
        results.append(result)
        
        if metrics:
            print(f"  Performance: {metrics.get('performance', 'N/A')} {metrics.get('units', '')}")
            print(f"  Result: {metrics.get('result', 'N/A')}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print()
    
    # Print comparison summary
    print("=" * 100)
    print("Comparison Summary")
    print("=" * 100)
    
    for result in results:
        print(f"{result['name']}:")
        print(f"  Flags: {result['flags']}")
        if result['metrics']:
            print(f"  Performance: {result['metrics'].get('performance', 'N/A')} {result['metrics'].get('units', '')}")
        print(f"  Elapsed: {result['elapsed_time']:.2f}s")
        print()
    
    return results


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python compare_optimizations.py <kernel_file.py>")
        print("\nExample:")
        print("  python compare_optimizations.py ../kernels/fused_matmul_relu.py")
        sys.exit(1)
    
    kernel_file = Path(sys.argv[1])
    
    if not kernel_file.exists():
        print(f"Error: Kernel file not found: {kernel_file}")
        sys.exit(1)
    
    # Define configurations to compare
    configs = [
        {
            'name': 'No Optimizations',
            'flags': {
                'enableMultiThreading': False,
                'enableVTCMTiling': False,
                'enableConvertToHexagonmem': False,
                'enableHexagonmemCopyToDMA': False,
            }
        },
        {
            'name': 'Multi-threading Only',
            'flags': {
                'enableMultiThreading': True,
                'enableVTCMTiling': False,
                'enableConvertToHexagonmem': False,
                'enableHexagonmemCopyToDMA': False,
            }
        },
        {
            'name': 'VTCM Tiling Only',
            'flags': {
                'enableMultiThreading': False,
                'enableVTCMTiling': True,
                'enableConvertToHexagonmem': False,
                'enableHexagonmemCopyToDMA': False,
            }
        },
        {
            'name': 'All Optimizations',
            'flags': {
                'enableMultiThreading': True,
                'enableVTCMTiling': True,
                'enableConvertToHexagonmem': True,
                'enableHexagonmemCopyToDMA': True,
            }
        },
    ]
    
    print("Note: This script requires kernels to be modified to accept optimization flags.")
    print("For now, it demonstrates the comparison framework.")
    print()
    
    # Note: Actual comparison would require kernel modifications
    # This is a template for how to structure such comparisons
    compare_configurations(str(kernel_file), configs)


if __name__ == "__main__":
    main()
