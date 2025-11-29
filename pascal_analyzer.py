#!/usr/bin/env python3
"""
Pascal Unified Memory Profiler — Diagnostic Tool for cudaMallocManaged Performance

Automated profiler for analyzing Pascal GPU Unified Memory behavior under
demand-paged migration. Measures page-fault-driven PCIe transfer overhead and
demonstrates the effect of cudaMemPrefetchAsync on restoring DRAM-resident execution.

Measured Performance (GTX 1080):
    Naive UM:    ~8–9 GB/s     (~360–390 ms, ~31k page faults)
    Prefetch UM: ~241–242 GB/s (~13 ms, zero page faults)
    Speedup:     ~25–30x

Reference:
    https://stackoverflow.com/questions/39782746

Repository:
    https://github.com/parallelArchitect/pascal-um-benchmark

Author: Joe McLaren — Human–AI Collaborative Engineering
License: MIT
Version: 2.4.0

Tested On:
    - GPU: NVIDIA GeForce GTX 1080 (8 GB GDDR5X, SM 6.1)
    - Driver: 535.274.02
    - CUDA Toolkit: 12.0
    - Nsight Systems: 2025.5.1
    - Python: 3.10
    - OS: Ubuntu 24.04
    - Dependencies: reportlab, pycuda

Quick Start:
    pip install reportlab pycuda
    make
    python3 pascal_analyzer.py --pdf

Usage:
    python3 pascal_analyzer.py --pdf       # Generate PDF report
    python3 pascal_analyzer.py --json      # Machine-readable metrics
    python3 pascal_analyzer.py --nvprof    # Raw nvprof output
    python3 pascal_analyzer.py --nsys      # Nsight Systems trace
    python3 pascal_analyzer.py --diagnose  # Verify installation

Output Formats:
    PDF:  Hardware specs, kernel timing, page fault analysis
    JSON: Structured data (GPU info, timings, page faults)
    nsys: Timeline visualization (.nsys-rep)
"""


import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Third-party imports
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


# PDF Layout Constants
PDF_MARGIN = 0.4 * inch
PDF_TOP_MARGIN = 0.4 * inch
PDF_BOTTOM_MARGIN = 0.5 * inch

# Font Sizes
PDF_TITLE_SIZE = 12
PDF_HEADER_SIZE = 10
PDF_TABLE_SIZE = 9

# Spacing
PDF_SECTION_SPACER = 0.06 * inch
PDF_TABLE_PADDING = 2

# Profiling Configuration
NVPROF_TIMEOUT = 60
BINARY_TIMEOUT = 30
GPU_WARMUP_ITERATIONS = 3


# -----------------------------
# GPU HARDWARE DETECTION
# -----------------------------

def get_gpu_info_raw() -> Optional[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap,"
        "clocks.current.graphics,clocks.current.memory,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        line = result.stdout.strip()
        if not line:
            return None

        fields = line.split(",")
        if len(fields) < 6:
            return None

        gpu_name = fields[0].strip()
        vram_mb = float(fields[1].strip())
        compute_cap = fields[2].strip()
        gpu_clock = int(fields[3].strip())
        mem_clock = int(fields[4].strip())
        pstate = fields[5].strip()

        return {
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            "compute_cap": compute_cap,
            "gpu_clock_mhz": gpu_clock,
            "mem_clock_mhz": mem_clock,
            "pstate": pstate,
        }
    except Exception:
        return None


def get_gpu_info(diagnose: bool = False) -> Dict[str, Any]:
    gpu_info = get_gpu_info_raw()
    if gpu_info is None:
        if diagnose:
            print("[diagnose] nvidia-smi failed or empty")
        return {
            "gpu_name": "Unknown",
            "vram_mb": 0,
            "compute_cap": "Unknown",
            "gpu_clock_mhz": 0,
            "mem_clock_mhz": 0,
            "pstate": "Unknown",
            "cuda_cores": 0,
            "sm_count": 0,
        }

    compute_cap = gpu_info["compute_cap"]
    sm_count = 0
    cuda_cores = 0
    path_used = "nvidia-smi"

    if PYCUDA_AVAILABLE:
        try:
            dev = cuda.Device(0)
            sm_count = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            path_used = "PyCUDA driver API"
        except Exception:
            sm_count = 0
            path_used = "nvidia-smi (PyCUDA failed)"

    pascal_lookup = {"6.0": 64, "6.1": 128, "6.2": 128}
    if sm_count > 0:
        cores_per_sm = pascal_lookup.get(compute_cap, 64)
        cuda_cores = sm_count * cores_per_sm

    if diagnose:
        print("=== GPU DIAGNOSTICS ===")
        print(f"Detection path:     {path_used}")
        print(f"GPU:                {gpu_info['gpu_name']}")
        print(f"Compute capability: {compute_cap}")
        print(f"SM count:           {sm_count}")
        print(f"CUDA cores:         {cuda_cores}")
        print(f"VRAM (MB):          {gpu_info['vram_mb']:.0f}")
        print(f"Core clock (MHz):   {gpu_info['gpu_clock_mhz']}")
        print(f"Memory clock (MHz): {gpu_info['mem_clock_mhz']}")
        print(f"P-state:            {gpu_info['pstate']}")
        print("=======================")

    gpu_info["cuda_cores"] = cuda_cores
    gpu_info["sm_count"] = sm_count
    return gpu_info


# -----------------------------
# GPU WARMUP — NO P0 CHECK
# -----------------------------

def warmup_gpu(binary: str, size_mb: Optional[int] = None,
               max_attempts: int = GPU_WARMUP_ITERATIONS) -> Dict[str, Any]:
    import time
    print(f"Warming up GPU... (attempts: {max_attempts})")
    for i in range(1, max_attempts + 1):
        try:
            cmd = [binary]
            if size_mb:
                cmd += ["--mb", str(size_mb)]
            subprocess.run(cmd, capture_output=True, timeout=BINARY_TIMEOUT)
        except Exception:
            pass
        print(f"  Warmup {i}/{max_attempts}")
        time.sleep(0.3)
    print("Warmup complete.\n")
    return get_gpu_info()


# -----------------------------
# NVPROF RUNNER
# -----------------------------

def run_nvprof(binary: str, size_mb: Optional[int] = None) -> str:
    cmd = [
        "nvprof",
        "--unified-memory-profiling",
        "per-process-device",
        "--print-summary",
    ]
    if size_mb:
        cmd += [binary, "--mb", str(size_mb)]
    else:
        cmd += [binary]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=NVPROF_TIMEOUT)
        return result.stderr  # nvprof summary is on stderr
    except Exception as e:
        print(f"nvprof failed: {e}")
        sys.exit(1)


# -----------------------------
# PASCAL BANDWIDTH BINARY
# -----------------------------

def get_pascal_bandwidth(binary: str, size_mb: Optional[int] = None) -> Tuple[float, float]:
    try:
        cmd = [binary]
        if size_mb:
            cmd += ["--mb", str(size_mb)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=BINARY_TIMEOUT)

        naive_bw = 0.0
        prefetch_bw = 0.0

        for line in result.stdout.splitlines():
            parts = line.split()
            if "Naive:" in parts:
                idx = parts.index("Naive:") + 1
                try:
                    naive_bw = float(parts[idx])
                except Exception:
                    pass
            if "Prefetch:" in parts:
                idx = parts.index("Prefetch:") + 1
                try:
                    prefetch_bw = float(parts[idx])
                except Exception:
                    pass

        return naive_bw, prefetch_bw
    except Exception:
        return 0.0, 0.0


# -----------------------------
# NVPROF SUMMARY PARSER
# -----------------------------

def parse_summary(nvprof_output: str) -> Dict[str, Any]:
    """
    Parse nvprof unified memory summary.

       31965  ...  Host To Device
        5255  ...  Gpu page fault groups
    Total CPU Page faults: 6144
    """
    data = {
        "prefetch_ms": 0.0,
        "naive_ms": 0.0,
        "transfer_count": 0,
        "transfer_gb": 0.0,
        "transfer_time_ms": 0.0,
        "page_fault_groups": 0,
        "cpu_page_faults": 0,
    }

    for line in nvprof_output.splitlines():
        l = line.strip()
        ll = l.lower()

        # Kernel timing from GPU activities line
        # Format: "GPU activities:  100.00%  398.47ms  2  199.24ms  13.298ms  385.17ms  vec_add(...)"
        # We want: Min (prefetch) and Max (naive)
        if "gpu activities:" in ll and "vec_add" in ll:
            parts = l.split()
            # Find all millisecond values
            ms_vals = []
            for p in parts:
                if p.endswith("ms"):
                    try:
                        ms_vals.append(float(p.replace("ms", "")))
                    except:
                        pass
            
            # We expect: [total_ms, avg_ms, min_ms, max_ms]
            # Min = prefetch (fast), Max = naive (slow)
            if len(ms_vals) >= 4:
                data["prefetch_ms"] = ms_vals[2]  # Min time
                data["naive_ms"] = ms_vals[3]     # Max time

        # Host To Device line
        if "host to device" in ll:
            parts = l.split()
            try:
                data["transfer_count"] = int(parts[0].replace(",", ""))
            except Exception:
                pass

            for p in parts:
                if p.endswith("GB"):
                    try:
                        data["transfer_gb"] = float(p.replace("GB", ""))
                    except Exception:
                        pass
                if p.endswith("ms"):
                    try:
                        data["transfer_time_ms"] = float(p.replace("ms", ""))
                    except Exception:
                        pass

        # GPU page fault groups
        if "gpu page fault groups" in ll:
            try:
                parts = l.split()
                data["page_fault_groups"] = int(parts[0].replace(",", ""))
            except Exception:
                pass

        # CPU Page faults
        if "cpu page faults" in ll:
            try:
                val = l.split(":")[-1].strip().replace(",", "").split()[0]
                data["cpu_page_faults"] = int(val)
            except Exception:
                pass

    return data


# -----------------------------
# PDF REPORT GENERATOR
# -----------------------------

def generate_pdf_report(data: Dict[str, Any],
                        gpu_state: Optional[Dict[str, Any]],
                        filename: str) -> str:
    if not REPORTLAB_AVAILABLE:
        print("Error: reportlab not installed. Cannot generate PDF.")
        sys.exit(1)

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=PDF_TOP_MARGIN,
        bottomMargin=0.4 * inch,
        title="Pascal Unified Memory Analysis",
        author="Joe McLaren - Human-AI Collaborative Engineering",
        subject="CUDA GPU Performance Profiling Report"
    )

    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=PDF_TITLE_SIZE,
        textColor=colors.HexColor("#1A1A1A"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    story.append(Paragraph("PASCAL UNIFIED MEMORY ANALYSIS", title_style))
    story.append(Spacer(1, PDF_SECTION_SPACER))

    # Section header
    header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=PDF_HEADER_SIZE,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=3,
    )

    # GPU HARDWARE STATE
    if gpu_state:
        story.append(Paragraph("GPU HARDWARE STATE", header_style))

        test_mb = data.get("test_size_mb", 1024)
        test_gb = test_mb / 1024.0

        gpu_rows = [
            ["GPU Model", gpu_state.get("gpu_name", "Unknown")],
            ["VRAM", f"{gpu_state.get('vram_mb', 0) / 1024.0:.2f} GB"],
            ["Compute Capability", gpu_state.get("compute_cap", "Unknown")],
            ["CUDA Cores", str(gpu_state.get("cuda_cores", 0))],
            ["Streaming Multiprocessors", str(gpu_state.get("sm_count", 0))],
            ["GPU Clock", f"{gpu_state.get('gpu_clock_mhz', 0)} MHz"],
            ["Memory Clock", f"{gpu_state.get('mem_clock_mhz', 0)} MHz"],
            ["Performance State", gpu_state.get("pstate", "Unknown")],
            ["Test Size", f"{test_mb} MB ({test_gb:.2f} GB)"],
        ]

        gpu_table = Table(gpu_rows, colWidths=[2.8 * inch, 3.4 * inch])
        gpu_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECF0F1")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), PDF_TABLE_SIZE),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(gpu_table)
        story.append(Spacer(1, PDF_SECTION_SPACER))

    # KERNEL EXECUTION
    story.append(Paragraph("KERNEL EXECUTION", header_style))

    naive_bw = data.get("naive_bw", 0.0)
    prefetch_bw = data.get("prefetch_bw", 0.0)
    speedup_bw = prefetch_bw / naive_bw if naive_bw > 0 else 0.0

    kernel_rows = [
        ["Method", "Time (ms)", "Speedup"],
        ["Naive UM", f"{data['naive_ms']:.1f}", "1.0x (baseline)"],
        ["Prefetch UM", f"{data['prefetch_ms']:.1f}", f"{speedup_bw:.1f}x"],
    ]

    kernel_table = Table(kernel_rows, colWidths=[2.4 * inch, 2.0 * inch, 1.8 * inch])
    kernel_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), PDF_TABLE_SIZE),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ]
        )
    )
    story.append(kernel_table)
    story.append(Spacer(1, PDF_SECTION_SPACER))

    # PAGE FAULT TRANSFERS
    story.append(Paragraph("PAGE FAULT TRANSFERS (Naive Only)", header_style))

    avg_kb = (
        int((data["transfer_gb"] * 1024 * 1024) / data["transfer_count"])
        if data["transfer_count"]
        else 0
    )

    transfer_rows = [
        ["Metric", "Value"],
        ["Transfer Count", f"{data['transfer_count']:,}"],
        ["Total Volume", f"{data['transfer_gb']:.3f} GB"],
        ["Transfer Time", f"{data['transfer_time_ms']:.1f} ms"],
        ["Average Size", f"{avg_kb} KB"],
    ]

    transfer_table = Table(transfer_rows, colWidths=[2.8 * inch, 3.4 * inch])
    transfer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), PDF_TABLE_SIZE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ]
        )
    )
    story.append(transfer_table)
    story.append(Spacer(1, PDF_SECTION_SPACER))

    # PAGE FAULT ANALYSIS
    story.append(Paragraph("PAGE FAULT ANALYSIS", header_style))

    fault_rows = [
        ["Type", "Count"],
        ["GPU Page Fault Groups", f"{data['page_fault_groups']:,}"],
        ["CPU Page Faults", f"{data['cpu_page_faults']:,}"],
    ]

    fault_table = Table(fault_rows, colWidths=[2.8 * inch, 3.4 * inch])
    fault_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), PDF_TABLE_SIZE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ]
        )
    )
    story.append(fault_table)
    story.append(Spacer(1, PDF_SECTION_SPACER))

    # BANDWIDTH MEASUREMENTS
    story.append(Paragraph("BANDWIDTH MEASUREMENTS", header_style))

    improvement = prefetch_bw / naive_bw if naive_bw > 0 else 0.0

    bw_rows = [
        ["Method", "Bandwidth"],
        ["Naive UM", f"{naive_bw:.1f} GB/s"],
        ["Prefetch UM", f"{prefetch_bw:.1f} GB/s"],
        ["Improvement", f"{improvement:.1f}x"],
    ]

    bw_table = Table(bw_rows, colWidths=[2.8 * inch, 3.4 * inch])
    bw_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), PDF_TABLE_SIZE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ]
        )
    )
    story.append(bw_table)
    story.append(Spacer(1, PDF_SECTION_SPACER))

    # CONCLUSION
    story.append(Paragraph("CONCLUSION", header_style))

    conclusion_text = (
        f"Naive UM triggers {data['transfer_count']:,} page fault transfers "
        f"during kernel execution."
    )
    conclusion_style = ParagraphStyle(
        "Conclusion",
        parent=styles["Normal"],
        fontSize=PDF_TABLE_SIZE,
        textColor=colors.black,
    )
    story.append(Paragraph(conclusion_text, conclusion_style))

    doc.build(story)
    return filename


# -----------------------------
# SAVE PDF / JSON
# -----------------------------

def save_pdf(data: Dict[str, Any], gpu_state: Optional[Dict[str, Any]]) -> str:
    Path("results/pascal_analysis").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    size = data.get("test_size_mb", 1024)
    suffix = f"_{size // 1024}G" if size >= 1024 else f"_{size}M"
    filename = f"results/pascal_analysis/pascal_um{suffix}_{ts}.pdf"
    generate_pdf_report(data, gpu_state, filename)
    return filename


def build_json_data(data: Dict[str, Any],
                    gpu_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if gpu_state is None:
        gpu_state = {}
    naive_bw = data.get("naive_bw", 0.0)
    prefetch_bw = data.get("prefetch_bw", 0.0)
    speedup_bw = prefetch_bw / naive_bw if naive_bw > 0 else 0.0
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": "./pascal",
        "gpu": {
            "model": gpu_state.get("gpu_name", "Unknown"),
            "vram_gb": round(gpu_state.get("vram_mb", 0) / 1024.0, 2),
            "compute_cap": gpu_state.get("compute_cap", "Unknown"),
            "cuda_cores": gpu_state.get("cuda_cores", 0),
            "sm_count": gpu_state.get("sm_count", 0),
            "clock_mhz": gpu_state.get("gpu_clock_mhz", 0),
            "memory_clock_mhz": gpu_state.get("mem_clock_mhz", 0),
            "pstate": gpu_state.get("pstate", "Unknown"),
        },
        "test_config": {
            "size_mb": data.get("test_size_mb", 1024),
            "size_gb": round(data.get("test_size_mb", 1024) / 1024, 2),
        },
        "kernels": {
            "prefetch_ms": round(data["prefetch_ms"], 3),
            "naive_ms": round(data["naive_ms"], 3),
            "speedup": round(speedup_bw, 1),
        },
        "transfers": {
            "count": data["transfer_count"],
            "volume_gb": round(data["transfer_gb"], 3),
            "time_ms": round(data["transfer_time_ms"], 1),
            "avg_size_kb": int(
                (data["transfer_gb"] * 1024 * 1024) / data["transfer_count"]
            ) if data["transfer_count"] else 0,
        },
        "page_faults": {
            "gpu_groups": data["page_fault_groups"],
            "cpu_faults": data["cpu_page_faults"],
        },
        "bandwidth": {
            "naive_gbps": round(naive_bw, 1),
            "prefetch_gbps": round(prefetch_bw, 1),
            "improvement": round(speedup_bw, 1),
        },
    }


def save_json(data: Dict[str, Any], gpu_state: Optional[Dict[str, Any]]):
    Path("results/pascal_analysis").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    size = data.get("test_size_mb", 1024)
    suffix = f"_{size // 1024}G" if size >= 1024 else f"_{size}M"
    filename = f"results/pascal_analysis/pascal_um{suffix}_{ts}.json"
    j = build_json_data(data, gpu_state)
    with open(filename, "w") as f:
        json.dump(j, f, indent=2)
    return filename


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pascal Unified Memory Profiler")

    out = parser.add_mutually_exclusive_group()
    out.add_argument("--pdf", action="store_true", help="Generate PDF report")
    out.add_argument("--json", action="store_true", help="Print JSON output")
    out.add_argument("--nvprof", action="store_true", help="Show nvprof output")
    out.add_argument("--nsys", action="store_true", help="Create Nsight Systems profile")
    out.add_argument("--diagnose", action="store_true", help="Run diagnostics")

    parser.add_argument("--binary", default="./pascal", help="Path to CUDA binary")
    parser.add_argument("--no-warmup", action="store_true", help="Disable GPU warmup")
    parser.add_argument("--size", type=str, help="Test size: 512M, 1G, 2G, 4G")
    parser.add_argument("--device", type=int, default=None, help="CUDA device ID")

    args = parser.parse_args()

    if args.diagnose:
        print("Pascal Unified Memory Profiler — Diagnostics")
        print("===========================================")
        print(f"PyCUDA available:   {PYCUDA_AVAILABLE}")
        print(f"reportlab available:{REPORTLAB_AVAILABLE}")
        get_gpu_info(diagnose=True)
        return

    if not Path(args.binary).exists():
        print(f"Error: binary '{args.binary}' not found")
        print("Compile first: nvcc -o pascal pascal.cu")
        sys.exit(1)

    test_size_mb = None
    if args.size:
        s = args.size.upper()
        try:
            if s.endswith("G"):
                test_size_mb = int(s[:-1]) * 1024
            elif s.endswith("M"):
                test_size_mb = int(s[:-1])
            else:
                test_size_mb = int(s)
            print(f"Test size: {test_size_mb} MB")
        except Exception:
            print(f"Error: invalid size '{args.size}'. Use: 512M, 1G, 2G, 4G")
            sys.exit(1)

    if args.no_warmup:
        gpu_state = get_gpu_info()
    else:
        gpu_state = warmup_gpu(args.binary, test_size_mb)

    if args.nvprof:
        print("Running nvprof analysis...")
        print("=" * 80)
        cmd = "nvprof --unified-memory-profiling per-process-device --print-summary"
        if args.device is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {cmd}"
        cmd += f" {args.binary}"
        if test_size_mb:
            cmd += f" --mb {test_size_mb}"
        os.system(cmd)
        print("=" * 80)
        return

    if args.nsys:
        print("Running Nsight Systems profiling...")
        Path("results/nsight").mkdir(parents=True, exist_ok=True)
        size_suffix = ""
        if test_size_mb:
            size_suffix = f"_{test_size_mb // 1024}G" if test_size_mb >= 1024 else f"_{test_size_mb}M"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"results/nsight/pascal_um{size_suffix}_{ts}"
        cmd = f"nsys profile -t cuda,nvtx,osrt -o {out_file}"
        if args.device is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {cmd}"
        cmd += f" {args.binary}"
        if test_size_mb:
            cmd += f" --mb {test_size_mb}"
        cmd += " 2>/dev/null"
        os.system(cmd)
        print(f"\nNsight Systems report created: {out_file}.nsys-rep")
        print("Open with: nsys-ui")
        return

    print("Running nvprof analysis...")
    nvprof_output = run_nvprof(args.binary, test_size_mb)
    print("Extracting metrics...")
    naive_bw, prefetch_bw = get_pascal_bandwidth(args.binary, test_size_mb)
    parsed = parse_summary(nvprof_output)
    parsed["naive_bw"] = naive_bw
    parsed["prefetch_bw"] = prefetch_bw
    if test_size_mb:
        parsed["test_size_mb"] = test_size_mb

    if args.json:
        j = build_json_data(parsed, gpu_state)
        print("\n" + "=" * 80)
        print(json.dumps(j, indent=2))
        print("=" * 80 + "\n")
        return

    if args.pdf:
        name = save_pdf(parsed, gpu_state)
        print(f"PDF generated: {name}")
        print("\nSummary:")
        size = parsed.get("test_size_mb", 1024)
        print(f"  Size: {size} MB")
        print(f"  Naive: {naive_bw:.1f} GB/s  ({parsed['naive_ms']:.1f} ms)")
        print(f"  Prefetch: {prefetch_bw:.1f} GB/s  ({parsed['prefetch_ms']:.1f} ms)")
        if naive_bw > 0:
            print(f"  Speedup: {prefetch_bw / naive_bw:.1f}x")
        print(f"  Page faults: ~{parsed['transfer_count']:,}")
        return

    print("\nSelect output:")
    print("  --pdf       Generate PDF report")
    print("  --json      Print JSON")
    print("  --nvprof    Show nvprof")
    print("  --nsys      Nsight Systems profile")
    print("  --diagnose  Diagnostics")


if __name__ == "__main__":
    main()
