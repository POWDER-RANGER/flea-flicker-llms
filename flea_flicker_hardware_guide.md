# Flea-Flicker LLMs: Hardware Procurement & Setup Guide

Detailed component selection, configuration, and validation for building functional Flea-Flicker systems.

---

## 1. Hardware Configuration Profiles

### Profile A: Hobbyist / Research Lab (~$800-1200)

**Target:** Run 13B models; experimentation; single user

| Component | Recommendation | Rationale | Cost |
|-----------|-----------------|-----------|------|
| **GPU** | NVIDIA RTX 4060 8GB | Entry-level CUDA; sufficient for single-layer residency | $250 |
| **CPU** | AMD Ryzen 5 5500 (6-core) | Adequate for async I/O; no bottleneck | $150 |
| **Motherboard** | B550 chipset | PCIe 4.0 x4 NVMe slot | $120 |
| **RAM** | 32GB DDR4 3600MHz | Hidden state staging; prefetch buffer | $90 |
| **NVMe** | Samsung 990 Pro 2TB (PCIe 4.0) | 7GB/s sequential; crucial for latency | $180 |
| **PSU** | 750W 80+ Gold | RTX 4060 + CPU + storage I/O | $80 |
| **Case/Cooling** | Standard ATX + Tower cooler | Thermal management for sustained inference | $100 |

**Total:** ~$970

**Performance Baseline (Llama-2 13B Q4_K):**
- Token latency: 450-500ms
- Throughput: 2-2.5 tokens/sec
- Peak VRAM: 4.2GB
- Peak RAM: 7GB

---

### Profile B: Small Team / Production Lab (~$4000-6000)

**Target:** Run 30-70B models; multiple concurrent users; production testing

| Component | Recommendation | Rationale | Cost |
|-----------|-----------------|-----------|------|
| **GPU** | NVIDIA RTX 4090 24GB | Professional VRAM; high bandwidth | $1800 |
| **CPU** | AMD Ryzen 9 7950X (16-core) | Parallelize async I/O; fast layer processing | $550 |
| **Motherboard** | X870 with PCIe 5.0 | Future-proof; faster storage | $300 |
| **RAM** | 128GB DDR5 6000MHz | Multiple concurrent hidden states; prefetch | $600 |
| **NVMe** | 2x Samsung PM1735 2TB (PCIe 5.0) RAID-0 | 12GB/s aggregate; SSD striping | $1200 |
| **PSU** | 1200W 80+ Platinum | RTX 4090 peak draw + sustained I/O | $250 |
| **Case/Cooling** | Server chassis + AIO liquid | Professional thermal + airflow | $400 |
| **Networking** | 10GbE (optional for multi-node) | Model distribution, fallback | $200 |

**Total:** ~$5300

**Performance Baseline (Llama-2 70B FP16, quantized to Q4_K):**
- Token latency: 180-220ms
- Throughput: 4.5-5.5 tokens/sec
- Peak VRAM: 10GB
- Peak RAM: 30GB

---

### Profile C: Enterprise / Data Center (~$50k-100k)

**Target:** 70B+ models; high throughput; distributed inference

| Component | Recommendation | Rationale | Cost |
|-----------|-----------------|-----------|------|
| **GPU** | 2x NVIDIA A6000 48GB | Professional VRAM; dual-GPU support | $24000 |
| **CPU** | 2x EPYC 7763 64-core | Heavy async I/O; distributed layer processing | $8000 |
| **Motherboard** | EPYC TRX50 socket | Dual CPU; NVMe bifurcation | $2000 |
| **RAM** | 256GB DDR5 ECC | ECC for reliability; massive prefetch | $5000 |
| **NVMe** | 4x Samsung PM1735 2TB (PCIe 5.0) RAID-0 | 24GB/s aggregate throughput | $2400 |
| **NVLink Bridge** | NVIDIA NVLink 3 (A6000 compatible) | GPU-GPU 1.8TB/s | $3000 |
| **Network** | 40GbE InfiniBand | Multi-node orchestration | $5000 |
| **Cooling** | Datacenter liquid cooling | Sustained 300W+ per GPU | $10000 |
| **Enclosure** | 2U server chassis | Professional redundancy | $3000 |

**Total:** ~$62,400

**Performance Baseline (Llama-2 70B FP16, distributed):**
- Token latency: 60-90ms
- Throughput: 11-16 tokens/sec
- Peak VRAM per GPU: 12GB
- Peak aggregate I/O: 20GB/s

---

## 2. Component Deep-Dive: Critical Selections

### 2.1 GPU Selection Matrix

| GPU Model | VRAM | Memory BW | TF32 | Cost | Flea-Flicker Fit |
|-----------|------|-----------|------|------|------------------|
| RTX 4060 | 8GB | 432 GB/s | 91 TF | $250 | ✓ (13B max) |
| RTX 4070 | 12GB | 504 GB/s | 183 TF | $500 | ✓ (20B max) |
| RTX 4080 | 16GB | 576 GB/s | 367 TF | $1200 | ✓ (30B max) |
| RTX 4090 | 24GB | 1008 GB/s | 733 TF | $1800 | ✓ (70B max) |
| L40 | 48GB | 864 GB/s | 730 TF | $8000 | ✓ (140B max) |
| A6000 | 48GB | 576 GB/s | 667 TF | $4500 | ✓ (140B max) |

**Critical metric for Flea-Flicker:** Memory bandwidth matters more than compute (TF32) because inference is bandwidth-bound.

**Recommendation:** Prioritize high memory bandwidth per dollar:
- RTX 4090: 0.41 GB/s per dollar (best value)
- A6000: 0.13 GB/s per dollar (professional reliability)

### 2.2 NVMe Selection Matrix

| Drive | Interface | Seq Read | Seq Write | QD | Cost | Endurance |
|-------|-----------|----------|-----------|----|----|-----------|
| 990 Pro | PCIe 4.0 | 7.1 GB/s | 6.0 GB/s | 4096 | $0.09/GB | 600 TBW |
| 990 EVO | PCIe 4.0 | 5.0 GB/s | 4.3 GB/s | 4096 | $0.05/GB | 300 TBW |
| PM1735 | PCIe 5.0 | 12.4 GB/s | 10.2 GB/s | 32 | $0.30/GB | 1200 TBW |
| Optane DC | PCIe 3.0 | 2.7 GB/s | 2.2 GB/s | 16 | $0.20/GB | 2500 TBW |

**Critical for Flea-Flicker:** Sequential read bandwidth is the primary performance lever.

**Recommendation:**
- **Consumer:** Samsung 990 Pro (7GB/s, proven reliability)
- **Enterprise:** Samsung PM1735 (12GB/s, required for <100ms latency)

**NVMe Configuration:**
- **Single drive (Profile A):** Direct mount, no RAID overhead
- **2-4 drives (Profile B/C):** RAID-0 stripe for aggregate throughput
  - 2x 990 Pro: ~14 GB/s
  - 4x PM1735: ~24 GB/s
  - RAID-0 penalty: None for sequential reads (our use case)

### 2.3 CPU Selection Matrix

| CPU | Cores | Memory BW | PCIe Lanes | Cost |
|-----|-------|-----------|-----------|------|
| Ryzen 5 5500 | 6 | 44.8 GB/s | 16+4 | $150 |
| Ryzen 9 7950X | 16 | 89.6 GB/s | 24+8 | $550 |
| EPYC 7763 | 64 | 179.2 GB/s | 128 | $4000 |

**Critical for Flea-Flicker:** CPU primarily handles async I/O scheduling, not compute.

**Recommendation:** Choose CPU based on concurrency needs:
- 1 concurrent inference: Ryzen 5 5500 sufficient
- 2-4 concurrent: Ryzen 9 7950X recommended
- 8+ concurrent: EPYC required

### 2.4 Motherboard PCIe Topology

**Critical for Flea-Flicker:** NVMe slot PCIe generation directly impacts layer load latency.

| Motherboard Type | NVMe Slot PCIe | GPU Slot PCIe | Cost |
|------------------|-----------------|----------------|------|
| B550 | 4.0 x4 | 4.0 x16 | $120 |
| X870 | 5.0 x4 | 5.0 x16 | $300 |
| TRX50 | 5.0 x4×4 (bifurcation) | 5.0 x16×2 | $2000 |

**Key point:** Ensure NVMe slot is NOT shared with GPU slot (bifurcation can cause bandwidth contention).

**Validation checklist:**
- [ ] NVMe slot is PCIe 4.0+ (not 3.0)
- [ ] GPU slot is PCIe 4.0+ (not 3.0)
- [ ] No bifurcation contention (independent lanes for NVMe + GPU)
- [ ] BIOS updated to latest version

---

## 3. Disk Performance Validation

Before deploying model, validate NVMe performance:

```bash
# Test 1: Sequential read performance (target: 90%+ of spec)
fio --name=sequential_read \
    --filename=/path/to/nvme \
    --ioengine=libaio \
    --iodepth=32 \
    --rw=read \
    --bs=1M \
    --size=10G \
    --direct=1 \
    --group_reporting

# Expected for 990 Pro: ~6.5 GB/s (achieves 92% of 7.1 GB/s spec)

# Test 2: Random read latency (target: <100µs)
fio --name=random_read_latency \
    --filename=/path/to/nvme \
    --ioengine=libaio \
    --iodepth=1 \
    --rw=randread \
    --bs=4K \
    --size=1G \
    --direct=1 \
    --output-format=json | jq '.jobs[0].read.lat_ns.percentile'

# Expected: P50 < 30µs, P99 < 100µs

# Test 3: Sustained random load (simulating layer loading)
fio --name=layer_simulation \
    --filename=/path/to/nvme \
    --ioengine=libaio \
    --iodepth=4 \
    --rw=read \
    --bs=512K \
    --size=50G \
    --direct=1 \
    --runtime=300 \
    --numjobs=2

# Expected: Throughput should remain stable >4 GB/s for 300s
```

---

## 4. GPU-CPU PCIe Bandwidth Verification

```bash
# Test GPU-CPU PCIe bandwidth
sudo nvidia-smi -lms 1000 -pm 1  # Persistent mode + memory logging

# Run bandwidth test (NVIDIA GPU-PCIe benchmark)
./GPCIe_BW --help
./GPCIe_BW -h2d -s1000  # Host-to-Device, 1GB transfer

# Expected for PCIe 4.0 x16: ~12 GB/s
# Expected for PCIe 5.0 x16: ~24 GB/s

# If bandwidth is <8 GB/s, check:
# 1. GPU not bifurcated to x8 mode
# 2. No other PCIe devices contending (disable Thunderbolt, etc.)
# 3. BIOS PCIe generation not set to "auto" (lock to Gen4/Gen5)
```

---

## 5. System Configuration Checklist

### BIOS Settings (for all profiles)

```
Power Management:
  [ ] Enable PCIe Precision Boost (AMD)
  [ ] Enable PCIE Link ASPM: Disabled (latency)
  [ ] Enable C6 State: Disabled (latency-sensitive)
  [ ] Enable S.M.A.R.T: Enabled (NVMe monitoring)

Storage:
  [ ] NVMe mode: NVMe (not AHCI)
  [ ] NVMe slot Gen setting: PCIe Gen 4 (minimum)
  [ ] SATA/NVMe bifurcation: Disabled (unless intentional)
  [ ] Secure Boot: Disabled (not needed)

Graphics/PCIe:
  [ ] PCIe generation: Gen 4.0 (B550) or Gen 5.0 (X870)
  [ ] Above 4GB decoding: Enabled
  [ ] ARI support: Enabled
  [ ] IOMMU/AMD-Vi: Disabled (unless using vGPU)

CPU/Memory:
  [ ] Memory profile: XMP/DOCP (stable timing)
  [ ] SVM (AMD virtualization): Disabled
  [ ] Thermal throttle: Normal (allow up to 95°C for sustained loads)
```

### Linux Kernel Tuning (for I/O performance)

```bash
# /etc/sysctl.conf additions

# I/O elevator for NVMe: default (no-op scheduler)
kernel.sched_migration_cost_ns = 500000

# Memory pressure: optimize for bulk I/O
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_writeback_centisecs = 500

# NVMe queue depth optimization
# (already kernel default, no change needed)

# Disable swap (inference latency sensitive)
vm.swappiness = 0
```

```bash
# /etc/modprobe.d/nvme.conf
options nvme core_io_timeout=30

# /etc/security/limits.conf (for ulimits)
* soft memlock unlimited
* hard memlock unlimited
* soft nofile 1000000
* hard nofile 1000000
```

### Python Environment Setup

```bash
# Create isolated environment
python3.11 -m venv /opt/flea-flicker-env
source /opt/flea-flicker-env/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors numpy pandas psutil

# For GGUF support
pip install llama-cpp-python

# For benchmarking
pip install tensorboard jupyterlab matplotlib

# Optional: For C++ extension compilation
pip install scikit-build cmake ninja

# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 6. Deployment Validation Script

```python
# validate_setup.py
import torch
import subprocess
import os
from pathlib import Path

class SetupValidator:
    def __init__(self):
        self.results = {}
    
    def validate_gpu(self):
        """Verify GPU presence and capabilities."""
        print("Testing GPU...")
        try:
            assert torch.cuda.is_available(), "CUDA not detected"
            device = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            self.results['gpu'] = {
                'available': True,
                'device': device,
                'vram_gb': vram_gb,
                'compute_capability': torch.cuda.get_device_properties(0).major
            }
            print(f"  ✓ GPU: {device} ({vram_gb:.1f}GB VRAM)")
        except Exception as e:
            self.results['gpu'] = {'available': False, 'error': str(e)}
            print(f"  ✗ GPU: {e}")
    
    def validate_nvme(self):
        """Measure NVMe bandwidth."""
        print("Testing NVMe...")
        try:
            # Assume /data/flea-flicker mount point
            nvme_path = "/data/flea-flicker"
            Path(nvme_path).mkdir(exist_ok=True, parents=True)
            
            # Sequential read test
            test_file = Path(nvme_path) / "bandwidth_test.tmp"
            test_size = 1000 * 1024 * 1024  # 1GB
            
            # Write test data
            with open(test_file, 'wb') as f:
                f.write(b'0' * test_size)
            
            # Read test
            import time
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                data = f.read(test_size)
            elapsed = time.perf_counter() - start
            
            bandwidth_gbps = test_size / elapsed / 1e9
            test_file.unlink()
            
            self.results['nvme'] = {
                'path': nvme_path,
                'bandwidth_gbps': bandwidth_gbps,
                'target_gbps': 7.0,
                'meets_target': bandwidth_gbps >= 6.0  # 85% of target
            }
            
            status = "✓" if bandwidth_gbps >= 6.0 else "⚠"
            print(f"  {status} NVMe: {bandwidth_gbps:.1f} GB/s (target: 7.0)")
        except Exception as e:
            self.results['nvme'] = {'error': str(e)}
            print(f"  ✗ NVMe: {e}")
    
    def validate_pcie(self):
        """Check PCIe bandwidth between GPU and CPU."""
        print("Testing PCIe...")
        try:
            # Simple GPU-CPU transfer test
            test_size = 1024 * 1024 * 1024  # 1GB
            tensor = torch.randn(test_size // 8, dtype=torch.float32)  # 256M floats
            
            import time
            # GPU → CPU
            tensor_gpu = tensor.cuda()
            start = time.perf_counter()
            tensor_cpu = tensor_gpu.cpu()
            elapsed = time.perf_counter() - start
            
            bandwidth_gbps = test_size / elapsed / 1e9
            
            self.results['pcie'] = {
                'gpu_to_cpu_bandwidth_gbps': bandwidth_gbps,
                'target_gbps': 12.0,  # PCIe 4.0 x16
                'meets_target': bandwidth_gbps >= 10.0
            }
            
            status = "✓" if bandwidth_gbps >= 10.0 else "⚠"
            print(f"  {status} PCIe: {bandwidth_gbps:.1f} GB/s (target: 12.0)")
        except Exception as e:
            self.results['pcie'] = {'error': str(e)}
            print(f"  ✗ PCIe: {e}")
    
    def validate_memory(self):
        """Check system RAM availability."""
        print("Testing System Memory...")
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / 1e9
            available_ram_gb = psutil.virtual_memory().available / 1e9
            
            self.results['memory'] = {
                'total_gb': total_ram_gb,
                'available_gb': available_ram_gb,
                'utilization_percent': psutil.virtual_memory().percent
            }
            
            print(f"  ✓ RAM: {total_ram_gb:.1f}GB total, "
                  f"{available_ram_gb:.1f}GB available")
        except Exception as e:
            self.results['memory'] = {'error': str(e)}
            print(f"  ✗ RAM: {e}")
    
    def run_all(self):
        """Run all validation tests."""
        print("=" * 50)
        print("FLEA-FLICKER SETUP VALIDATOR")
        print("=" * 50 + "\n")
        
        self.validate_gpu()
        self.validate_nvme()
        self.validate_pcie()
        self.validate_memory()
        
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        # Pass/fail determination
        all_pass = (
            self.results['gpu'].get('available', False) and
            self.results['nvme'].get('meets_target', False) and
            self.results['pcie'].get('meets_target', False)
        )
        
        if all_pass:
            print("✓ All systems ready for Flea-Flicker deployment!")
        else:
            print("⚠ Some systems need attention. See above for details.")
        
        return self.results

if __name__ == "__main__":
    validator = SetupValidator()
    results = validator.run_all()
```

---

## 7. Procurement Checklist

### Pre-Purchase

- [ ] Verify CPU socket compatibility (AM5, LGA1851, etc.)
- [ ] Confirm motherboard PCIe 4.0+ availability
- [ ] Check GPU power requirements (750W PSU min for RTX 4090)
- [ ] Verify NVMe form factor (M.2 2280)
- [ ] Confirm RAM DDR4/DDR5 requirement (motherboard-dependent)
- [ ] Check case compatibility (GPU length, radiator clearance)

### Post-Assembly

- [ ] Update motherboard BIOS to latest version
- [ ] Enable XMP/DOCP in BIOS
- [ ] Set NVMe slot to PCIe Gen 4.0+ (not "Auto")
- [ ] Disable C-states and power management features
- [ ] Install latest GPU drivers (NVIDIA 550+)
- [ ] Test with `nvidia-smi` and `cuda-samples`

### Pre-Deployment

- [ ] Run `validate_setup.py` (all systems green)
- [ ] Benchmark NVMe with `fio` (meet sequential read target)
- [ ] Benchmark PCIe with GPU bandwidth test
- [ ] Install Python environment (verified CUDA support)
- [ ] Download target model (verify checksum)
- [ ] Run inference on dummy input (latency baseline)

---

## 8. Expected Costs & ROI

### Cost Breakdown (Profile A: $970)

| Year | Operational Cost | Notes |
|------|------------------|-------|
| **Year 1** | $970 (hardware) + $0 (electricity) | ~$0.30/kWh, 8W avg at idle |
| **Year 2** | $150 (storage upgrade) | Add 2TB capacity |
| **Year 3** | $0 | System stable; no replacements |

**Total 3-year cost:** $1120

**Comparison to cloud GPU rental:**
- Cloud API: $1-2/hour for 70B model
- 100 hours inference/month: $100-200/month = $3600-7200/year
- Flea-Flicker breaks even after 5-10 hours of use

### ROI Justification

**Scenario 1: Research Lab (100 hours inference/month)**
- Cloud cost: $3600/year
- Flea-Flicker: $970 first year, $0 after
- **Payback: ~3.2 months**

**Scenario 2: Production Agent (24/7 inference)**
- Cloud cost: $8760-17520/year
- Flea-Flicker: $970 (Profile A) or $5300 (Profile B)
- **Payback: 20-60 days**

---

This guide provides everything needed for hardware procurement and deployment. Follow the validation checklist before deploying models to ensure optimal performance.
