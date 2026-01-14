# Contributing to Flea-Flicker LLMs

Thank you for your interest in contributing to Flea-Flicker! This project formalizes transient-residency inference as a repeatable, measurable architecture.

## Philosophy

Flea-Flicker treats memory pressure as normal operating context, not an error condition. Contributions should:

- **Embrace bandwidth constraints** as a design parameter
- **Measure everything** - reproducible benchmarks are required
- **Document hardware dependencies** explicitly
- **Maintain sequential clarity** - avoid hidden state retention

## Getting Started

### Prerequisites

```bash
# Required
Python 3.10+
CUDA 11.8+ / ROCm 5.7+
NVMe SSD (Gen3 x4 minimum)
8GB+ VRAM GPU

# Recommended for benchmarking
Gen4/Gen5 NVMe
Pinned memory support
io_uring capable kernel (5.10+)
```

### Development Setup

```bash
git clone https://github.com/POWDER-RANGER/flea-flicker-llms
cd flea-flicker-llms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements-dev.txt

# Run validation
pytest tests/
ruff check .
black --check .
```

## How to Contribute

### Reporting Issues

**Performance Issues**:
- Include full hardware spec (GPU, NVMe model, PCIe gen)
- Attach `fio` sequential read benchmark results
- Provide model size, quantization, and batch configuration

**Correctness Issues**:
- Compare output against reference implementation (llama.cpp, etc.)
- Include random seed and full inference parameters
- Note if issue is layer-specific or accumulates

### Pull Requests

1. **Fork and create a feature branch**
   ```bash
   git checkout -b feature/my-optimization
   ```

2. **Make your changes**
   - Follow existing code style (black, ruff)
   - Add tests for new functionality
   - Update documentation

3. **Run the full test suite**
   ```bash
   pytest tests/ --benchmark
   ruff check .
   black .
   ```

4. **Submit PR with**:
   - Clear description of changes
   - Benchmark results (before/after)
   - Hardware configuration used for testing
   - Any new dependencies justified

### Code Style

- **Black** for formatting
- **Ruff** for linting
- Type hints for public APIs
- Docstrings for modules, classes, and non-trivial functions

```python
def load_layer_transient(
    layer_idx: int,
    staging_buffer: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Load layer weights into staging, transfer to GPU.
    
    Args:
        layer_idx: Zero-indexed transformer layer
        staging_buffer: Pre-allocated pinned host buffer
        device: Target CUDA device
    
    Returns:
        GPU-resident layer weights
    """
    ...
```

## Areas for Contribution

### High Priority

- **io_uring backend** - Replace `mmap` with async I/O
- **KV cache streaming** - Extend transient residency to attention states
- **Multi-GPU** - Hidden state handoff between devices
- **Benchmarking harness** - Automated hardware profiling

### Documentation

- Hardware compatibility matrix
- Cost/performance analysis ($/token for various configs)
- Architecture comparison (vs ZeRO, tensor parallelism, etc.)
- Video walkthrough of lab setup

### Testing

- Correctness tests against reference implementations
- Stress tests (OOM conditions, thermal throttling)
- Cross-platform validation (Linux, Windows, AMD GPUs)

## Measurement Standards

### Required Metrics for PRs

```python
# Include in PR description
{
  "gpu": "RTX 4060 8GB",
  "nvme": "Samsung 990 Pro 2TB",
  "model": "Llama-2-13B-Q4_K",
  "baseline_tokens_per_sec": 2.1,
  "optimized_tokens_per_sec": 2.8,
  "vram_peak_mb": 4200,
  "correctness": "validated vs llama.cpp commit abc123"
}
```

### Reproducibility

All performance claims must be reproducible using:
```bash
python benchmark/run_lab_harness.py --config configs/pr_benchmark.yaml
```

## Community

- **Questions**: Open a Discussion
- **Bugs**: Open an Issue with reproduction steps
- **Features**: Discuss in Issues before implementing
- **Documentation**: PRs always welcome

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Remember**: Flea-Flicker is about possibility space, not speed. Contributions that enable running larger models on smaller hardware are more valuable than marginal throughput gains.
