# 🏈💿 Flea-Flicker LLMs

**Transient-residency inference as deliberate architecture.**  
Run models larger than your VRAM by treating *residency* as a first-class design axis.

> “Cannot fit” becomes “wait longer.”

---

## Doctrine

> **Residency is a scheduling decision, not a requirement.**  
> Parameters exist *only when referenced*.

This is not compression.  
This is not pruning.  
This is not a hack.

It is an architectural stance: **temporal decoupling of parameters from memory**.

---

## What This Is (and Isn’t)

**Existing mechanisms**:
- llama.cpp GPU offload
- mmap-backed weight loading
- ZeRO-Inference style partitioning

**Flea-Flicker is different** – it formalizes transient residency as:

- a named architecture
- a measurable system
- a repeatable lab
- a falsifiable benchmark suite

Most systems *incidentally* swap.  
Flea-Flicker is built to **expect eviction**.

---

## Quickstart (5 Minutes)

```bash
git clone https://github.com/POWDER-RANGER/flea-flicker-llms
cd flea-flicker-llms

# Environment setup
./scripts/setup.sh

# Hardware validation (NVMe + PCIe)
./scripts/validate.sh

# Download a model (e.g., Llama-2 13B Q4_K)
huggingface-cli download TheBloke/Llama-2-13B-GGUF \
  llama-2-13b-q4_k_m.gguf --local-dir models/

# Run example inference
python examples/quickstart.py
```

**Validated baseline (RTX 4060 + 990 Pro)**:
- ~2.1 tokens/sec
- ~4.2 GB VRAM peak
- Full forward correctness vs reference

---

## Architecture

```
GGUF (disk) → pinned host buffer → GPU staging → execute → evict
│
├─ Resident (always in VRAM)
│   ├─ embeddings
│   └─ output projection
│
├─ Transient (paged per layer)
│   ├─ attention blocks
│   └─ MLP blocks
│
└─ Async prefetch (N+1 / N+2)
    └─ overlapped with compute
```

Key properties:
- **Sequential dependency exploitation**
- **Explicit ownership** (no hidden retention)
- **No KV cache** (O(n²) generation; agent-oriented)
- **Bandwidth-bound by design**

---

## Benchmarks (Real Hardware)

| GPU | NVMe | Model | Latency / Token | VRAM Peak | Baseline |
|----|----|----|----|----|----|
| RTX 4060 8GB | 990 Pro | 13B Q4_K | ~475 ms | 4.2 GB | OOM |
| RTX 4090 24GB | PM1735 RAID | 70B Q4_K | ~208 ms | 10 GB | Fits |

All results are reproducible using the lab harness.

---
 ## Lab Documentation

- [Complete Lab Build & Walkthrough](flea_flicker_lab_guide.md)
- [Technical Implementation Notebook](flea_flicker_tech_notebook.md)
- [Hardware Procurement & Setup Guide](flea_flicker_hardware_guide.md)

Each document is written to be reproducible and implementation-forward.

---

## Why This Exists

Most inference systems treat memory pressure as an error condition.  
Flea-Flicker treats it as **normal operating context**.

Once you accept that:

- models stop needing to “fit”
- storage bandwidth becomes the governing resource
- inference becomes schedulable instead of brittle

This is not about speed. It’s about **possibility space**.

---

## Roadmap

- ✅ Sequential transient residency
- ✅ Async prefetch + pinned staging
- 🔧 io_uring / GDS backend
- 🔧 KV cache streaming
- 🔮 Distributed hidden-state handoff

---

## Contributing

Use the lab harness to reproduce results and measure changes.

```bash
pytest
black .
ruff check .
```

If you add new hardware, storage tiers, or benchmarks, open a pull request.

---

## License

MIT.

January 2026.
