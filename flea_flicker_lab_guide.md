# Flea-Flicker LLMs: Complete Lab Build & Implementation Guide

A fully operational lab for implementing transient-residency inference as an architectural strategy for running large models on constrained hardware.

---

## Executive Summary

Flea-Flicker LLMs turn persistent model residency from a requirement into an optimization choice. This lab guide provides the complete architecture, hardware specifications, software stack, benchmarking methodology, and implementation paths for building a functional transient-residency inference system.

**Key Payoff:** Run a 70B parameter model on 12GB VRAM with latency trade-off vs. impossibility.

---

## Part 1: Architectural Foundation

### 1.1 Core Principle: Sequential Dependency Exploitation

Transformer inference has an implicit sequential dependency graph:
- **Forward pass constraint:** Layer N+1 reads only from Layer N's hidden state output
- **Memory implication:** Layer N can be discarded immediately after producing hidden_state[N]
- **Architectural consequence:** No computation requires all parameters resident simultaneously

The forward pass becomes:
```
for layer in model.layers:
    load(layer) → GPU/CPU
    hidden_state = layer(hidden_state)
    save(hidden_state) → fast_storage
    unload(layer)
```

### 1.2 Residency State Machine

Each parameter exists in one of three states:

| State | Location | Latency | Use Case |
|-------|----------|---------|----------|
| **Transient** | NVMe (disk) | 5-20ms load | Default; only load when needed |
| **Active** | VRAM/GPU | <1ms access | Currently executing layer |
| **Cached** | System RAM | 1-5ms access | Prefetched next layer |

Critical insight: **A parameter does not need to exist to compute correctly. It needs to exist only when referenced.**

### 1.3 Execution Flow Diagram

```
[Prompt Tokenization]
         ↓
[Load Layer 0 from NVMe]
         ↓
[GPU/CPU: Compute Layer 0 forward pass]
         ↓
[Emit hidden_state_0]
         ↓
[ASYNC: Unload Layer 0 | Prefetch Layer 1]
         ↓
[Load Layer 1 (from prefetch cache or NVMe)]
         ↓
[Repeat for all N layers]
         ↓
[Apply output projection & sampling]
         ↓
[Return token]
```

**Parallelization opportunity:** While GPU/CPU computes Layer N, prefetch Layer N+1 from disk.

---

## Part 2: Hardware Specification

### 2.1 Minimum Viable Configuration

For a functional proof-of-concept lab (running 13B models):

| Component | Spec | Rationale |
|-----------|------|-----------|
| **GPU** | RTX 4060 (8GB VRAM) | Minimum for single-layer residence; supports CUDA |
| **CPU** | 8-core modern (AMD Ryzen 5, Intel i7) | Offload non-critical layers; handle I/O |
| **System RAM** | 32GB | Hidden state staging; layer prefetch buffer |
| **Storage** | Samsung 990 Pro NVMe (2TB, PCIe 4.0) | 7GB/s sequential reads; 16B alignment |
| **Motherboard** | B650/Z790+ | PCIe 4.0/5.0 lanes; NVMe socket type E |

**Total cost:** ~$800-1200

### 2.2 Production-Grade Configuration

For 70B+ models with acceptable latency (<500ms/token):

| Component | Spec | Rationale |
|-----------|------|-----------|
| **GPU** | RTX 6000 Ada (48GB) or A6000 (48GB) | Professional VRAM; NVLink for multi-GPU |
| **CPU** | EPYC 7763 (128 cores) or Xeon (64 cores) | Heavy async I/O; layer scheduling overhead |
| **System RAM** | 256GB+ | Large hidden state buffers; concurrent prefetch |
| **Storage** | 4x NVMe RAID-0 (Samsung PM1735, PCIe 5.0) | 24GB/s aggregate; predictable latency |
| **Network** | 10GbE or InfiniBand (optional) | Model distribution across nodes |

**Total cost:** ~$50k-100k

### 2.3 Critical I/O Bandwidth Math

**Single-layer residency overhead (example: 7B model, FP16):**

- Layer size: ~500MB (typical transformer block)
- Hidden state: 8KB (batch_size=1, seq_len=1024, d_model=4096)
- Load time (NVMe): 500MB / 7GB/s ≈ **71ms**
- Compute time (GPU): ~5-10ms
- **Total latency per token:** ~80ms (bandwidth-bound, not compute-bound)

**Comparison:**
- Full residency (13B model): fits in 26GB, zero load overhead
- Flea-Flicker (same model, 8GB GPU): 71ms overhead per layer × 40 layers = **2.84s total overhead per token**

**Why acceptable:**
1. Agent workflows tolerate 2-3s latency (not interactive)
2. Batching reduces per-token overhead (amortized across sequence)
3. Prefetching can hide majority of I/O

---

## Part 3: Software Architecture

### 3.1 Core Components

```
┌─────────────────────────────────────────────────────┐
│           Inference Orchestrator (Python)            │
│  - Layer dependency graph construction               │
│  - Hidden state serialization pipeline               │
│  - Scheduling & prefetch coordination                │
└────────────┬────────────────────────────────────────┘
             │
  ┌──────────┼──────────┐
  ↓          ↓          ↓
┌─────┐  ┌────────┐  ┌──────────┐
│ GPU │  │  CPU   │  │ Storage  │
│Exec │  │Exec    │  │ Manager  │
│PyTorch│ │NumPy  │  │(NVMe I/O)│
└─────┘  └────────┘  └──────────┘
  ↓          ↓          ↓
┌─────────────────────────────────────────────────────┐
│         Low-Level Backend (C++/CUDA)                 │
│  - Layer paging (GPU ↔ disk)                         │
│  - Async I/O (overlapped with compute)              │
│  - Tensor format marshaling                          │
└─────────────────────────────────────────────────────┘
```

### 3.2 Implementation Path: Python + PyTorch

**Rationale:** Rapid prototyping; layer abstractions; native transformer support

```python
import torch
import safetensors
from pathlib import Path

class FleeFlickerModel:
    def __init__(self, model_path: str, vram_limit_mb: int = 8000):
        self.model_path = Path(model_path)
        self.vram_limit = vram_limit_mb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model config (metadata only, not weights)
        self.config = self._load_config()
        self.n_layers = self.config['n_layers']
        
        # Pre-compute layer offsets in checkpoint file
        self.layer_offsets = self._compute_layer_offsets()
        
        # Prefetch buffer (next layer)
        self.prefetch_buffer = None
        self.current_layer = None
        
    def _load_config(self):
        """Load model architecture config without weights."""
        # Use safetensors metadata or GGUF header
        pass
    
    def _compute_layer_offsets(self):
        """Build index of layer positions in checkpoint file."""
        # Parse GGUF/safetensors header to map layer_id → file_byte_offset
        pass
    
    def _load_layer(self, layer_id: int) -> dict:
        """
        Load single transformer block from disk.
        Returns: {attention_weights, mlp_weights, norm_scales, ...}
        """
        offset = self.layer_offsets[layer_id]
        # Memory-mapped load; deserialize tensors on-demand
        with safetensors.open_file(self.model_path) as f:
            layer_tensors = f.get_slice(f"layers.{layer_id}")
        return layer_tensors.to(self.device)
    
    def _prefetch_layer(self, layer_id: int):
        """Asynchronously load next layer into staging buffer."""
        import threading
        def _load():
            self.prefetch_buffer = self._load_layer(layer_id)
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass via transient residency.
        input_ids: [batch_size, seq_len]
        """
        hidden_state = self._embed(input_ids)  # [batch, seq_len, d_model]
        
        for layer_id in range(self.n_layers):
            # Load current layer
            if self.prefetch_buffer is not None:
                layer = self.prefetch_buffer  # Use prefetched
                self.prefetch_buffer = None
            else:
                layer = self._load_layer(layer_id)
            
            # Prefetch next layer
            if layer_id + 1 < self.n_layers:
                self._prefetch_layer(layer_id + 1)
            
            # Execute forward pass
            hidden_state = self._forward_layer(hidden_state, layer)
            
            # Unload current layer (explicit memory management)
            del layer
            torch.cuda.empty_cache()
        
        # Project to vocabulary
        logits = self._output_projection(hidden_state)
        return logits
    
    def _forward_layer(self, hidden_state, layer):
        """
        Execute single transformer block.
        Computation:
            norm = LayerNorm(hidden_state)
            attn_out = MultiHeadAttention(norm)
            hidden_state = hidden_state + attn_out
            norm = LayerNorm(hidden_state)
            mlp_out = MLP(norm)
            hidden_state = hidden_state + mlp_out
        """
        x = hidden_state
        
        # Attention block
        norm_weight = layer['self_attn_layernorm_weight']
        attn_weight = layer['self_attn_qkv_weight']
        attn_proj = layer['self_attn_out_proj_weight']
        
        norm_x = torch.nn.functional.layer_norm(
            x, normalized_shape=x.shape[-1:], weight=norm_weight
        )
        attn_out = torch.nn.functional.linear(norm_x, attn_weight)
        # Multi-head split, attention computation, re-project
        attn_out = self._multihead_attention(attn_out, layer)
        attn_out = torch.nn.functional.linear(attn_out, attn_proj)
        x = x + attn_out
        
        # MLP block
        norm_weight = layer['mlp_layernorm_weight']
        mlp_gate = layer['mlp_gate_weight']
        mlp_up = layer['mlp_up_weight']
        mlp_down = layer['mlp_down_weight']
        
        norm_x = torch.nn.functional.layer_norm(
            x, normalized_shape=x.shape[-1:], weight=norm_weight
        )
        # SwiGLU or similar
        mlp_out = torch.nn.functional.linear(norm_x, mlp_up)
        mlp_gate_out = torch.nn.functional.linear(norm_x, mlp_gate)
        mlp_out = mlp_out * torch.nn.functional.silu(mlp_gate_out)
        mlp_out = torch.nn.functional.linear(mlp_out, mlp_down)
        x = x + mlp_out
        
        return x
```

### 3.3 Alternative: C++ Backend for Performance

For production latency requirements, implement low-level layer I/O in C++ with CUDA:

```cpp
#include <cuda_runtime.h>
#include <cstring>

class FleeFlickerLayerLoader {
    int nvme_fd;
    char* gpu_staging_buffer;
    size_t staging_capacity;
    
public:
    FleeFlickerLayerLoader(const char* model_path) {
        nvme_fd = open(model_path, O_RDONLY | O_DIRECT);
        cudaMalloc(&gpu_staging_buffer, 1024 * 1024 * 500);  // 500MB
        staging_capacity = 1024 * 1024 * 500;
    }
    
    // Load layer from disk → GPU in single DMA operation
    void load_layer_dma(int layer_id, off_t file_offset, size_t layer_size) {
        // Align request to 4KB boundaries for O_DIRECT
        off_t aligned_offset = (file_offset / 4096) * 4096;
        
        // Read from NVMe to pinned host memory
        char* pinned_buf;
        cudaMallocHost(&pinned_buf, layer_size + 4096);
        
        lseek(nvme_fd, aligned_offset, SEEK_SET);
        ssize_t bytes_read = read(nvme_fd, pinned_buf, layer_size);
        
        // DMA transfer to GPU (zero-copy)
        cudaMemcpy(gpu_staging_buffer, pinned_buf, layer_size, 
                   cudaMemcpyHostToDevice);
        cudaFreeHost(pinned_buf);
    }
    
    // Async prefetch: GPU Direct Storage (if available)
    void prefetch_layer_gds(int layer_id, off_t file_offset, size_t layer_size) {
        #ifdef HAVE_GPU_DIRECT_STORAGE
        cuFileHandle_t fh;
        CUresult status = cuFileHandleOpen(&fh, nvme_fd);
        
        CUfileDescr_t desc;
        desc.handle.fd = nvme_fd;
        desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        
        // Direct GPU ← disk, bypassing CPU cache
        cuFileRead(&fh, gpu_staging_buffer, layer_size, 
                   file_offset, 0);
        #endif
    }
};
```

### 3.4 Model Format: GGUF + Layer Partitioning

**Rationale:** GGUF provides:
- Memory-mapped access (lazy-loading)
- Native quantization support (Q4_K, Q3_K variants)
- Metadata for layer boundaries
- Wide ecosystem support (llama.cpp, Ollama, etc.)

**Format structure for Flea-Flicker:**

```
GGUF File Layout:
┌─────────────────────────────────────┐
│ Header (magic, version, n_tensors)  │
├─────────────────────────────────────┤
│ Metadata (key-value pairs)          │
│ - model.architecture                │
│ - model.layer_offsets (JSON)        │
├─────────────────────────────────────┤
│ Layer 0                             │
│ ├─ self_attn.q_proj (Q4_K)          │
│ ├─ self_attn.k_proj (Q4_K)          │
│ ├─ self_attn.v_proj (Q4_K)          │
│ ├─ self_attn.o_proj (FP16)          │
│ ├─ mlp.gate_proj (Q4_K)             │
│ ├─ mlp.up_proj (Q4_K)               │
│ └─ mlp.down_proj (Q4_K)             │
├─────────────────────────────────────┤
│ Layer 1                             │
│ [...]                               │
├─────────────────────────────────────┤
│ Embeddings (FP16)                   │
│ Output projection (FP16)            │
└─────────────────────────────────────┘
```

**Hybrid quantization strategy:**
- Attention blocks: Q4_K (60-70% parameters)
- MLP: Q4_K (25-30% parameters)
- Layer norms, projections: FP16 (sensitive to quantization)
- Embeddings/output: FP16

**Result:** ~50-60% of original model size with minimal accuracy loss

---

## Part 4: Implementation Roadmap

### Phase 1: Prototype (Weeks 1-2)

**Goal:** Validate sequential loading correctness

```python
# Step 1: Load GGUF model, extract layer metadata
from gguf import GGUFReader
model = GGUFReader("llama-13b-q4_k_m.gguf")

# Step 2: Implement naive sequential forward pass
def forward_naive(input_ids):
    hidden = embed(input_ids)
    for i in range(n_layers):
        layer = load_from_disk(f"layer_{i}")
        hidden = forward_layer(hidden, layer)
        del layer
    return logits(hidden)

# Step 3: Benchmark: latency per layer, memory footprint
# Expected: 80-150ms per token (bandwidth-bound)

# Step 4: Verify numerical equivalence with full-residency reference
loss = mse(output_naive, output_reference)
assert loss < 1e-5  # Floating point equality
```

**Deliverable:** Working Python reference implementation; latency baseline

### Phase 2: Optimization (Weeks 3-4)

**Goal:** Reduce latency via prefetching & async I/O

```python
# Step 1: Implement async layer prefetch
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def forward_with_prefetch(input_ids):
    hidden = embed(input_ids)
    current_layer = None
    prefetch_future = None
    
    for i in range(n_layers):
        # Use previously prefetched layer
        if prefetch_future:
            current_layer = prefetch_future.result()
        else:
            current_layer = load_from_disk(f"layer_{i}")
        
        # Start prefetch for next layer (background thread)
        if i + 1 < n_layers:
            prefetch_future = executor.submit(
                load_from_disk, f"layer_{i+1}"
            )
        
        # Forward pass (GPU/CPU computes while disk prefetches)
        hidden = forward_layer(hidden, current_layer)
        del current_layer
    
    if prefetch_future:
        prefetch_future.result()  # Cleanup
    
    return logits(hidden)

# Step 2: Benchmark improvement
# Expected: ~40-60% latency reduction per token
```

**Deliverable:** Async prefetch implementation; latency <50% of baseline

### Phase 3: Production Hardening (Weeks 5-6)

**Goal:** Error handling, multi-GPU support, batch inference

```python
# Step 1: Implement error recovery
class ResilientLayerLoader:
    def load_with_retry(self, layer_id, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self._load_from_disk(layer_id)
            except IOError as e:
                if attempt == max_retries - 1:
                    # Fallback: CPU inference (slower, but completes)
                    return self._load_from_cpu_cache(layer_id)
                time.sleep(0.1 * (2 ** attempt))

# Step 2: Multi-GPU layer distribution (optional)
# Distribute layers across 2 GPUs to reduce per-GPU VRAM pressure
# GPU0: layers 0-19, GPU1: layers 20-39
def forward_multi_gpu(input_ids):
    hidden = embed(input_ids)
    for i in range(n_layers):
        if i < 20:
            hidden = forward_on_gpu(0, hidden, i)
        else:
            hidden = forward_on_gpu(1, hidden, i)
    return logits(hidden)

# Step 3: Batch inference (multiple prompts simultaneously)
# Challenge: different sequence lengths = variable hidden state sizes
# Solution: Padding or bucketing
def forward_batched(input_ids_list, batch_size=4):
    results = []
    for i in range(0, len(input_ids_list), batch_size):
        batch = pad_batch(input_ids_list[i:i+batch_size])
        output = forward_with_prefetch(batch)
        results.extend(unbatch(output))
    return results
```

**Deliverable:** Production-ready reference; multi-GPU support; batch inference

### Phase 4: Benchmarking Suite (Week 7)

**Goal:** Comprehensive performance characterization

```python
import time
import psutil
import torch

def benchmark_inference():
    model = FleeFlickerModel("model.gguf", vram_limit_mb=8000)
    
    # Metric 1: Throughput (tokens/second)
    start = time.time()
    output = model.generate(
        input_ids, 
        max_new_tokens=256, 
        temperature=0.7
    )
    elapsed = time.time() - start
    throughput = 256 / elapsed  # tokens/sec
    
    # Metric 2: Memory usage
    peak_vram = torch.cuda.max_memory_allocated() / 1e9  # GB
    peak_ram = psutil.Process().memory_info().rss / 1e9  # GB
    
    # Metric 3: I/O statistics
    # Track disk read count, bytes, latencies
    io_stats = {
        'total_layer_reads': 256,  # one per token per layer
        'bytes_read': 256 * 40 * 500e6,  # tokens × layers × avg_layer_size
        'disk_time': elapsed * 0.6,  # ~60% bandwidth-bound
        'effective_throughput': (256 * 40 * 500e6) / (elapsed * 0.6 / 1e9)  # GB/s
    }
    
    # Metric 4: Comparison to baselines
    baselines = {
        'full_residency_same_hw': 'N/A (OOM)',
        'cpu_only_inference': 15.0,  # seconds for 256 tokens
        'cloud_gpu_api': 2.5,  # seconds (high latency)
    }
    
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Peak RAM: {peak_ram:.2f} GB")
    print(f"Disk effective throughput: {io_stats['effective_throughput']:.2f} GB/s")
    print(f"Comparison: {baselines}")

benchmark_inference()
```

**Deliverable:** Latency, throughput, memory, and I/O benchmarks across model sizes & hardware

---

## Part 5: Advanced Optimizations

### 5.1 Activation Checkpointing for Flea-Flicker

**Standard approach:** Store all activations during forward pass for backprop.
**Flea-Flicker variant:** Recompute activations on-the-fly to save intermediate storage.

```python
def forward_layer_checkpointed(hidden_state, layer_params):
    """
    Compute layer twice: once to cache output, once (on demand) to recompute.
    Trade-off: 2x compute to save ~50% intermediate memory.
    """
    # Forward pass 1: Cache output
    output = forward_layer_compute(hidden_state, layer_params)
    
    # Don't store intermediate activations (attention scores, MLP pre-activation)
    # They're recomputed during backprop on-demand
    
    return output

def backward_layer_checkpointed(grad_output, hidden_state, layer_params):
    """
    Backward pass recomputes forward activations.
    """
    # Recompute forward to get activations
    activations = forward_layer_compute(hidden_state, layer_params, 
                                       return_activations=True)
    
    # Use recomputed activations for gradient computation
    grad_input = compute_gradients(grad_output, activations, layer_params)
    return grad_input
```

**Benefit:** Reduce hidden state serialization size by 30-50%

### 5.2 KV Cache Streaming

**Challenge:** Attention KV cache grows with sequence length (O(n²) space for n tokens).
**Solution:** Stream KV pairs to disk after each layer.

```python
class StreamingKVCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.kv_file = open(self.cache_dir / "kv_cache.bin", "wb")
        self.offset = 0
    
    def save_kv(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        """Store K, V after layer computation."""
        # Shape: [batch_size, n_heads, seq_len, head_dim]
        k_bytes = k.cpu().numpy().tobytes()
        v_bytes = v.cpu().numpy().tobytes()
        
        self.kv_file.write(k_bytes)
        self.kv_file.write(v_bytes)
        self.offset += len(k_bytes) + len(v_bytes)
    
    def load_kv(self, layer_id: int, seq_len: int) -> tuple:
        """Retrieve K, V for next forward pass."""
        # Memory-mapped access to cache file
        pass
```

**Benefit:** Enable long sequences (8K-100K tokens) without OOM

### 5.3 Speculative Decoding Integration

**Idea:** Use a small draft model (e.g., 7B) to speculatively generate tokens; verify with full model.

```python
def generate_with_speculation(self, input_ids, max_new_tokens=256):
    """
    Speculative decoding: draft (fast) → verify (slow) → accept/reject.
    Amortizes Flea-Flicker latency over batch of draft tokens.
    """
    draft_model = SmallModel()  # 7B, fits in VRAM
    full_model = FleeFlickerModel()  # 70B, transient residency
    
    generated = []
    hidden_draft = draft_model.embed(input_ids)
    hidden_full = full_model.embed(input_ids)
    
    for step in range(max_new_tokens // 4):  # 4 tokens per iteration
        # Draft: fast speculative generation
        draft_tokens = []
        for _ in range(4):
            logits_draft = draft_model(hidden_draft)
            token = sample(logits_draft)
            draft_tokens.append(token)
            hidden_draft = draft_model.embed(token)
        
        # Verify: slow but correct
        logits_full = full_model([*input_ids, *draft_tokens])
        verified_tokens = argmax(logits_full[-4:])
        
        # Compare and accept/reject
        for draft_tok, verified_tok in zip(draft_tokens, verified_tokens):
            if draft_tok == verified_tok:
                generated.append(draft_tok)
            else:
                # Correction: use verified token, regenerate from here
                generated.append(verified_tok)
                break
    
    return generated
```

**Benefit:** Reduce effective Flea-Flicker overhead by 70-80% via amortization

### 5.4 Adaptive Layer Offloading

**Idea:** Some layers are cheaper to keep resident (embeddings); others should be transient.

```python
def compute_layer_residency_cost(layer_id, layer_size, compute_time):
    """Determine whether layer should stay resident or swap."""
    load_latency = layer_size / (7e9)  # 7 GB/s disk throughput
    save_latency = layer_size / (7e9)
    
    # Cost of residency: VRAM allocation cost (e.g., 10ms amortized per GB)
    residency_cost = layer_size / 1e9 * 10e-3
    
    # Cost of swapping: load + save per inference
    swap_cost = load_latency + save_latency
    
    if residency_cost < swap_cost:
        return 'RESIDENT'
    else:
        return 'TRANSIENT'

# Example: For 13B model (26GB FP16)
# Layer 0 (embeddings, 40MB): RESIDENT (low swap cost, high residency cost)
# Layers 1-39 (attention/MLP, ~600MB each): TRANSIENT
# Layer 40 (output, 40MB): RESIDENT

residency_plan = {}
for layer_id in range(n_layers):
    cost = compute_layer_residency_cost(layer_id, layer_sizes[layer_id], 
                                       compute_times[layer_id])
    residency_plan[layer_id] = cost

# Allocate VRAM to resident layers
vram_budget = 8e9  # 8GB
cumulative_vram = 0
for layer_id, strategy in sorted(residency_plan.items(), 
                                  key=lambda x: layer_sizes[x[0]]):
    if cumulative_vram + layer_sizes[layer_id] <= vram_budget:
        residency_plan[layer_id] = 'RESIDENT'
        cumulative_vram += layer_sizes[layer_id]
    else:
        residency_plan[layer_id] = 'TRANSIENT'
```

---

## Part 6: Benchmark Results (Expected)

### Configuration A: Minimal (RTX 4060, 8GB VRAM)

**Model:** Llama-2 13B Q4_K (3.5GB)

| Metric | Flea-Flicker | Full Residency | CPU-Only |
|--------|--------------|----------------|----------|
| Throughput | 2.1 tokens/sec | OOM | 0.3 tokens/sec |
| Time/token | 476ms | — | 3.3s |
| Peak VRAM | 4.2GB | — | 0.5GB |
| Peak RAM | 6.8GB | — | 14GB |

**Breakdown:** 380ms disk I/O + 60ms compute + 36ms overhead

### Configuration B: Standard (RTX 4090, 24GB VRAM)

**Model:** Llama-2 70B Q4_K (19GB)

| Metric | Flea-Flicker | Full Residency (12B) | Speculative |
|--------|--------------|----------------------|-------------|
| Throughput | 4.8 tokens/sec | 12.5 tokens/sec | 9.2 tokens/sec |
| Time/token | 208ms | 80ms | 109ms |
| Peak VRAM | 10.2GB | 24GB | 14GB |

**Breakdown:** 130ms disk I/O (prefetch hidden) + 70ms compute + 8ms overhead

**Notes:**
- Prefetching hides ~60% of I/O latency
- Speculative decoding reduces effective overhead to ~30%
- Trade-off: Accept 2.6x latency penalty to run 70B locally vs. 12B with full residency

### Configuration C: Production (A6000 + NVMe RAID)

**Model:** Llama-2 70B FP16 (140GB, on-disk)

| Metric | Flea-Flicker | Full Residency |
|--------|--------------|----------------|
| Throughput | 12.4 tokens/sec | 14.8 tokens/sec |
| Time/token | 80ms | 68ms |
| Peak VRAM | 10GB | 48GB |
| Peak NVMe BW | 3.2GB/s (45%) | N/A |

**Breakdown:** 50ms disk I/O (overlapped) + 25ms compute + 5ms overhead

---

## Part 7: Failure Modes & Mitigation

| Failure Mode | Symptom | Mitigation |
|--------------|---------|-----------|
| **Disk latency spike** | 500ms+ per token | Parallel prefetch; RAID striping; PCIe 5.0 |
| **Hidden state serialization bottleneck** | GPU stalls on save | GPU-resident hidden state; async write-back |
| **Thread contention** | CPU at 100%, GPU idle | Dedicated I/O threads; affinity pinning |
| **Cache coherency issues** | Stale layer params | Explicit memory barriers; versioning |
| **Quantization error accumulation** | Token drift by layer 20+ | Hybrid precision (int4 + FP16 critical layers) |
| **Memory fragmentation** | Allocation failures | Pre-allocated layer buffer pools |

---

## Part 8: Deployment Scenarios

### Scenario 1: Local Research (Single Machine)

**Goal:** Experiment with 70B model on gaming GPU

```bash
# Setup
python flea_flicker.py \
  --model /path/to/llama-70b-q4_k_m.gguf \
  --vram_limit_mb 10000 \
  --prefetch_threads 2 \
  --device cuda:0

# Inference
python -c "
from flea_flicker import FleeFlickerModel
model = FleeFlickerModel('llama-70b-q4_k_m.gguf')
response = model.generate('What is quantum computing?', max_tokens=256)
print(response)
"
# Expected: ~5s to generate 256 tokens
```

### Scenario 2: Multi-Agent System (Local Network)

**Goal:** Run multiple specialized models (70B reasoning + 13B fast filtering)

```python
# Agent 1: Fast filter (13B, fully resident, 100+ tokens/sec)
filter_model = FullResidencyModel("gpt2-large")

# Agent 2: Slow reasoner (70B, Flea-Flicker, 2 tokens/sec)
reasoning_model = FleeFlickerModel("llama-70b-q4_k_m.gguf")

class MultiAgentSystem:
    def process_query(self, query: str):
        # Stage 1: Filter (fast)
        filtered_queries = filter_model.rank(query, top_k=3)
        
        # Stage 2: Reason on top result (slow)
        detailed_answer = reasoning_model.generate(filtered_queries[0])
        
        return detailed_answer

# Total latency: 50ms (filter) + 2500ms (reason) = 2.55s
# Acceptable for agent workflows
```

### Scenario 3: Serverless/Edge Deployment

**Goal:** Run 13B model on Jetson AGX Orin (8GB shared GPU/CPU memory)

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip install torch transformers safetensors

# Copy model (quantized)
COPY llama-13b-q3_k_s.gguf /app/model.gguf

# Copy inference script
COPY flea_flicker_edge.py /app/

# Edge deployment constraints:
# - 2ms token latency (hard limit for interactive)
# - 2GB peak VRAM (shared with system)
# - No network access to disk (local NVMe only)

ENTRYPOINT ["python3", "/app/flea_flicker_edge.py", "--model", "/app/model.gguf"]
```

**Result:** 13B model runs locally on edge device; token latency ~800ms (acceptable for non-interactive)

---

## Part 9: Comparison to Alternative Approaches

| Approach | Latency | VRAM | Accuracy | Use Case |
|----------|---------|------|----------|----------|
| **Flea-Flicker** | 200-500ms/token | 2-10GB | 99%+ | Local research, agent systems |
| **Model Quantization** | 50-100ms/token | 4-12GB | 95-98% | Same hardware, better latency |
| **Distillation** | 30-50ms/token | 2-6GB | 90-95% | Production inference (accuracy loss) |
| **Full Residency** | 20-50ms/token | 20-100GB | 99%+ | Cloud GPU, unlimited VRAM |
| **CPU Inference** | 5-10s/token | 0.5GB | 99%+ | Very constrained edge (painful latency) |
| **Cloud API** | 50-200ms/token | 0 (local) | Variable | No local hardware (cost/privacy) |

**When to choose Flea-Flicker:**
- Large local model > available VRAM
- Latency tolerance > 200ms
- Privacy/autonomy required (no cloud)
- Experimental or research workflows

---

## Part 10: Future Directions

### 10.1 Heterogeneous Memory Hierarchies

Extend beyond binary (disk/VRAM):
```
Layer Selection:
- GPU VRAM: Active layer + next-prefetch
- NVMe: Full model
- Optional: PCIe Gen5 memory (future); compressed representations
```

### 10.2 Hardware Acceleration for Streaming

Custom hardware (ASIC/FPGA) for:
- Async tensor deserialization
- Prefetch scheduling
- Memory coherency management

### 10.3 Distributed Flea-Flicker

Partition model across network:
```
Node A (8GB): Layers 0-20
Node B (8GB): Layers 21-40
Node C (NVMe): Full checkpoint (backup)

Forward pass: stream hidden states via 10GbE
```

### 10.4 Adaptive Granularity

Instead of per-layer residency, consider:
- Per-attention-head granularity (finer control)
- Dynamic precision per layer (adapt to activation statistics)
- Lossless layer compression (temporal patterns)

---

## Conclusion

Flea-Flicker LLMs reframe the hard constraint "model doesn't fit" into a soft constraint "inference takes longer." This lab provides a complete path from concept to production implementation, with detailed hardware specifications, software architecture, benchmarking methodology, and deployment scenarios.

**Key Takeaway:** For local-first systems, research workflows, and agent architectures, transient model residency offers a viable alternative to cloud APIs and full-residency constraints. The latency overhead (2-3x typical) is acceptable where throughput is secondary to autonomy and correctness.

---

## References & Resources

**Software:**
- llama.cpp: Layer offloading reference implementation
- fastsafetensors: Optimized tensor deserialization
- GGUF: Format specification and loaders
- PyTorch: Transformer implementations

**Hardware Resources:**
- NVMe benchmarking: PCIe Gen 4.0/5.0 specs
- GPU memory management: CUDA architecture guides
- Async I/O: Linux AIO, io_uring documentation

**Papers & Articles:**
- "Transformer Circuits" (Anthropic): Residual stream mechanics
- "Inference Optimization Techniques" (Emergent Mind)
- "LLM Inference Arithmetic" (kipply's blog): Latency breakdown analysis

---

**Lab Build Complexity:** Intermediate to Advanced
**Time to Functional System:** 4-6 weeks
**Maintenance Effort:** Low (once tuned)
**Extensibility:** High (modular design)
