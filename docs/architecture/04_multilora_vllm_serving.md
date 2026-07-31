# 🏛️ Architecture Spec 04: High-Throughput Multi-LoRA Serving Infrastructure

## 1. Executive Summary
Deploying isolated base models per task leads to severe GPU memory fragmentation and cost inefficiency. This spec describes a scalable Multi-LoRA inference topology leveraging vLLM / LoRAX for dynamic adapter swapping over a unified base model using PagedAttention.

---

## 2. Serving Topology Diagram

```text
                          [ Client API Requests ]
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │   Async Gateway Router    │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │  Dynamic Micro-Batcher    │
                       │   & Token-Bucket Limiter  │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ GPU VRAM Memory Space (vLLM Engine)                                    │
 │                                                                        │
 │  ┌──────────────────────────────────────────────────────────────────┐  │
 │  │ Base Model Weights (FP16 / INT4) - Frozen Frozen in High VRAM    │  │
 │  └──────────────────────────────────────────────────────────────────┘  │
 │                                                                        │
 │  ┌──────────────────────────────────────────────────────────────────┐  │
 │  │ PagedAttention Block KV-Cache Allocator                          │  │
 │  └──────────────────────────────────────────────────────────────────┘  │
 │                                                                        │
 │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
 │  │ Dynamic Adapter A│  │ Dynamic Adapter B│  │ Dynamic Adapter C│      │
 │  │ (Finance LoRA)   │  │ (Medical LoRA)   │  │ (Code LoRA)      │      │
 │  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
 └────────────────────────────────────────────────────────────────────────┘
```
## 3. Memory & Swapping Math
For base model weight $W_0 \in \mathbb{R}^{d \times k}$, low-rank updates are computed on-the-fly without static weight merging:$$h = W_0 x + \Delta W x = W_0 x + \frac{\gamma}{r} (B \cdot A) x$$Where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, rank $r \ll \min(d, k)$, and $\gamma$ is the scaling factor.VRAM Footprint Reduction Calculation:Base Llama-3-70B (FP16): $\approx 140\text{ GB}$50 Custom Fine-Tuned Full Models: $50 \times 140\text{ GB} = 7,000\text{ GB}$ (Requires 88x H100s)1 Base Model + 50 LoRA Adapters ($r=16$): $140\text{ GB} + (50 \times 0.2\text{ GB}) = 150\text{ GB}$ (Fits on 2x H100s)