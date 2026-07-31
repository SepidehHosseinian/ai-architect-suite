# 🏛️ Architecture Spec 05: Production Evaluation & Real-Time Guardrails

## 1. Executive Summary
Ensuring enterprise compliance requires a dual-stage validation engine: fast sub-10ms deterministic proxies at inference time coupled with asynchronous batch evaluation pipelines for continuous quality tracking.

---

## 2. Two-Tier Guardrail Architecture

```text
 [ Prompt Request ]
        │
        ▼
┌───────────────────────────────┐
│ Tier 1: Real-Time Inline Guard│
│ - Regex Schema Validation     │
│ - PII Masking & Detoxification│
│ - Embedding Distance Cache    │
└───────────────┬───────────────┘
                │
                ├─────────────────────────────────┐ (Pass)
                │ (Fail)                          │
                ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Reject / Sanitize Payload     │ │ Model Generation Engine       │
└───────────────────────────────┘ └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                ┌───────────────────────────────┐
                                │ Tier 2: Async Eval Pipeline   │
                                │ - MLflow Telemetry Collection │
                                │ - RAGAS Hallucination Score   │
                                │ - Toxicity & Faithfulness     │
                                └───────────────────────────────┘
```
## 3. Evaluation Metric Formulas
Faithfulness / Hallucination Score ($F$)Measuring claims in answer $A$ supported by retrieved context $C$:$$F = \frac{\vert{}\text{Supported Claims in } A\vert{}}{\vert{}\text{Total Claims Extracted from } A\vert{}}$$Answer Relevance Score ($R$)Average cosine similarity between original query $q$ and synthetic queries $q_i$ generated from response $A$:$$R = \frac{1}{N} \sum_{i=1}^{N} \frac{\mathbf{v}_q \cdot \mathbf{v}_{q_i}}{\Vert{}\mathbf{v}_q\Vert{}_2 \Vert{}\mathbf{v}_{q_i}\Vert{}_2}$$