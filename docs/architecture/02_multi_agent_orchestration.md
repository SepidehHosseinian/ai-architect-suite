# 🏛️ Architecture Spec 02: Multi-Agent Dynamic Routing & State Machine

## 1. Executive Summary
High-reliability AI workflows require deterministic orchestration over non-deterministic LLM agents. This spec describes a stateful, cyclic **Multi-Agent Orchestration Architecture** utilizing explicit state transition graphs, dynamic tool routing, and human-in-the-loop (HITL) pause mechanisms.

---

## 2. Orchestration Flow Diagram

```text
                          ┌─────────────────────┐
                          │   State Initiator   │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │    Supervisor Agent │ ◄────────────────┐
                          └──────────┬──────────┘                  │
                                     │                             │
          ┌──────────────────────────┼──────────────────────────┐  │
          │ (Code Task)              │ (Research Task)          │ (Validation Failure)
          ▼                          ▼                          │  │
┌──────────────────┐       ┌──────────────────┐                 │  │
│ Execution Agent  │       │  Research Agent  │                 │  │
└────────┬─────────┘       └─────────┬────────┘                 │  │
         │                           │                          │  │
         └─────────────┬─────────────┘                          │  │
                       │                                        │  │
                       ▼                                        │  │
            ┌─────────────────────┐                             │  │
            │ Evaluator / Guard   ├─────────────────────────────┘  │
            └──────────┬──────────┘                                │
                       │                                           │
                       ├──────────────────────┐ (Confidence < Threshold)
                       │ (Pass)               │                    │
                       ▼                      ▼                    │
            ┌──────────────────┐    ┌──────────────────┐           │
            │ Finalize State   │    │ Human-in-the-Loop│           │
            └──────────────────┘    │  Approval Node   ├───────────┘
                                    └──────────────────┘
```

## 3. Key Architectural Components
Shared Memory Context: Immutable state dictionary versioned across transitions using append-only state deltas.Evaluator Guard Node: Deterministic assertions validating code correctness, schema adherence, and hallucination bounds prior to state advancement.Human-in-the-Loop (HITL) Interrupt: State machine pauses execution state into persistence storage (Redis/PostgreSQL) when output confidence falls below $\tau = 0.82$, awaiting async operator approval.         

## 4. State Machine Transition Protocol
\delta: S_t \times A_t \rightarrow S_{t+1}$$Where $S_t$ represents current state dictionary, $A_t$ represents the agent action response, and $S_{t+1}$ represents the resulting state validated by guardrail policy $\pi_{\text{guard}}$