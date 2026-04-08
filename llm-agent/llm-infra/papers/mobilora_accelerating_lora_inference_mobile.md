# MobiLoRA: Accelerating LoRA-based LLM Inference on Mobile Devices
> ACL 2025 | Date: 20260409

## Core Contribution
Optimizes LoRA-based LLM inference on mobile devices through intelligent KV cache management combining semantic and system-level context awareness. Achieves 57.6% acceleration.

## Key Techniques
- **Similarity-aware delta encoding**: Compresses KV cache across LoRA adapters by encoding deltas
- **Context-aware cache management**: Combines semantic (attention patterns) and system-level (app state, memory) signals
- **Adaptive cache eviction**: Retains important KV entries based on dual-context signals
- **Shared-prefix prompt optimization**: Reuses common prefix computations across adapters

## Technical Details
- Delta encoding exploits similarity between base model and adapter KV caches
- System-level context includes foreground/background state, available memory
- 57.6% latency reduction on mobile hardware

## Industrial Implications
- Enables on-device personalization without cloud dependency
- Critical for privacy-preserving recommendation on mobile
- Applicable to edge-deployed LLM serving

## Interview Points
- Q: How to serve LoRA models on mobile? A: KV cache compression + context-aware eviction
- Q: What signals drive cache eviction? A: Both semantic attention patterns and system resource state
