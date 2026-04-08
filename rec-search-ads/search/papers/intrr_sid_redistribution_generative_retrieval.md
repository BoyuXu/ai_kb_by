# IntRR: Integrating SID Redistribution and Length Reduction in Generative Retrieval
> 2025 | Date: 20260409

## Core Contribution
Addresses inefficiencies in semantic identifier (SID) generation by integrating objective-aligned redistribution and structural length reduction via Recursive-Assignment Network (RAN).

## Key Techniques
- **Recursive-Assignment Network (RAN)**: Processes SID hierarchy recursively without flattening
- **Adaptive SID Redistribution**: Uses unique IDs as collaborative anchors for semantic weight refinement
- **Fixed token cost**: One token per item, eliminating multi-pass autoregressive bottleneck
- **Dynamic semantic weight refinement**: Adjusts importance across hierarchical codebook layers

## Technical Details
```
Traditional GR: Item → Multi-token SID → Autoregressive generation (slow)
IntRR:          Item → RAN → Single-token prediction (fast)
```

## Results
- Consistent lowest latency across search spaces
- Superior accuracy compared to multi-token SID approaches
- Scales well with vocabulary size

## Industrial Implications
- Eliminates autoregressive bottleneck in generative retrieval
- Fixed token cost enables predictable serving latency
- Practical for large-scale industrial retrieval systems

## Interview Points
- Q: Bottleneck in generative retrieval? A: Multi-token autoregressive SID generation
- Q: How to speed up GR? A: RAN for single-token prediction with SID redistribution
