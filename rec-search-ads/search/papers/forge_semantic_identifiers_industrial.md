# FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets
> Taobao Deployment | Date: 20260409

## Core Contribution
Large-scale benchmark for optimizing semantic identifiers in generative retrieval with 14B user interactions and multimodal features from 250M Taobao items. Proposes training-free SID quality metrics.

## Key Techniques
- **Multimodal feature integration**: Text + image + behavioral signals for SID construction
- **ID collision mitigation**: Strategies to reduce semantic ID conflicts at scale
- **Novel SID metrics**: Correlate with recommendation performance without expensive GR training
- **Training-free evaluation**: Evaluate SID quality without full model training

## Scale
- 14 billion user interactions
- 250 million items (Taobao)
- Multimodal features per item

## Results (Taobao deployment)
- +0.35% transaction count increase
- Faster online convergence

## Industrial Implications
- Training-free SID evaluation enables rapid experimentation
- Collision mitigation critical at billion-item scale
- Multimodal SIDs capture richer item semantics

## Interview Points
- Q: How to evaluate SID quality cheaply? A: Training-free metrics that correlate with GR performance
- Q: SID challenges at scale? A: ID collisions, multimodal integration, evaluation cost
