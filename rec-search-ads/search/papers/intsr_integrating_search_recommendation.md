# IntSR: Integrating Search and Recommendation via Query-Driven Block Unified Framework
> Industrial Deployment (Amap) | Date: 20260409

## Core Contribution
Unified generative framework integrating search and recommendation using Query-Driven Block (QDB) architecture with customized masking, reducing computational complexity while handling multiple task types.

## Key Techniques
- **Query-Driven Block (QDB)**: Unified module with customized masks for search vs recommendation
- **Query placeholder integration**: Handles disparate query modalities (ranking candidates vs natural language)
- **Unified computational structure**: Single model serves both retrieval and ranking sub-tasks
- **Customized masking**: Different attention patterns for different task types

## Results (Amap deployment)
- +3.02% GMV for digital assets
- +2.76% CTR for POI recommendation
- +5.13% accuracy for travel mode suggestion

## Industrial Implications
- Single model replaces separate search and recommendation systems
- Reduces serving infrastructure complexity and cost
- Demonstrates viability of unified search+rec in production

## Interview Points
- Q: How to unify search and recommendation? A: QDB with customized masking per task type
- Q: Benefits of unified models? A: Shared representation learning, reduced infra cost, cross-task knowledge transfer
