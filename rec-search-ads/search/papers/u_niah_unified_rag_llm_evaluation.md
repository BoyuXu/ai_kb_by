# U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-in-a-Haystack
> 2025 | Date: 20260409

## Core Contribution
Unified evaluation framework comparing LLMs and RAG methods in controlled long-context settings using extended needle-in-haystack paradigm with synthetic datasets to eliminate pre-training biases.

## Key Techniques
- **Extended NIAH configurations**: Multi-needle, long-needle, needle-in-needle
- **Synthetic Starlight Academy dataset**: Eliminates pre-training data leakage bias
- **Multiple retrieval settings evaluation**: Systematic comparison across configurations
- **Error pattern analysis**: Identifies omissions, hallucinations, self-doubt behaviors

## Key Findings
- RAG achieves 82.58% win rate over direct LLM answers
- Smaller LLMs benefit significantly more from RAG augmentation
- Critical failure modes: retrieval noise, chunk ordering effects

## Industrial Implications
- Provides principled evaluation methodology for RAG system design decisions
- Demonstrates when to use RAG vs long-context LLMs
- Identifies failure modes to mitigate in production

## Interview Points
- Q: When is RAG better than long context? A: RAG wins 82% of time, especially for smaller models
- Q: How to evaluate RAG systems? A: NIAH variants with synthetic data to avoid leakage
