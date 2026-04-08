# Retrieval Augmented Generation Evaluation: A Comprehensive Survey
> 2025 Survey | Date: 20260409

## Core Contribution
Systematic review of traditional and emerging evaluation approaches for RAG systems across performance, factual accuracy, safety, and computational efficiency dimensions.

## Evaluation Dimensions
1. **System Performance**: End-to-end answer quality (F1, EM, ROUGE)
2. **Factual Accuracy**: Faithfulness to retrieved evidence, hallucination detection
3. **Safety**: Adversarial robustness, toxicity, bias in generated answers
4. **Computational Efficiency**: Latency, throughput, resource utilization

## Key Datasets & Benchmarks
- NQ, TriviaQA, HotpotQA for QA evaluation
- FEVER for fact verification
- KILT for knowledge-intensive tasks
- RGB, RECALL for RAG-specific evaluation

## Industrial Implications
- Guides selection of appropriate metrics for production RAG systems
- Highlights gap between offline metrics and online user satisfaction
- Identifies need for safety evaluation in deployed systems

## Interview Points
- Q: How to evaluate RAG? A: Multi-dimensional: performance, factuality, safety, efficiency
- Q: Key RAG failure modes? A: Hallucination, retrieval noise, faithfulness violations
