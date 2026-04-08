# LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs
> 2024 | Date: 20260409

## Core Contribution
Leverages long-context LLMs to improve RAG by using 4K-token retrieval units (30x longer than traditional 100-word chunks), reducing Wikipedia index from 22M to 700K units.

## Key Techniques
- **Long retrieval units (~4K tokens)**: Preserves comprehensive context within each unit
- **Zero-shot answer generation**: Uses GPT-4o/Gemini-1.5-Pro directly without fine-tuning
- **Dramatic index reduction**: 22M → 700K units (31x compression)
- **Light retriever / Heavy reader paradigm flip**: Shifts computation burden to the reader

## Results
- NQ: Answer recall 52% → 71% (+19%)
- HotpotQA: Answer recall 47% → 72% (+25%)
- Zero-shot EM scores competitive with fine-tuned state-of-the-art

## Industrial Implications
- Simplifies RAG pipeline by reducing retriever complexity
- Leverages existing long-context LLMs without additional training
- Reduces storage and indexing costs significantly

## Interview Points
- Q: How does chunk size affect RAG? A: Larger chunks (4K) dramatically improve recall with long-context LLMs
- Q: Light retriever vs heavy retriever? A: With long-context readers, light retrievers + large chunks win
