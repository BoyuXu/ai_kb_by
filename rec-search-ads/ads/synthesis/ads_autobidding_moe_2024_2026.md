# Ads Auto-Bidding & MoE Research Notes (2024-2026)

> 10 papers covering auto-bidding optimization, CVR prediction, and Mixture-of-Experts advances.
> Related concepts: [[multi_objective_optimization]], [[sequence_modeling_evolution]]

---

## Part 1: Auto-Bidding & Ads Optimization

### AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising (arxiv 2602.22650)
- **Problem**: Cross-channel advertising requires budget allocation across channels with distinct behavioral patterns; existing auto-bidding methods struggle with multi-channel coordination.
- **Method**: Hierarchical framework with (1) high-level diffusion-model-based generative planner for budget/constraint allocation across channels, and (2) low-level control-based bidding algorithm combining historical knowledge with real-time signals for individual impression bids.
- **Key Innovation**: First to apply diffusion models as generative planners for cross-channel budget allocation; constraint enforcement mechanism ensures compliance while maintaining adaptability.
- **Results**: 13.57% increase in overall return vs. baselines in both offline large-scale datasets and online A/B tests.
- **Industry**: Deployed at scale (authors from industry); WWW 2026.
- **Keywords**: cross-channel bidding, diffusion planning, hierarchical RL, budget allocation, auto-bidding

### JD-BP: A Joint-Decision Generative Framework for Auto-Bidding and Pricing (arxiv 2604.05845)
- **Problem**: Auto-bidding under KPI constraints (ROI, budget) suffers from model prediction errors and feedback latency, causing deviation from ex-post optimality. Bidding and pricing are traditionally decoupled.
- **Method**: Generative decision framework that jointly outputs a bid value and a pricing correction term (additive to GSP payment rule). Uses memory-less Return-to-Go to encourage future value maximization while pricing correction handles cumulated constraint bias. Trajectory augmentation generates joint bidding-pricing trajectories from arbitrary base policies.
- **Key Innovation**: First joint bidding+pricing generative framework; memory-less RTG decouples historical constraint violations from future decisions.
- **Results**: Demonstrates improved allocation efficiency over decoupled baselines (specific metrics in full paper).
- **Industry**: JD.com advertising platform.
- **Keywords**: auto-bidding, pricing correction, generative model, GSP auction, trajectory augmentation

### DenoiseBid: Uncertainty Quantification of Click and Conversion Estimates for Autobidding (arxiv 2603.01825)
- **Problem**: CTR/CVR model predictions are noisy/uncertain, causing autobidding algorithms to generate suboptimal bids, wasting advertiser budgets.
- **Method**: Bayesian approach that replaces noisy CTR/CVR point estimates with posterior expectations conditioned on model predictions and a prior distribution recovered from historical auction data. Corrects bids to be more efficient under uncertainty.
- **Key Innovation**: Principled uncertainty quantification for autobidding; treats CTR/CVR noise as a denoising problem with Bayesian posterior correction rather than point-estimate optimization.
- **Results**: Validated on synthetic, iPinYou, and BAT datasets showing improved bidding efficiency.
- **Industry**: E-commerce auction platforms.
- **Keywords**: uncertainty quantification, Bayesian bidding, CTR/CVR denoising, autobidding, auction efficiency

### MEBS: Multi-task End-to-end Bid Shading for Multi-slot Display Ads (arxiv 2403.02607)
- **Problem**: Multi-slot display advertising (multiple ads shown in a list) creates varying cost-effectiveness across positions; advertisers need bid price adjustment to win economical positions.
- **Method**: Three-model architecture: win rate model, pCTR calibration model, and shading ratio model. Estimates optimal shading ratio by modeling how bid adjustment affects win rate and CTR. Multi-task learning addresses data sparsity; end-to-end optimization directly maximizes advertiser surplus (cost saved).
- **Key Innovation**: First end-to-end bid shading method for multi-slot scenarios; jointly models position-dependent win probability and CTR calibration.
- **Results**: +7.01% GMV, +7.42% ROI, +3.26% ad buy count in online experiments.
- **Industry**: Deployed in production (CIKM 2023).
- **Keywords**: bid shading, multi-slot display, multi-task learning, win rate modeling, surplus maximization

### CVR Prediction Survey: Modeling Techniques, Performance Evaluation and Future Directions (arxiv 2512.01171)
- **Problem**: Comprehensive review needed for CVR prediction methods which are fragmented across different technical approaches.
- **Method**: Classifies state-of-the-art CVR prediction models into six categories by underlying technique. Analyzes framework, advantages/disadvantages of each, and how they serve CVR prediction.
- **Key Innovation**: First systematic taxonomy of CVR prediction into 6 categories; identifies that reported performance evaluations across prior studies are inconsistent/non-unanimous.
- **Results**: Survey paper; identifies future directions: semantics-enriched CVR, attribution-enhanced CVR, debiased CVR, and joint CTR+CVR modeling.
- **Industry**: Cross-industry applicability.
- **Keywords**: CVR prediction, survey, delayed feedback, multi-task learning, attribution modeling

### Practical Multi-Task Learning for Rare Conversions in Ad Tech (arxiv 2507.20161)
- **Problem**: Rare conversion events (<1% rate) in online advertising are hard to predict accurately due to extreme data sparsity.
- **Method**: Multi-task learning approach that classifies conversions as "rare" or "frequent" based on historical statistics. Shared representations across all conversion signals with separate task towers for each type, enabling knowledge transfer from frequent to rare events.
- **Key Innovation**: Simple, production-ready MTL design specifically targeting the rare-vs-frequent conversion split; demonstrates that shared bottom layers effectively transfer knowledge to data-sparse rare conversion tasks.
- **Results**: +0.69% AUC lift offline, -2% CPA (Cost per Action) online. Fully deployed to production.
- **Industry**: Production deployment; RecSys 2025.
- **Keywords**: multi-task learning, rare conversions, CVR prediction, knowledge transfer, production system

---

## Part 2: Mixture-of-Experts (MoE)

### MoE Comprehensive Survey: Algorithms, Theory, and Applications (arxiv 2503.07137)
- **Problem**: MoE models have become critical for scaling LLMs efficiently, but lack a unified survey covering algorithms, theory, and applications.
- **Method**: Covers MoE design (gating functions, expert networks, routing mechanisms, training strategies, system design) and its role in continual learning, meta-learning, multi-task learning, and RL. Reviews theoretical foundations and CV/NLP applications.
- **Key Innovation**: Most comprehensive MoE survey to date; bridges algorithm design, theoretical understanding, and practical application across multiple ML paradigms.
- **Results**: Survey paper; synthesizes landscape of MoE research across domains.
- **Industry**: Cross-domain reference (updated through Jan 2026, v4).
- **Keywords**: MoE survey, gating mechanism, routing, sparse models, efficient scaling

### Advancing Expert Specialization for Better MoE (arxiv 2505.22323)
- **Problem**: Auxiliary load balancing loss in MoE leads to expert overlap and overly uniform routing, hindering expert specialization and degrading post-training performance.
- **Method**: Two complementary objectives: (1) orthogonality loss encouraging experts to process distinct token types, and (2) variance loss encouraging more discriminative routing decisions. Compatible with existing auxiliary loss via gradient-level analysis.
- **Key Innovation**: No architectural changes needed; simple loss additions improve specialization up to 23.79% over baselines while maintaining load balance. Gradient-level proof of compatibility with auxiliary loss.
- **Results**: Up to 23.79% improvement over classic MoE baselines with auxiliary loss; maintains load balancing in downstream tasks.
- **Industry**: General LLM training.
- **Keywords**: expert specialization, orthogonality loss, routing discrimination, load balancing, MoE training

### eMoE: Task-aware Memory Efficient MoE Inference (arxiv 2503.06823)
- **Problem**: MoE models require loading all experts into memory despite only activating a subset, creating massive memory overhead during inference.
- **Method**: Predicts and loads only required experts based on recurrent routing patterns; invokes expert predictor every few prompts (not each) to reduce loading latency; skips predictions for tasks less sensitive to routing accuracy. Task-aware scheduling considers SLOs, output lengths, and loading latencies.
- **Key Innovation**: Task-aware expert prediction that reduces prediction frequency while maintaining accuracy; combines routing pattern prediction with SLO-aware scheduling.
- **Results**: Up to 80% memory reduction while maintaining accuracy; up to 17% inference latency reduction vs. existing systems.
- **Industry**: General MoE inference deployment.
- **Keywords**: MoE inference, memory efficiency, expert prediction, task-aware scheduling, SLO optimization

### MoE Routing Testbed: Studying Expert Specialization and Routing Behavior at Small Scale (arxiv 2604.07030)
- **Problem**: Understanding routing dynamics in sparse MoE is difficult at production scale; hard to measure whether experts truly specialize.
- **Method**: Proposes a controlled testbed pairing a data mix of clearly distinguishable domains with a reference router that prescribes ideal routing. Enables quantifiable measurement of expert specialization at small scale using realistic data.
- **Key Innovation**: First standardized testbed for MoE routing research; discovers that balancing scope is the crucial factor enabling specialization while maintaining high expert utilization. Confirms findings generalize to 35x larger models.
- **Results**: Identifies balancing scope as key factor; results validated at 35x scale.
- **Industry**: Research tool for MoE development.
- **Keywords**: routing testbed, expert specialization, load balancing scope, MoE analysis, scalable insights

---

## Synthesis: Key Trends in Ads/MoE Research (2024-2026)

**Auto-Bidding is going generative.** Both AHBid and JD-BP employ generative models (diffusion, trajectory generation) to replace traditional optimization-based bidding. This reflects a broader trend: treating bidding as sequence generation enables capturing complex temporal dependencies and multi-agent dynamics that constrained optimization cannot easily model.

**Uncertainty-aware and denoising approaches are maturing.** DenoiseBid's Bayesian correction of CTR/CVR estimates and MEBS's calibration models both address the fundamental problem that prediction models are imperfect. The field is moving from "better point estimates" to "better decisions under uncertainty."

**Multi-task learning dominates production systems.** MEBS, the Rare Conversions paper, and the CVR Survey all converge on MTL as the practical workhorse for ads: shared representations transfer knowledge across sparse signals, and task-specific towers handle distributional differences. The key design decision is the granularity of task splitting.

**MoE research is solving the specialization-balance tradeoff.** The orthogonality/variance losses (Expert Specialization paper) and the Routing Testbed both tackle the same core tension: load balancing kills specialization, but without balancing, training collapses. The emerging consensus is that balancing scope and gradient-compatible specialization losses are the path forward.

**Efficiency at inference time is a first-class concern.** eMoE's 80% memory reduction shows that deploying MoE models requires fundamentally rethinking which experts to load, not just which to activate. This is directly relevant to ads systems where latency SLOs are strict.

**Cross-channel and multi-slot complexity is the new frontier.** AHBid's hierarchical cross-channel design and MEBS's multi-slot bid shading reflect that real advertising platforms have moved far beyond single-auction optimization. The systems must jointly optimize across channels, positions, and time horizons.
