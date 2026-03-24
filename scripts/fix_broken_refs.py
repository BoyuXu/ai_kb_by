#!/usr/bin/env python3
"""Fix broken references to deleted duplicate papers."""

import os
import re

KB = os.path.expanduser("~/Documents/ai-kb")

# Mapping from deleted files to their kept replacements (new names after rename)
DELETED_TO_KEPT = {
    # ads duplicates (old deleted name -> new name of kept file)
    '20260319_ctr-prediction-survey.md': 'A_Comprehensive_Survey_on_Advertising_Click_Through_Rate.md',
    '20260322_autobid_rl_ad_bidding.md': 'AutoBid_Reinforcement_Learning_for_Automated_Ad_Bidding_w.md',
    '20260312_CADET_広告CTR_Decoder_Only_Transformer.md': 'CADET_Context_Conditioned_Ads_CTR_Prediction_With_a_Decod.md',
    '20260312_CADET_\u5e7f\u544aCTR_Decoder_Only_Transformer.md': 'CADET_Context_Conditioned_Ads_CTR_Prediction_With_a_Decod.md',
    '20260322_counterfactual_unbiased_ad_ranking.md': 'Counterfactual_Learning_for_Unbiased_Ad_Ranking_in_Indust.md',
    '20260312_IDProxy_冷启動CTR_小红书多模态LLM.md': 'IDProxy_Cold_Start_CTR_Prediction_for_Ads_and_Recommendat.md',
    '20260312_IDProxy_\u51b7\u542f\u52a8CTR_\u5c0f\u7ea2\u4e66\u591a\u6a21\u6001LLM.md': 'IDProxy_Cold_Start_CTR_Prediction_for_Ads_and_Recommendat.md',
    '20260322_llm_ad_creative_generation.md': 'LLM_Enhanced_Ad_Creative_Generation_and_Optimization_for.md',
    '20260322_multi_objective_online_advertising.md': 'Multi_Objective_Optimization_for_Online_Advertising_Balan.md',
    '20260319_no-regret-autobidding-first-price-auction.md': 'No_Regret_Online_Autobidding_Algorithms_in_First_price_Au.md',
    '20260319_optimal-boost-autobidding-publisher-quality.md': 'Optimal_Boost_Design_for_Auto_bidding_Mechanism_with_Publ.md',
    # rec-sys duplicates
    '20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md': 'A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md',
    '20260322_spotify_unified_lm_search_rec.md': 'A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md',
    '20260322_spotify_semantic_id_podcast.md': 'Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md',
    '20260322_diffgrm_diffusion_generative_rec.md': 'DiffGRM_Diffusion_based_Generative_Recommendation_Model.md',
    '20260322_gems_long_sequence_generative_rec.md': 'GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md',
    '20260322_interplay_conversational_rec.md': 'Interplay_Training_Independent_Simulators_for_Reference_F.md',
    '20260319_kgmel-knowledge-graph-multimodal-entity-linking.md': 'KGMEL_Knowledge_Graph_Enhanced_Multimodal_Entity_Linking.md',
    '20260322_variable_length_semantic_id.md': 'Variable_Length_Semantic_IDs_for_Recommender_Systems.md',
    # search duplicates
    '20260323_dllm-searcher_adapting_diffusion_large_language_mod.md': 'DLLM_Searcher_Adapting_Diffusion_Large_Language_Model_for.md',
    '20260321_dense-retrieval-vs-sparse-retrieval-a-unified-evaluation-framework-for-large-scale-product-search.md': 'Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md',
    '20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md': 'Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md',
    '20260322_intent_aware_query_reformulation.md': 'Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md',
    '20260319_legalmalr-multi-agent-chinese-statute-retrieval.md': 'LegalMALR_Multi_Agent_Query_Understanding_LLM_Based_Reran.md',
    '20260322_location_aware_embedding_geotargeting.md': 'Location_Aware_Embedding_for_Geotargeting_in_Sponsored_Se.md',
    '20260322_splade_v3_sparse_retrieval.md': 'SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M.md',
    '20260319_multimodal-document-retrieval-survey.md': 'Unlocking_Multimodal_Document_Intelligence_Visual_Documen.md',
    '20260319_multimodal-visual-document-retrieval-survey.md': 'Unlocking_Multimodal_Document_Intelligence_Visual_Documen.md',
}

# Also need to verify what the actual new names are
# Let me check which files actually exist now
def get_actual_papers():
    """Get all current paper filenames by directory."""
    result = {}
    for d in ['ads/papers', 'rec-sys/papers', 'search/papers']:
        full = os.path.join(KB, d)
        if os.path.isdir(full):
            result[d] = set(os.listdir(full))
    return result

def fix_refs():
    actual = get_actual_papers()
    
    # Find all md files
    all_md = []
    for root, dirs, files in os.walk(KB):
        dirs[:] = [d for d in dirs if d not in ('.git', 'repos', 'repo')]
        for f in files:
            if f.endswith('.md'):
                all_md.append(os.path.join(root, f))
    
    updated = 0
    for md_file in all_md:
        try:
            with open(md_file, 'r') as f:
                content = f.read()
        except: continue
        
        original = content
        for old, new in DELETED_TO_KEPT.items():
            if old in content:
                content = content.replace(old, new)
        
        if content != original:
            with open(md_file, 'w') as f:
                f.write(content)
            updated += 1
            print(f"  Fixed: {os.path.relpath(md_file, KB)}")
    
    print(f"\nFixed {updated} files")

if __name__ == '__main__':
    # First check what files actually exist now
    actual = get_actual_papers()
    
    # Verify targets exist
    missing = []
    for old, new in DELETED_TO_KEPT.items():
        # Determine which dir
        found = False
        for d, files in actual.items():
            if new in files:
                found = True
                break
        if not found:
            missing.append((old, new))
    
    if missing:
        print("WARNING: Target files not found:")
        for old, new in missing:
            print(f"  {old} -> {new} (MISSING)")
    
    fix_refs()
