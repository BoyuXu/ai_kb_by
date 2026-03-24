#!/usr/bin/env python3
"""
Knowledge Base Round 3 Restructuring - File Rename Script
"""

import os
import re
import subprocess
import json

KB_ROOT = os.path.expanduser("~/Documents/ai-kb")

def read_title(filepath):
    """Read the first heading from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()
    except:
        pass
    return None

def clean_filename(name, max_len=55):
    """Clean a string for use as filename."""
    # Remove problematic chars
    name = re.sub(r'[<>"|?*]', '', name)
    # Keep colons in Chinese context but remove at end
    name = name.strip().rstrip(':：-_ ')
    if len(name) > max_len:
        name = name[:max_len].rstrip('_- ')
    return name

# Manual override mappings for problematic files
SYNTHESIS_OVERRIDES = {
    'rec-sys/synthesis/01_ctr_models_deep_dive.md': 'CTR模型深度解析.md',
    'rec-sys/synthesis/03_feature_engineering.md': '推荐系统特征工程深度笔记.md',
    'rec-sys/synthesis/05_industry_ranking_papers.md': '精排工业界精华_AlgoNotes精读.md',
    'rec-sys/synthesis/05_ranking_deep.md': '精排模型进阶深度解析.md',
    'rec-sys/synthesis/06_industry_recall_papers.md': '召回系统工业界最佳实践.md',
    'rec-sys/synthesis/07_causal_inference.md': '推荐系统因果推断.md',
    'rec-sys/synthesis/08_rerank_diversity.md': '重排与多样性.md',
    'rec-sys/synthesis/09_feature_store_practice.md': '特征工程与Feature_Store实践.md',
    'rec-sys/synthesis/00_overview.md': '推荐系统全链路架构概览.md',
    'rec-sys/synthesis/00_personal_notes.md': 'Boyu个人学习档案.md',
    'search/synthesis/01_search_ranking.md': '搜索排序专项笔记.md',
    # Date-prefixed synthesis
    'ads/synthesis/20260320_ads_budget_pacing.md': '广告预算Pacing算法全景.md',
    'ads/synthesis/20260321_ad_bidding_evolution.md': '广告出价体系_从手动规则到RL自动出价.md',
    'ads/synthesis/20260322_ad_bias_correction_trilogy.md': '广告系统偏差治理三部曲.md',
    'ads/synthesis/20260323_ads_synthesis.md': '广告系统综合总结.md',
    'ads/synthesis/ads_ranking_evolution.md': '广告排序系统演进路线图.md',
    'ads/synthesis/auto_bidding_evolution.md': 'AutoBidding技术演进_从规则到RL.md',
    'ads/synthesis/mixing_ranking_evolution.md': '广告系统混排演进路线.md',
    'ads/synthesis/llm_integration_framework.md': '广告系统LLM集成框架.md',
    'ads/synthesis/std_ads_attribution.md': '广告效果归因.md',
    'ads/synthesis/std_ads_bidding_landscape.md': '广告出价体系全景.md',
    'ads/synthesis/std_ads_cold_start.md': '广告系统冷启动.md',
    'ads/synthesis/std_ads_creative_optimization.md': '广告创意优化.md',
    'ads/synthesis/std_ads_ctr_cvr_calibration.md': '广告CTR_CVR预估与校准.md',
    'ads/synthesis/std_ads_multi_objective.md': '广告系统多目标优化.md',
    'ads/synthesis/std_ads_rtb_architecture.md': '广告系统RTB架构全景.md',
    # rec-sys synthesis
    'rec-sys/synthesis/20260321_semantic_id_generative_retrieval.md': 'SemanticID与生成式检索.md',
    'rec-sys/synthesis/20260322_generative_rec_paradigm_comparison.md': '生成式推荐范式对比.md',
    'rec-sys/synthesis/20260322_semantic_id_full_picture.md': 'SemanticID从论文到Spotify部署.md',
    'rec-sys/synthesis/20260323_generative_rec_full_spectrum.md': '生成式推荐完整技术图谱.md',
    'rec-sys/synthesis/20260323_rec_sys_synthesis.md': '推荐系统综合总结.md',
    'rec-sys/synthesis/20260323_recommendation_scaling_law_wukong.md': '推荐系统ScalingLaw_Wukong.md',
    'rec-sys/synthesis/llm_integration_framework.md': '推荐系统LLM集成框架.md',
    'rec-sys/synthesis/std_rec_cold_start.md': '推荐系统冷启动.md',
    'rec-sys/synthesis/std_rec_embedding_learning.md': 'Embedding学习_推荐系统表示基石.md',
    'rec-sys/synthesis/std_rec_feature_engineering.md': '推荐系统特征工程体系.md',
    'rec-sys/synthesis/std_rec_graph_neural_network.md': '图神经网络在推荐中的应用.md',
    'rec-sys/synthesis/std_rec_online_experiment.md': '推荐广告AB测试与在线实验.md',
    'rec-sys/synthesis/std_rec_ranking_evolution.md': '推荐系统排序范式演进.md',
    'rec-sys/synthesis/std_rec_recall_evolution.md': '推荐系统召回范式演进.md',
    'rec-sys/synthesis/std_rec_rerank_diversity.md': '推荐系统重排与多样性.md',
    'rec-sys/synthesis/std_rec_user_behavior_modeling.md': '用户行为序列建模.md',
    # search synthesis
    'search/synthesis/20260320_hybrid_retrieval_evolution.md': '混合检索的工业化演进.md',
    'search/synthesis/20260321_sparse_dense_retrieval.md': '稀疏检索vs稠密检索.md',
    'search/synthesis/20260322_sparse_vs_dense_retrieval_decision.md': '稀疏vs密集检索决策.md',
    'search/synthesis/20260323_retrieval_triangle_dense_sparse_late.md': '检索三角_Dense_Sparse_LateInteraction.md',
    'search/synthesis/llm_integration_framework.md': '搜索系统LLM集成框架.md',
    'search/synthesis/std_search_hybrid_retrieval.md': '混合检索融合_多路召回实践.md',
    'search/synthesis/std_search_learning_to_rank.md': 'LearningToRank搜索排序三大范式.md',
    'search/synthesis/std_search_query_understanding.md': '搜索Query理解.md',
    'search/synthesis/std_search_reranker_evolution.md': '搜索Reranker演进.md',
    'search/synthesis/std_search_retrieval_triangle.md': '检索三角形深析.md',
    'search/synthesis/std_search_temporal_graph.md': '搜索时间序列图.md',
    # llm-infra synthesis
    'llm-infra/synthesis/20260320_kv_cache_compression.md': 'KVCache压缩技术全景.md',
    'llm-infra/synthesis/20260320_moe_disaggregated_inference.md': 'MoE推理解耦架构.md',
    'llm-infra/synthesis/20260321_flashattention3_llm_infra.md': 'FlashAttention3与LLM推理基础设施.md',
    'llm-infra/synthesis/20260321_grpo_rl_alignment.md': 'GRPO大模型推理RL算法.md',
    'llm-infra/synthesis/20260322_llm_efficiency_trifecta.md': 'LLM推理效率三角.md',
    'llm-infra/synthesis/20260323_kvcache_and_llm_inference_optimization.md': 'KVCache与LLM推理优化全景.md',
    'llm-infra/synthesis/20260323_rlvr_vs_rlhf_posttraining.md': 'RLVR_vs_RLHF后训练路线.md',
    'llm-infra/synthesis/std_llm_alignment_evolution.md': 'LLM对齐方法演进.md',
    'llm-infra/synthesis/std_llm_fine_tuning.md': 'LLM微调技术.md',
    'llm-infra/synthesis/std_llm_inference_optimization.md': 'LLM推理优化完整版.md',
    'llm-infra/synthesis/std_llm_moe_architecture.md': 'MoE架构设计.md',
    'llm-infra/synthesis/std_llm_pretraining.md': 'LLM预训练技术演进.md',
    'llm-infra/synthesis/std_llm_rag_system.md': 'RAG系统全景.md',
    'llm-infra/synthesis/std_llm_serving_system.md': 'LLMServing系统实践.md',
    # cross-domain synthesis
    'cross-domain/synthesis/20260320_llm_recommendation_retrieval.md': 'LLM赋能推荐召回.md',
    'cross-domain/synthesis/20260322_unified_model_search_rec.md': '统一模型搜索推荐_Spotify_ULM.md',
    'cross-domain/synthesis/20260322_weekly_rec_ads_synthesis.md': '推荐广告系统周总结_0316_0322.md',
    'cross-domain/synthesis/20260324_engineering_practices.md': '工程实践_从论文到生产.md',
    'cross-domain/synthesis/20260324_rec_ads_algo_evolution.md': '广告推荐系统算法演进脉络.md',
    'cross-domain/synthesis/std_cross_bias_governance.md': '偏差治理体系.md',
    'cross-domain/synthesis/std_cross_generative_paradigm.md': '生成式范式统一视角.md',
    'cross-domain/synthesis/std_cross_long_sequence.md': '长序列处理_推荐搜索LLM共同挑战.md',
    'cross-domain/synthesis/std_cross_ml_fundamentals.md': '机器学习基础面试必备.md',
    'cross-domain/synthesis/std_cross_multi_objective_unified.md': '多目标优化统一框架.md',
    'cross-domain/synthesis/std_cross_system_design.md': '系统设计面试要点.md',
    # interview synthesis
    'interview/synthesis/20260324_interview_storytelling.md': '面试项目讲故事_30个案例.md',
    'interview/synthesis/card_001_multi_task_learning.md': '多任务学习_MMoE_PLE.md',
    'interview/synthesis/card_002_dual_tower_recall.md': '双塔召回模型.md',
    'interview/synthesis/card_003_ocpc_ocpa_bidding.md': 'oCPC_oCPA智能出价系统.md',
    'interview/synthesis/card_004_query_rewriting_rag.md': 'QueryRewriting与RAG优化.md',
    'interview/synthesis/card_005_recommendation_pipeline_overview.md': '推荐系统全链路串联.md',
}

def get_synthesis_mapping():
    """Get synthesis rename mapping from overrides."""
    mapping = {}
    for old_rel, new_basename in SYNTHESIS_OVERRIDES.items():
        dirname = os.path.dirname(old_rel)
        new_rel = os.path.join(dirname, new_basename)
        mapping[old_rel] = new_rel
    return mapping

def title_to_paper_filename(title, max_len=57):
    """Convert paper title to English filename."""
    if not title:
        return None
    
    # Remove Chinese characters
    clean = re.sub(r'[\u4e00-\u9fff\uff00-\uffef]+', '', title)
    # Remove special chars but keep basic alphanumeric
    clean = re.sub(r'[^\w\s\-]', ' ', clean)
    clean = clean.strip()
    
    if not clean or len(clean) < 5:
        # Title is mostly Chinese, try to extract English parts
        english_parts = re.findall(r'[A-Za-z][A-Za-z0-9\-]+', title)
        if english_parts:
            clean = ' '.join(english_parts)
        else:
            return None
    
    # Convert to snake_case
    clean = re.sub(r'[\s\-:,./]+', '_', clean)
    clean = re.sub(r'_+', '_', clean)
    clean = clean.strip('_')
    
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip('_')
    
    return clean

def get_paper_files():
    """Get all paper files needing rename (date-prefixed)."""
    dirs = ['ads/papers', 'rec-sys/papers', 'search/papers']
    files = []
    for d in dirs:
        full_dir = os.path.join(KB_ROOT, d)
        if os.path.isdir(full_dir):
            for f in sorted(os.listdir(full_dir)):
                if f.endswith('.md') and re.match(r'2026\d+_', f):
                    files.append(os.path.join(d, f))
    return files

def generate_paper_mapping():
    """Generate paper rename mapping."""
    mapping = {}
    used_names = {}  # track name collisions per directory
    
    files = get_paper_files()
    for rel_path in files:
        full_path = os.path.join(KB_ROOT, rel_path)
        title = read_title(full_path)
        dirname = os.path.dirname(rel_path)
        
        new_name = title_to_paper_filename(title)
        if not new_name:
            continue
        
        # Handle duplicates in same directory
        key = (dirname, new_name)
        if key in used_names:
            # Skip duplicate - same paper processed twice
            continue
        used_names[key] = rel_path
        
        new_rel = os.path.join(dirname, new_name + '.md')
        if new_rel != rel_path:
            mapping[rel_path] = new_rel
    
    return mapping

def execute_renames(mapping, label):
    """Execute git mv for a mapping."""
    success = 0
    skip = 0
    for old, new in sorted(mapping.items()):
        old_full = os.path.join(KB_ROOT, old)
        new_full = os.path.join(KB_ROOT, new)
        if os.path.exists(old_full) and not os.path.exists(new_full):
            result = subprocess.run(['git', 'mv', old_full, new_full], 
                                   cwd=KB_ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                success += 1
            else:
                print(f"  ❌ git mv failed: {os.path.basename(old)} → {result.stderr.strip()}")
                skip += 1
        elif os.path.exists(new_full):
            print(f"  ⚠️ SKIP (exists): {os.path.basename(new)}")
            skip += 1
        else:
            print(f"  ❌ SKIP (missing): {os.path.basename(old)}")
            skip += 1
    print(f"\n{label}: {success} renamed, {skip} skipped")
    return success

def update_references(syn_map, paper_map):
    """Update all cross-references in MD files."""
    # Build basename mapping
    basename_map = {}
    for old, new in {**syn_map, **paper_map}.items():
        old_base = os.path.basename(old)
        new_base = os.path.basename(new)
        if old_base != new_base:
            basename_map[old_base] = new_base
    
    # Find all MD files (excluding .git and repos)
    all_md = []
    for root, dirs, files in os.walk(KB_ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git', 'repos', 'repo')]
        for f in files:
            if f.endswith('.md'):
                all_md.append(os.path.join(root, f))
    
    updated = 0
    for md_file in all_md:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        original = content
        for old_base, new_base in basename_map.items():
            content = content.replace(old_base, new_base)
        
        if content != original:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            updated += 1
    
    print(f"Updated references in {updated} files")
    return updated

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'preview'
    
    if mode == 'preview-synthesis':
        mapping = get_synthesis_mapping()
        print(f"=== Synthesis Rename Preview ({len(mapping)} files) ===")
        for old, new in sorted(mapping.items()):
            print(f"  {os.path.basename(old):50s} → {os.path.basename(new)}")
    
    elif mode == 'preview-papers':
        mapping = generate_paper_mapping()
        print(f"=== Papers Rename Preview ({len(mapping)} files) ===")
        for old, new in sorted(mapping.items()):
            print(f"  {os.path.basename(old):60s} → {os.path.basename(new)}")
    
    elif mode == 'execute-synthesis':
        mapping = get_synthesis_mapping()
        execute_renames(mapping, "Synthesis")
        with open(os.path.join(KB_ROOT, 'scripts/synthesis_rename_map.json'), 'w') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    elif mode == 'execute-papers':
        mapping = generate_paper_mapping()
        execute_renames(mapping, "Papers")
        with open(os.path.join(KB_ROOT, 'scripts/paper_rename_map.json'), 'w') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    elif mode == 'update-refs':
        syn_map = {}
        paper_map = {}
        try:
            with open(os.path.join(KB_ROOT, 'scripts/synthesis_rename_map.json')) as f:
                syn_map = json.load(f)
        except: pass
        try:
            with open(os.path.join(KB_ROOT, 'scripts/paper_rename_map.json')) as f:
                paper_map = json.load(f)
        except: pass
        update_references(syn_map, paper_map)
    
    else:
        print("Usage: python rename_kb.py [preview-synthesis|preview-papers|execute-synthesis|execute-papers|update-refs]")
