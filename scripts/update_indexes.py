#!/usr/bin/env python3
"""Update INDEX.md files for each domain."""

import os

KB = os.path.expanduser("~/Documents/ai-kb")

def list_md_files(dirpath):
    """List all .md files in a directory."""
    if not os.path.isdir(dirpath):
        return []
    return sorted([f for f in os.listdir(dirpath) if f.endswith('.md')])

def read_title(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('# '):
                    return line.strip()[2:]
    except: pass
    return os.path.basename(filepath).replace('.md', '')

def generate_index(domain_dir, domain_name, description):
    """Generate INDEX.md content for a domain."""
    papers = list_md_files(os.path.join(KB, domain_dir, 'papers'))
    synthesis = list_md_files(os.path.join(KB, domain_dir, 'synthesis'))
    practices = list_md_files(os.path.join(KB, domain_dir, 'practices'))
    cards_dir = os.path.join(KB, domain_dir, 'cards')
    cards = list_md_files(cards_dir) if os.path.isdir(cards_dir) else []
    
    total = len(papers) + len(synthesis) + len(practices) + len(cards)
    
    lines = [f"# {domain_name}知识库导航\n"]
    lines.append(f"## 📊 领域概览\n")
    lines.append(f"| 分类 | 文档数 | 描述 |")
    lines.append(f"|------|--------|------|")
    if papers:
        lines.append(f"| **Papers** | {len(papers)}篇 | 学术论文笔记 |")
    if synthesis:
        lines.append(f"| **Synthesis** | {len(synthesis)}篇 | 提炼总结 |")
    if practices:
        lines.append(f"| **Practices** | {len(practices)}篇 | 工业实践 |")
    if cards:
        lines.append(f"| **Cards** | {len(cards)}篇 | 知识卡片 |")
    lines.append(f"| **总计** | {total}篇 | - |")
    lines.append(f"\n---\n")
    
    # Synthesis section
    if synthesis:
        lines.append(f"## 📝 Synthesis 总结文档\n")
        for f in synthesis:
            title = read_title(os.path.join(KB, domain_dir, 'synthesis', f))
            lines.append(f"- [{title}](./synthesis/{f})")
        lines.append("")
    
    # Papers section
    if papers:
        lines.append(f"## 📚 论文笔记\n")
        for f in papers:
            title = read_title(os.path.join(KB, domain_dir, 'papers', f))
            short_title = title[:60] + '...' if len(title) > 60 else title
            lines.append(f"- [{short_title}](./papers/{f})")
        lines.append("")
    
    # Practices section
    if practices:
        lines.append(f"## 🏭 工业实践\n")
        for f in practices:
            title = read_title(os.path.join(KB, domain_dir, 'practices', f))
            lines.append(f"- [{title}](./practices/{f})")
        lines.append("")
    
    # Cards section
    if cards:
        lines.append(f"## 🃏 知识卡片\n")
        for f in cards:
            title = read_title(os.path.join(KB, domain_dir, 'cards', f))
            lines.append(f"- [{title}](./cards/{f})")
        lines.append("")
    
    return '\n'.join(lines)

DOMAINS = {
    'ads': ('广告系统', '广告 CTR 预估、竞拍机制、自动出价'),
    'rec-sys': ('推荐系统', '召回、排序、重排、多目标优化'),
    'search': ('搜索系统', '检索、排序、Query 理解'),
    'llm-infra': ('LLM 基础设施', '推理优化、微调、对齐'),
    'cross-domain': ('跨领域', '偏差治理、多目标、系统设计'),
    'interview': ('面试准备', '知识卡片、项目讲故事'),
}

def update_all():
    for domain_dir, (name, desc) in DOMAINS.items():
        index_path = os.path.join(KB, domain_dir, 'INDEX.md')
        content = generate_index(domain_dir, name, desc)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Updated {domain_dir}/INDEX.md")

if __name__ == '__main__':
    update_all()
