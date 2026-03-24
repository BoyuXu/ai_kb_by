#!/usr/bin/env python3
"""Apply formula and Q&A enhancements to synthesis files."""

import os
import re
import json
import glob

KB = os.path.expanduser("~/Documents/ai-kb")

def count_questions(content):
    return len(re.findall(r'^### Q\d+:', content, re.MULTILINE))

def has_formula_section(content):
    return '## 📐 核心公式与原理' in content

def get_last_q_num(content):
    nums = re.findall(r'^### Q(\d+):', content, re.MULTILINE)
    return max(int(n) for n in nums) if nums else 0

def add_formulas(content, formulas_text):
    """Insert formulas section after the 参考文献 block."""
    if has_formula_section(content):
        return content
    
    # Find the end of the 参考文献 section (after the last > line before ---)
    lines = content.split('\n')
    insert_idx = None
    
    # Find first --- after the reference block
    in_refs = False
    for i, line in enumerate(lines):
        if line.strip().startswith('> 📚') or line.strip().startswith('>📚'):
            in_refs = True
        if in_refs and line.strip() == '---':
            insert_idx = i
            break
    
    if insert_idx is None:
        # Find first --- in the file
        for i, line in enumerate(lines):
            if line.strip() == '---' and i > 3:
                insert_idx = i
                break
    
    if insert_idx is None:
        # Append before the first ## section
        for i, line in enumerate(lines):
            if line.startswith('## ') and i > 0:
                insert_idx = i
                break
    
    if insert_idx is None:
        # Append at end
        content += '\n\n' + formulas_text
        return content
    
    lines.insert(insert_idx, '\n' + formulas_text + '\n')
    return '\n'.join(lines)

def add_extra_qa(content, extra_qa_text):
    """Add extra Q&A after existing Q&A section."""
    current_q = count_questions(content)
    if current_q >= 10:
        return content
    
    # Renumber the extra QAs starting from current_q + 1
    next_num = get_last_q_num(content) + 1
    
    # Extract QAs from extra_qa_text
    qas = re.findall(r'(### Q\d+:.*?)(?=### Q\d+:|$)', extra_qa_text, re.DOTALL)
    
    renumbered = []
    for qa in qas:
        if current_q + len(renumbered) >= 10:
            break  # enough
        new_qa = re.sub(r'### Q\d+:', f'### Q{next_num + len(renumbered)}:', qa, count=1)
        renumbered.append(new_qa.strip())
    
    if not renumbered:
        return content
    
    # Find the last Q&A in the file
    last_qa_match = None
    for m in re.finditer(r'^### Q\d+:', content, re.MULTILINE):
        last_qa_match = m
    
    if last_qa_match:
        # Find the end of the last Q&A section
        # Look for the next ## heading or ## section or --- or end of file
        rest = content[last_qa_match.end():]
        # Find next section break
        next_section = re.search(r'\n## [^#]|\n---\n|\n## 🌐', rest)
        if next_section:
            insert_pos = last_qa_match.end() + next_section.start()
        else:
            insert_pos = len(content)
        
        extra_text = '\n\n' + '\n\n'.join(renumbered)
        content = content[:insert_pos] + extra_text + content[insert_pos:]
    
    return content

def load_enhancements():
    """Load enhancement data from JSON files."""
    path = os.path.join(KB, 'scripts', 'enhancements.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def process_file(filepath, formulas=None, extra_qa=None):
    """Process a single synthesis file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    if formulas and not has_formula_section(content):
        content = add_formulas(content, formulas)
    
    if extra_qa:
        content = add_extra_qa(content, extra_qa)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def scan_status():
    """Scan all synthesis files and report status."""
    dirs = ['ads/synthesis', 'rec-sys/synthesis', 'search/synthesis',
            'llm-infra/synthesis', 'cross-domain/synthesis', 'interview/synthesis']
    
    total = 0
    has_formulas = 0
    has_10q = 0
    
    for d in dirs:
        full_dir = os.path.join(KB, d)
        if not os.path.isdir(full_dir):
            continue
        for f in sorted(os.listdir(full_dir)):
            if not f.endswith('.md'):
                continue
            fp = os.path.join(full_dir, f)
            with open(fp, 'r', encoding='utf-8') as fh:
                content = fh.read()
            
            nq = count_questions(content)
            hf = has_formula_section(content)
            total += 1
            if hf: has_formulas += 1
            if nq >= 10: has_10q += 1
            
            if not hf or nq < 10:
                print(f"  {'✅' if hf else '❌'}F {'✅' if nq>=10 else f'❌{nq:2d}'}Q  {d}/{f}")
    
    print(f"\nTotal: {total} | Formulas: {has_formulas}/{total} | Q>=10: {has_10q}/{total}")

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'status'
    
    if mode == 'status':
        scan_status()
    elif mode == 'apply':
        # Apply from enhancements.json
        enhancements = load_enhancements()
        applied = 0
        for filename, data in enhancements.items():
            # Find the file
            for root, dirs, files in os.walk(KB):
                if '.git' in root or 'repos' in root:
                    continue
                if filename in files:
                    fp = os.path.join(root, filename)
                    if process_file(fp, data.get('formulas'), data.get('extra_qa')):
                        applied += 1
                        print(f"  ✅ Enhanced: {filename}")
                    break
        print(f"\nApplied enhancements to {applied} files")
    else:
        print("Usage: status | apply")
