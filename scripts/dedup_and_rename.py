#!/usr/bin/env python3
"""
Step 1: Deduplicate papers (keep latest/shortest, remove older duplicates)
Step 2: Rename remaining papers to English names
Step 3: Handle synthesis renames
"""

import os
import re
import subprocess
import json

KB = os.path.expanduser("~/Documents/ai-kb")

def read_title(fp):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('# '):
                    return line.strip()[2:]
    except: pass
    return None

def file_size(fp):
    try:
        return os.path.getsize(fp)
    except:
        return 0

def find_duplicates():
    """Find duplicate papers by title within each directory."""
    dirs = ['ads/papers', 'rec-sys/papers', 'search/papers']
    dupes = {}  # (dir, title) -> [files]
    
    for d in dirs:
        full = os.path.join(KB, d)
        if not os.path.isdir(full): continue
        for f in sorted(os.listdir(full)):
            if not f.endswith('.md') or not re.match(r'2026', f): continue
            fp = os.path.join(full, f)
            title = read_title(fp)
            if not title: continue
            key = (d, title)
            if key not in dupes:
                dupes[key] = []
            dupes[key].append(f)
    
    return {k: v for k, v in dupes.items() if len(v) > 1}

def choose_best_duplicate(dir_path, files):
    """Choose the best file among duplicates (largest content = most complete)."""
    best = None
    best_size = 0
    for f in files:
        fp = os.path.join(KB, dir_path, f)
        sz = file_size(fp)
        if sz > best_size:
            best = f
            best_size = sz
    return best

def dedup_papers():
    """Remove duplicate papers, keeping the best version."""
    dupes = find_duplicates()
    removed = []
    
    for (d, title), files in sorted(dupes.items()):
        best = choose_best_duplicate(d, files)
        for f in files:
            if f != best:
                fp = os.path.join(KB, d, f)
                print(f"  rm {d}/{f} (keeping {best})")
                subprocess.run(['git', 'rm', fp], cwd=KB, capture_output=True)
                removed.append(os.path.join(d, f))
    
    print(f"\nRemoved {len(removed)} duplicate papers")
    return removed

def title_to_filename(title, orig_filename, max_len=57):
    """Convert paper title to clean English filename."""
    if not title: return None
    
    # Strip Chinese
    clean = re.sub(r'[\u4e00-\u9fff\uff00-\uffef：。，；（）]+', '', title)
    clean = re.sub(r'[^\w\s\-]', ' ', clean)
    clean = clean.strip()
    
    if len(clean) < 15:
        # Mostly Chinese title - use original filename (minus date prefix)
        fallback = re.sub(r'^2026\d+_', '', orig_filename).replace('.md', '')
        # Extract English parts from title for enrichment
        parts = re.findall(r'[A-Za-z][A-Za-z0-9\-_]+', title)
        if parts and len('_'.join(parts)) > len(fallback):
            clean = '_'.join(parts)
        else:
            clean = fallback
    
    clean = re.sub(r'[\s\-:,./]+', '_', clean)
    clean = re.sub(r'_+', '_', clean)
    clean = clean.strip('_')
    
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip('_')
    
    return clean

def rename_papers():
    """Rename date-prefixed papers to English names."""
    dirs = ['ads/papers', 'rec-sys/papers', 'search/papers']
    mapping = {}
    used = {}  # dir -> set of used names
    
    for d in dirs:
        full = os.path.join(KB, d)
        if not os.path.isdir(full): continue
        used[d] = set()
        
        # Also track existing non-date files
        for f in os.listdir(full):
            if f.endswith('.md') and not re.match(r'2026', f):
                used[d].add(f)
        
        for f in sorted(os.listdir(full)):
            if not f.endswith('.md') or not re.match(r'2026', f): continue
            fp = os.path.join(full, f)
            title = read_title(fp)
            new_name = title_to_filename(title, f)
            
            if not new_name:
                # Fallback: use cleaned version of original name
                new_name = re.sub(r'^2026\d+_', '', f).replace('.md', '')
                new_name = new_name[:57]
            
            new_file = new_name + '.md'
            
            # Handle collisions
            if new_file in used[d]:
                # Add suffix
                new_file = new_name + '_v2.md'
                if new_file in used[d]:
                    continue  # skip
            
            used[d].add(new_file)
            old_rel = os.path.join(d, f)
            new_rel = os.path.join(d, new_file)
            mapping[old_rel] = new_rel
    
    return mapping

def execute_renames(mapping):
    """Execute git mv for mapping."""
    ok = 0
    for old, new in sorted(mapping.items()):
        old_fp = os.path.join(KB, old)
        new_fp = os.path.join(KB, new)
        if os.path.exists(old_fp) and not os.path.exists(new_fp):
            r = subprocess.run(['git', 'mv', old_fp, new_fp], cwd=KB, capture_output=True, text=True)
            if r.returncode == 0:
                ok += 1
            else:
                print(f"  ❌ {os.path.basename(old)}: {r.stderr.strip()}")
        elif not os.path.exists(old_fp):
            pass  # already moved/removed
    print(f"Renamed {ok} files")
    return ok

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'help'
    
    if mode == 'dedup-preview':
        dupes = find_duplicates()
        print(f"Found {sum(len(v)-1 for v in dupes.values())} duplicates to remove:")
        for (d, title), files in sorted(dupes.items()):
            best = choose_best_duplicate(d, files)
            print(f"\n  {title[:60]}...")
            for f in files:
                mark = ' ✅ KEEP' if f == best else ' ❌ REMOVE'
                sz = file_size(os.path.join(KB, d, f))
                print(f"    {f} ({sz}b){mark}")
    
    elif mode == 'dedup':
        dedup_papers()
    
    elif mode == 'rename-preview':
        mapping = rename_papers()
        print(f"=== Paper Rename Preview ({len(mapping)} files) ===")
        for old, new in sorted(mapping.items()):
            print(f"  {os.path.basename(old):60s} → {os.path.basename(new)}")
    
    elif mode == 'rename':
        mapping = rename_papers()
        execute_renames(mapping)
        with open(os.path.join(KB, 'scripts/paper_rename_map.json'), 'w') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    else:
        print("Usage: dedup-preview | dedup | rename-preview | rename")
