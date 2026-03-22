#!/usr/bin/env python3
"""初始化/检查队列文件结构"""
import json, os

QUEUE_FILE = os.path.expanduser("~/Documents/ai-kb/papers_queue.jsonl")
LOG_FILE = os.path.expanduser("~/Documents/ai-kb/processed_log.jsonl")

# 创建空文件（如果不存在）
for f in [QUEUE_FILE, LOG_FILE]:
    if not os.path.exists(f):
        open(f, 'w').close()
        print(f"Created: {f}")
    else:
        # 统计现有条目
        lines = [l for l in open(f) if l.strip()]
        print(f"Exists: {f} ({len(lines)} entries)")

# 统计队列状态
pending, done = 0, 0
for line in open(QUEUE_FILE):
    if line.strip():
        try:
            item = json.loads(line)
            if item.get('status') == 'done':
                done += 1
            else:
                pending += 1
        except:
            pass
print(f"\nQueue status: {pending} pending, {done} done")
