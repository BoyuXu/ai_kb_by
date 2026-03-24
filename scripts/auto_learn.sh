#!/bin/bash
# MelonEggLearn 持续自动学习脚本
# 运行方式: bash auto_learn.sh &
# 有内容就一直跑，无内容则等待新内容加入队列

QUEUE_FILE="$HOME/Documents/ai-kb/learning_queue.json"
LOG="$HOME/Documents/ai-kb/auto_learn.log"
KIMI="/usr/local/bin/kimi"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOG"; }

run_batch() {
  local id=$1 topic=$2 output=$3 prompt=$4
  log "▶ 开始: $id"
  $KIMI --work-dir "$HOME/Documents/ai-kb" --print -p "$prompt" >> "$LOG" 2>&1
  local code=$?
  if [ $code -eq 0 ]; then
    log "✅ 完成: $id → $output"
    # 更新队列状态
    python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
for s in q['sources']:
    if s['id'] == '$id':
        s['status'] = 'done'
        import datetime; s['completed_at'] = str(datetime.date.today())
        break
json.dump(q, open('$QUEUE_FILE', 'w'), ensure_ascii=False, indent=2)
"
    openclaw system event --text "MelonEgg学完: $id" --mode now 2>/dev/null || true
  else
    log "❌ 失败: $id (exit $code)"
  fi
}

log "=== MelonEggLearn 自动学习启动 ==="

# ---- Batch: 数字人基础 ----
run_batch "aigc_digital_human" "数字人基础" "interview/qa-bank.md" "
你是MelonEggLearn。读取 /Users/boyu/Documents/ai-kb/repos/AIGC-Interview-Book/数字人基础/ 所有文件。
整理数字人技术面试考点：
1. 数字人技术栈（语音合成TTS/唇形同步/驱动信号）
2. 实时渲染 vs 离线渲染
3. 与LLM结合的对话数字人架构
4. 面试高频问题5道+答案
追加到 ~/Documents/ai-kb/interview/qa-bank.md（不要覆盖，追加到末尾）
"

# ---- Batch: AI绘画基础 ----
run_batch "aigc_ai_painting" "AI绘画基础" "interview/qa-bank.md" "
你是MelonEggLearn。读取 /Users/boyu/Documents/ai-kb/repos/AIGC-Interview-Book/AI绘画基础/ 所有文件。
整理AIGC绘画核心考点：
1. Diffusion Model原理（DDPM/DDIM/Score-based）
2. Stable Diffusion架构（VAE+UNet+CLIP）
3. ControlNet/LoRA微调原理
4. SDXL/Flux最新进展
5. 面试高频问题5道
追加到 ~/Documents/ai-kb/interview/qa-bank.md
"

# ---- Batch: AI视频基础 ----
run_batch "aigc_ai_video" "AI视频基础" "interview/qa-bank.md" "
你是MelonEggLearn。读取 /Users/boyu/Documents/ai-kb/repos/AIGC-Interview-Book/AI视频基础/ 所有文件。
整理AI视频生成核心考点：
1. 视频生成技术演进（GAN→Diffusion→DiT）
2. Sora/CogVideoX/Wan核心架构
3. 视频理解 vs 视频生成
4. 时序一致性核心挑战
5. 面试高频问题5道
追加到 ~/Documents/ai-kb/interview/qa-bank.md
"

# ---- Batch: AlgoNotes 精排深度 ----
run_batch "algo_notes_ranking" "精排模型深度" "rec-sys/05_ranking_deep.md" "
你是MelonEggLearn。基于你的知识库，深度整理精排模型进阶内容：

1. 多任务学习进阶
   - MMOE/PLE/SNR/AITM对比实现
   - 任务冲突时的梯度处理（PCGrad/GradNorm）
   
2. 序列建模进阶
   - SIM超长序列（Hard/Soft检索）
   - ETA（Efficient Target Attention）
   - TWIN双塔序列
   
3. 粗排模型
   - COLD（级联精排）
   - 知识蒸馏从精排到粗排
   
4. 在线学习
   - 实时特征更新
   - 延迟反馈建模（DEFER/ES-DFM）

5. 代码实现要点（PyTorch伪代码）

输出：~/Documents/ai-kb/rec-sys/05_ranking_deep.md
"

# ---- Batch: 模拟面试-推荐系统 ----
run_batch "mock_interview_recsys" "推荐系统模拟面试" "interview/mock-interview-recsys.md" "
你是MelonEggLearn，模拟一场字节/美团/快手的推荐算法岗面试（60分钟）。

以面试官-候选人问答格式输出，涵盖：
1. 自我介绍引导（2分钟）
2. 基础概念（10题，含标准答案）
3. 系统设计题：设计一个直播推荐系统（详细答案）
4. 代码题：实现MMOE的PyTorch代码（含完整实现）
5. 场景题：如何解决推荐系统的流行度偏差？
6. 反问环节建议

输出：~/Documents/ai-kb/interview/mock-interview-recsys.md
格式：## Q1: ... **面试官**: ... **候选人（参考答案）**: ...
"

# ---- Batch: 模拟面试-LLM ----
run_batch "mock_interview_llm" "LLM模拟面试" "interview/mock-interview-llm.md" "
你是MelonEggLearn，模拟一场大模型算法岗面试。

涵盖：
1. Transformer架构细节（Self-Attention复杂度、位置编码、KV Cache）
2. 预训练：数据清洗、Loss设计、继续预训练
3. 对齐：RLHF/DPO/PPO原理对比
4. 推理加速：vLLM/连续批处理/投机采样
5. RAG系统设计：Naive RAG→Advanced RAG→Modular RAG
6. Agent框架：ReAct/CoT/Tool Use

每题给出候选人高质量答案（面试官会追问的角度也给出）

输出：~/Documents/ai-kb/interview/mock-interview-llm.md
"

log "=== 本轮自动学习完成 ==="
python3 -c "
import json
q = json.load(open('$QUEUE_FILE'))
done = sum(1 for s in q['sources'] if s['status']=='done')
total = len(q['sources'])
print(f'队列进度: {done}/{total}')
"
