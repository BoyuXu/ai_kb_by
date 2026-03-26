# CTCVR Optimization with Heterogeneous Hierarchical Decoder in Weixin Channels
> 来源：arxiv/2401.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
微信视频号广告的 CTCVR（点击-转化率联合优化）挑战：
- 视频广告与电商转化链路长（观看→点击→加购→购买）
- 视频内容特征（帧、音频、字幕）与广告转化难以统一建模
- 视频场景下用户行为异质性高（不同用户转化路径差异大）
- CTCVR = CTR × CVR，但 CVR 样本极度稀疏

## 核心方法与创新点
**HHD（Heterogeneous Hierarchical Decoder）**：异质分层解码器。

**分层转化路径建模：**
```
曝光 → 点击 → 跳转落地页 → 加购 → 购买
Level 1 Decoder: P(click | view) = CTR
Level 2 Decoder: P(land | click) = LTR（落地率）  
Level 3 Decoder: P(add_cart | land) = ACTR（加购率）
Level 4 Decoder: P(purchase | add_cart) = CVR
CTCVR = CTR × LTR × ACTR × CVR
```

**异质解码器设计：**
```python
# 每层解码器捕获该阶段特有的用户状态
decoder_k = TransformerDecoder(
    query=action_state_k,      # 该阶段的用户状态
    key_value=shared_encoder,  # 共享视频/广告特征
    cross_attention_mask=stage_mask_k  # 阶段特定掩码
)
```

**分层监督：**
```
L = Σ_k BCE(decoder_k_output, label_k)
# 每个阶段独立监督，解决稀疏标签问题
```

## 实验结论
- 微信视频号广告线上 A/B（100% 流量）：
  - GMV +4.1%，CTR +1.3%，CVR +2.8%
- 离线 AUC（各级）：CTCVR AUC +0.008（vs 标准 ESMM）
- 视频冷启动广告（<100 次曝光）：CTCVR +9.2%

## 工程落地要点
1. **标签稀疏性**：越深层标签越稀疏（购买仅 0.01%），用加权 BCE 平衡
2. **视频特征提取**：离线提取视频帧 embedding（CLIP）+ ASR 文本特征
3. **多阶段 ESMM**：全空间建模，CVR 在点击空间训练，避免 SSB
4. **实时更新**：点击信号实时，购买信号有延迟（T+1），需异步更新
5. **层级特征工程**：每层使用不同特征集（CTR 层用视频内容，CVR 层用商品特征）

## 面试考点
**Q1: 为什么要将 CTCVR 分解为多级联合建模？**
A: 直接建模 CTCVR 信号极稀疏（购买率 0.01%），梯度信号弱。分层建模每级都有监督信号（CTR 1%，LTR 40%，ACTR 10%），梯度更充分；同时可以分析各阶段的转化漏斗，定向优化。

**Q2: 样本选择偏差（SSB）问题如何解决？**
A: ESMM 框架：在全空间（所有曝光）训练 CTR，在点击空间训练 CVR，但通过 CTR × CVR = CTCVR 约束联合训练。这样 CVR 虽然在点击样本上更新，但通过 CTCVR 的全空间监督间接消除偏差。

**Q3: 视频广告特征与展示广告特征的主要区别？**
A: 视频广告有时序特征（帧序列）、音频（语音/背景音乐）、字幕文本，内容信息量大；展示广告主要是图片/文案。视频广告需要多模态 encoder（视觉 + 音频 + 文本），且用户行为受视频内容影响更强。

**Q4: 长链路转化（4+层）训练时梯度消失如何处理？**
A: ①残差连接：每层解码器有 skip connection ②分层独立监督：每层都有 BCE Loss，梯度直接到每层 ③较大学习率用于深层（购买层），小学习率用于浅层（点击层）。

**Q5: CTCVR 模型如何处理不同广告主的转化目标不统一？**
A: 多目标塔：每类转化目标（加购/表单/App激活）独立 Tower，共享底层 encoder；线上根据广告主设定的优化目标（oCPX 出价类型）动态选择对应的 Tower 打分。
