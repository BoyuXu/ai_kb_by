# Qwen2.5-VL Technical Report

**ArXiv:** 2502.13923 | **Org:** Alibaba Qwen Team | **Date:** 2025-02

## 核心贡献
Qwen2.5-VL 是 Alibaba 发布的强大多模态视觉-语言模型，在文档理解、精确定位、长视频分析等方面实现重大突破。

## 关键能力
- **视觉定位**：精确输出 bounding box 或点坐标
- **文档解析**：发票、表单、表格等结构化数据提取
- **动态分辨率**：任意尺寸图像的 native 分辨率处理
- **长视频理解**：支持数小时视频，秒级事件定位
- **绝对时间编码**：视频理解的关键技术创新
- **交互式 Agent**：操作电脑和移动设备的真实世界任务

## 技术架构
- 从头训练的 **Native Dynamic-Resolution ViT**
- 引入 **Window Attention** 降低计算开销
- 支持 3 个模型尺寸（edge AI 到高性能计算）

## 性能
- Qwen2.5-VL-72B 对标 GPT-4o 和 Claude-3.5-Sonnet
- 文档和图表理解方面尤为突出

## 面试考点
- Dynamic resolution ViT 如何处理不同尺寸输入？
- 绝对时间编码 vs 相对时间编码的优势？
- Window Attention 的原理和适用场景？

**Tags:** #llm-infra #multimodal #vision-language #qwen
