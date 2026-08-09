# 更新日志

**语言：** [English](../../../CHANGELOG.md) · [Español](../es/CHANGELOG.md) · [Deutsch](../de/CHANGELOG.md) · **中文** · [Русский](../ru/CHANGELOG.md)

本项目所有值得注意的变更都记录在此。格式大致遵循
[Keep a Changelog](https://keepachangelog.com/)，版本号力求符合
[语义化版本](https://semver.org/)。

## [未发布]

### 新增
- `CONTRIBUTING.md`、`CHANGELOG.md`、`.editorconfig`，以及一个包含常见开发目标的 `Makefile`。
- 固定版本的依赖清单（`requirements.txt`、`requirements-dev.txt`）。
- 位于 `docs/i18n/` 的多语言文档（西班牙语、德语、中文、俄语），涵盖 README、贡献指南和
  更新日志，并在每个文件顶部提供语言切换器。

### 变更
- 精简了代码中的注释与 docstring，以提升可读性。

## [0.1.0]

首个版本。

### 新增
- 手工编写的仅解码器 GPT（约 1530 万参数），基于 `torch.nn` 原语：`CausalSelfAttention`、
  `MLP`、`TransformerBlock`、`MiniGPT`。
- 字节级 BPE 分词器（`train_bpe`）和一个轻量的 `KronosTokenizer` 封装，带固定的特殊 token
  （`<pad>`、`<unk>`、`<bos>`、`<eos>`）。
- 数据流水线：带页眉/页脚剥离与归一化的 Gutenberg 语料源、按书切分的 train/val，以及
  `uint16` memmap 打包。
- CPU 优先的训练循环，支持梯度累积、`bfloat16` autocast、带预热的余弦计划、梯度裁剪、
  检查点保存以及 JSONL 指标日志。
- 带温度 / top-k / top-p / 重复惩罚采样的 KV 缓存生成。
- 用于聊天的 `ConversationBuffer`，按句边界裁剪。
- `kronos_synapse` CLI：`tokenize`、`train`、`eval`、`chat`。
- 针对模型形状、因果掩码、注意力/KV 缓存、打包和对话缓冲区的单元测试。
