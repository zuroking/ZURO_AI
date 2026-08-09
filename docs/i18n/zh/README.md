# Kronos Synapse Dialogue Core

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Params](https://img.shields.io/badge/params-%7E15.3M-informational)

**语言：** [English](../../../README.md) · [Español](../es/README.md) · [Deutsch](../de/README.md) · **中文** · [Русский](../ru/README.md)

一个仅解码器（decoder-only）的 GPT 风格语言模型（约 1530 万参数），完全基于纯 `torch.nn`
原语从零编写，并完全在 CPU 上训练。它自带一个 BPE 分词器、一条由 memmap 支撑的训练流水线，
以及一个围绕 KV 缓存生成器构建的流式 CLI 聊天。整个栈——模型、数据、训练、推理——都是手工
实现的，没有借助 `transformers` 或 `nanoGPT` 的脚手架。

> **它是什么（以及不是什么）。** 这个模型做的是**文本续写**，而不是遵循指令。它是在文学
> 散文（Project Gutenberg）上训练的，而不是在带标注的对话对上。你输入文本，它续写下去。
> 它不是一个角色扮演助手。参见[聊天范式与局限](#聊天范式与局限)。

---

## 目录

- [快速开始](#快速开始)
- [亮点](#亮点)
- [架构](#架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置](#配置)
- [准备数据](#准备数据)
- [训练](#训练)
- [评估](#评估)
- [聊天与推理](#聊天与推理)
- [聊天范式与局限](#聊天范式与局限)
- [测试](#测试)
- [开发](#开发)
- [性能说明](#性能说明)
- [路线图](#路线图)
- [常见问题](#常见问题)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 快速开始

如果你只想看它跑起来，这是从头到尾的完整流程：

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 把一些 .txt 书籍放进 data/raw/gutenberg/，然后：
kronos_synapse tokenize --corpus-dir data/raw/gutenberg

# 3. 训练（Ctrl+C 是安全的；之后可从检查点恢复）
kronos_synapse train

# 4. 与它对话
kronos_synapse chat checkpoints/best.pt
```

如果你更喜欢 `make tokenize`、`make train`、`make chat`，也有一个 `Makefile`。
一切都在 CPU 上运行——不需要 GPU、云账户或 API 密钥。

## 亮点

- **手工打造的 Transformer。** `CausalSelfAttention`、`MLP`、`TransformerBlock` 和
  `MiniGPT` 直接构建在 `nn.Linear`、`nn.LayerNorm`、`nn.Embedding` 和 `nn.Dropout` 之上。
  注意力经由 `F.scaled_dot_product_attention`，它会在 CPU 上选择优化过的融合内核。
- **CPU 优先。** 一切无需 CUDA 即可运行。训练循环通过梯度累积、`bfloat16` autocast、
  可配置的线程数以及可选的 `torch.compile` 回退来精细权衡。
- **KV 缓存生成。** 推理先对提示做一次预填充（prefill），随后针对缓存的键/值一次解码一个
  token——这正是笔记本 CPU 上聊天可用与不可用的区别。
- **可调的采样器。** 温度、top-k、top-p（nucleus）和重复惩罚都作为 CLI 参数暴露出来。
- **带类型的配置。** 三个 `pydantic-settings` 配置（`ModelConfig`、`TrainConfig`、
  `DataConfig`）用经过校验的、不可变的设置取代了魔法字典。

## 架构

一个标准的 pre-LN GPT 解码器。下表是默认的 `ModelConfig`。

| 组件 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 12,000 | BPE，从零训练 |
| `d_model` | 384 | 嵌入 / 隐藏维度 |
| `n_layers` | 6 | Transformer 块 |
| `n_heads` | 6 | 每头维度 = 64 |
| `d_ff` | 1,536 | 4× `d_model`，GELU MLP |
| `context_length` | 256 | 学习式绝对位置嵌入 |
| `dropout` | 0.1 | 嵌入、注意力输出、MLP 输出 |
| `lm_head` | 与 `tok_emb` 绑定 | 节省约 460 万参数 |
| 注意力内核 | `F.scaled_dot_product_attention` | 融合的，训练时因果 |
| 权重初始化 | GPT-2 方案 | `N(0, 0.02)`，残差投影按 `1/√(2·n_layers)` 缩放 |

**参数预算（约 1530 万）：**

| 组件 | 参数量 |
|---|---|
| Token 嵌入（与 `lm_head` 绑定） | 4,608,000 |
| 位置嵌入 | 98,304 |
| 6 × Transformer 块（attn + MLP + 2× LayerNorm） | 10,626,048 |
| 最终 LayerNorm | 768 |
| **总计** | **≈ 15,333,120** |

该计数在每次训练运行开始时都会打印，以便你确认它落在目标的 ±5% 以内。

## 项目结构

```text
kronos-synapse-dialogue/
├── pyproject.toml          # 依赖 + 控制台入口点
├── config.py               # ModelConfig / TrainConfig / DataConfig (pydantic-settings)
├── cli.py                  # Typer 入口：tokenize / train / eval / chat
├── tokenizer/
│   ├── trainer.py          # BPE 训练（HF tokenizers, ByteLevel）
│   └── wrapper.py          # KronosTokenizer：编码 / 解码 / 对话轮次
├── model/
│   ├── attention.py        # CausalSelfAttention + KVCache
│   ├── mlp.py              # GELU 前馈
│   ├── block.py            # pre-LN TransformerBlock
│   └── gpt.py              # MiniGPT（嵌入 + 块 + 绑定的 lm_head）
├── data/
│   ├── sources.py          # TextCorpusSource 抽象基类 + GutenbergCorpusSource
│   ├── packing.py          # 语料 → 按书切分 → uint16 memmap
│   └── dataset.py          # BinaryTokenDataset + DataLoader 工厂
├── training/
│   ├── scheduler.py        # 带线性预热的余弦衰减
│   ├── checkpoint.py        # 保存 / 加载 模型 + 优化器 + 调度器
│   └── loop.py             # 训练循环、指标、Rich 进度条
├── inference/
│   ├── generate.py         # KV 缓存采样生成器
│   └── conversation.py     # ConversationBuffer（按句边界裁剪）
├── tests/                  # 形状、因果掩码、打包、缓冲区单元测试
└── data/
    ├── raw/gutenberg/      # <-- 把 .txt 语料文件放在这里
    └── processed/          # 生成的：tokenizer.json, train.bin, val.bin
```

## 环境要求

- **Python** 3.12 或更高
- **PyTorch** 2.3+（CPU 版本；`bfloat16` autocast 在 Alder Lake 上是模拟的，但仍能节省
  内存带宽——并有 fp32 回退）
- **硬件**：一颗现代多核 CPU。开发目标是 Intel i7-1225U（2P+8E，AVX2，无 AVX-512）。无需 GPU。

## 安装

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

这会注册 `kronos_synapse` 控制台命令，并引入开发工具（`pytest`、`mypy`）。

## 配置

所有设置都作为不可变的 `pydantic-settings` 模型存放在 [`config.py`](../../../config.py) 中。
在那里编辑默认值即可重新调整模型或训练计划。

### `ModelConfig`

| 选项 | 默认值 | 描述 |
|---|---|---|
| `vocab_size` | `12_000` | BPE 词表大小 |
| `d_model` | `384` | 隐藏维度 |
| `n_layers` | `6` | Transformer 块 |
| `n_heads` | `6` | 注意力头数 |
| `d_ff` | `1536` | MLP 内部维度 |
| `context_length` | `256` | 最大序列长度 |
| `dropout` | `0.1` | dropout 概率 |

### `TrainConfig`

| 选项 | 默认值 | 描述 |
|---|---|---|
| `num_threads` | `8` | `torch.set_num_threads` |
| `batch_size` | `16` | 微批大小 |
| `grad_accum_steps` | `8` | 有效批量 = `16 × 8 = 128` |
| `max_iters` | `50_000` | 优化器步数 |
| `lr` | `3e-4` | 峰值学习率 |
| `lr_min` | `1e-5` | 余弦下限 |
| `warmup_iters` | `200` | 线性预热步数 |
| `weight_decay` | `0.1` | 仅应用于 2D 参数 |
| `max_grad_norm` | `1.0` | 梯度裁剪 |
| `use_bf16` | `True` | CPU 上的 `bfloat16` autocast |
| `compile_model` | `False` | 可选的 `torch.compile`，带 fp32 回退 |
| `checkpoint_every` | `1_000` | 检查点 + 验证评估的节奏 |
| `checkpoint_dir` | `checkpoints/` | 输出目录 |
| `log_file` | `logs/train.jsonl` | 每个检查点的指标日志 |

### `DataConfig`

| 选项 | 默认值 | 描述 |
|---|---|---|
| `raw_dir` | `data/raw/gutenberg/` | 输入 `.txt` 语料 |
| `processed_dir` | `data/processed/` | `train.bin` / `val.bin` 输出 |
| `tokenizer_dir` | `data/processed/tokenizer/` | `tokenizer.json` 输出 |
| `min_quote_line_ratio` | `0.0` | 可选的对话密度过滤 |
| `split_seed` | `42` | 按书切分的随机种子 |
| `val_fraction` | `0.10` | 保留的书（训练中未见） |

## 准备数据

1. 把 Project Gutenberg 的 `.txt` 文件放进 `data/raw/gutenberg/`。（19 世纪到 20 世纪初的
   小说——Austen、Dickens、Twain 等——效果很好：有大量直接引语和简短对白。）
2. 训练分词器，并把语料打包成内存映射的二进制切分：

```bash
kronos_synapse tokenize --corpus-dir data/raw/gutenberg
```

这会在 `data/processed/` 中产生三个产物：

- `tokenizer/tokenizer.json` — BPE 分词器
- `train.bin` — 作为 `uint16` memmap 的训练 token
- `val.bin` — 验证 token（整本书被保留，因此没有窗口泄漏）

语料清理会自动进行：去除 Project Gutenberg 的页眉/页脚样板文字，文本做 Unicode-NFKC
归一化，连续空白被折叠。把 `min_quote_line_ratio` 设为大于 0，可只保留引用对话密集的文档。

特殊 token 是固定的：`<pad>=0`、`<unk>=1`、`<bos>=2`、`<eos>=3`。不添加角色 token
（`<user>`/`<bot>`）——参见[聊天范式](#聊天范式与局限)说明。

## 训练

```bash
kronos_synapse train
```

该循环在开始时打印参数量，然后带着 Rich 进度条运行。每 `checkpoint_every` 步它会：

- 评估验证损失，
- 把 `{step, train_loss, val_loss}` 追加到 `logs/train.jsonl`，
- 写入 `checkpoints/step_<n>.pt`，
- 如果验证损失改善，则更新 `checkpoints/best.pt`。

**恢复**（会恢复优化器和调度器状态）任意检查点：

```bash
kronos_synapse train --resume checkpoints/best.pt
```

## 评估

在保留的验证书籍上评估某个检查点：

```bash
kronos_synapse eval checkpoints/best.pt --split val
```

在该切分上报告交叉熵损失和困惑度（`2^loss`）。

## 聊天与推理

```bash
kronos_synapse chat checkpoints/best.pt --temperature 0.8 --top-p 0.9
```

生成会逐 token 流式输出到终端。第一次调用会预填充提示并构建 KV 缓存；之后每个 token 都针对
该缓存解码，因此长对话也能保持响应迅速。

| 参数 | 默认值 | 描述 |
|---|---|---|
| `--temperature` | `0.8` | 使分布更尖锐（`<1`）或更平坦（`>1`） |
| `--top-k` | `50` | 仅保留概率最高的 `k` 个 token（`0` 表示禁用） |
| `--top-p` | `0.9` | nucleus 采样（`1.0` 表示禁用） |
| `--repetition-penalty` | `1.1` | 降低已生成 token 的权重 |
| `--max-new-tokens` | `128` | 每轮的生成上限 |

对话缓冲区保存一段不断增长的纯文本历史；当它接近 `context_length` 时，会从前端在最近的
句子边界处裁剪（绝不在词中间），并保留开头的 `<bos>`。这个滑动窗口*就是*短期记忆——模型
每一轮都读取整个缓冲区。

## 聊天范式与局限

本项目刻意限定了范围，并对一个 1500 万参数的模型能从连续散文中学到什么保持诚实：

- **它续写文本，不遵循指令。** 训练数据中没有 `<user>`/`<bot>` 轮次，所以模型没有显式轮流
  发言的信号。你输入的内容被视为一段文字的开头；模型写出接下来的内容。
- **连贯性是局部的。** 在上下文窗口内，预期约 5–6 句风格一致、实体可追踪。更长距离的连贯性
  不作保证。
- **它只懂英语**，且带有文学色彩，因为语料是英语小说。

一个检验上下文窗口是否真的被使用的合理方法：在缓冲区靠前处提到一个角色名，确认模型在 4–6 句
之后仍会提及它。`data/sources.py` 中留有一个 `DialogueDatasetSource` 存根，作为你日后在
带标注对话上做微调的扩展点。

## 测试

```bash
pytest
```

单元测试覆盖了最容易悄悄出错的部分：

- `tests/test_model_shapes.py` — 前向传播的张量形状、参数量
- `tests/test_attention.py` — 注意力输出形状与 KV 缓存行为
- `tests/test_causal_mask.py` — 不向未来位置泄漏
- `tests/test_packing.py` — 按书切分与二进制打包逻辑
- `tests/test_conversation_buffer.py` — 按句边界裁剪会保留 `<bos>`

这些测试使用一个微小的一次性配置和一个即时训练的小分词器，因此整个套件在几秒内就能跑完，
且不需要下载任何数据。

## 开发

常见任务都封装在 `Makefile` 里：

```bash
make dev         # 带开发额外依赖的可编辑安装
make test        # pytest -q
make typecheck   # mypy .
make clean       # 清除缓存和构建产物
```

代码完全带类型注解，并用 `mypy` 检查。添加代码时请保持类型标注；若你改动了模型、分词器或
打包路径，请补上测试。风格是普通的 4 空格缩进 Python；注释解释*为什么*，而非*是什么*。
完整说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 性能说明

在开发用 CPU（i7-1225U）上，训练期间大致可达 **400–800 token/秒**，在约 2000 万 token 的
语料上相当于每个 epoch 约 **7–14 小时**。吞吐量随 `num_threads`、`use_bf16` 和
`batch_size × grad_accum_steps` 变化；请按你自己的核心数和内存来调优。在同一颗 CPU 上，
带 KV 缓存的生成明显快于不带缓存的基线——这个差距正是这里强制使用缓存的原因。

## 路线图

大致顺序，并非承诺：

- **RoPE 位置编码**，取代学习式绝对位置，让模型能外推到 `context_length` 之外。
- **可选的指令微调**，在基础模型之上接入 `DialogueDatasetSource` 存根和真正的
  `<user>`/`<bot>` 角色 token。
- **更长的上下文**（512+），一旦训练吞吐量允许。
- **一个 `sample` CLI 命令**，用于无需交互循环的一次性续写。
- **KV 缓存淘汰**，让聊天会话能无限期运行，而不是裁剪文本。

## 常见问题

**为什么只用 CPU？** 因为初衷就是看看一个从零构建的模型在没有 GPU 的笔记本上能走多远。
一切都是围绕这个约束调优的。

**为什么不直接用 `transformers`？** 亲手搭建这些部件本身就是整个练习。你能确切看到注意力、
KV 缓存和训练循环在做什么。

**它回复乱码 / 不断重复。** 在有限数据上训练的小模型就是这样。给它更多语料、更多步数，
降低 `--temperature`，并稍微提高 `--repetition-penalty`。也别忘了它是*续写*文本——它不回答
问题（参见[聊天范式与局限](#聊天范式与局限)）。

**我能用自己的文本而不是 Gutenberg 吗？** 可以。任何纯 `.txt` 文件都行；Gutenberg 专属的
步骤只是去除页眉/页脚，对其他文本而言是空操作。若是其他格式，在 `data/sources.py` 中添加一个
`TextCorpusSource` 子类即可。

**怎么把它做得更大/更小？** 编辑 `config.py` 中的默认值（`d_model`、`n_layers`、`n_heads`、
`context_length`）。参数量会在训练开始时打印，方便你查看最终落点。

## 参与贡献

欢迎贡献——设置与期望见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本历史见
[CHANGELOG.md](CHANGELOG.md)。对于较大的改动，请先开一个 issue，以便我们就方案达成一致。

## 许可证

[MIT](../../../LICENSE) © 2026 Kronos
