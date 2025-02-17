# Awesome-DeepSeek-R1-Reproduction

A curated collection of projects, benchmarks, and research papers focused on reproducing and advancing the DeepSeek R1 framework. This repository aggregates various efforts in the fields of large language models (LLMs) and multimodal LLMs (MLLMs), offering implementations, evaluations, and training recipes to help you explore and extend state-of-the-art reasoning capabilities.

## Contents

- [LLM Related](#llm-related)
- [MLLM Related](#mllm-related)
- [Benchmarks](#benchmarks)
- [System Optimization](#system-optimization)
- [SFT Methods](#sft-methods)
- [Papers](#papers)

---

## LLM Related

- **SimpleRL-reason**: A simple reinforcement learning recipe designed to enhance model reasoning capabilities.
- **TinyZero**: A minimal and accessible reproduction of DeepSeek R1-Zero, focusing on countdown and multiplication tasks using reinforcement learning.
- **open-r1**: A fully open reproduction of DeepSeek-R1, including scripts for both training and evaluation with SFT and GRPO.
- **Logic-RL**: Implements DeepSeek R1 Zero on logic puzzles, showcasing enhanced reasoning through rule-based reinforcement learning.
- **DeepScaleR**: Demonstrates advancements by surpassing the O1 preview with a 1.5B model through reinforcement learning and model scaling techniques.
- **K1.5**: Describes the training recipe and system design for Kimi K1.5—a multi-modal LLM that leverages long context scaling and improved policy optimization to achieve state-of-the-art reasoning performance across diverse benchmarks and modalities.

---

## MLLM Related

- **lmm-r1**: Extends OpenRLHF to support multimodal RL training for reproducing DeepSeek-R1 on multimodal tasks, achieving significant speedups compared to other implementations.
- **R1-V**: Focuses on reinforcing super generalization abilities in vision-language models (VLMs) with minimal cost, demonstrating the effectiveness of RLVR.
- **open-r1-multimodal**: A fork of open-r1 that integrates multimodal training capabilities, supporting GRPO and other RL algorithms for multimodal tasks.
- **R1-Multimodal-Journey**: Chronicles large-scale experiments and training efficiency improvements using vLLM, emphasizing early "aha moments" in challenging geometry problems with R1-like reinforcement learning.

---

## Benchmarks

- **MME-CoT**: A benchmark designed to evaluate chain-of-thought (CoT) reasoning in large multimodal models. It covers six domains and introduces novel metrics for assessing reasoning quality, robustness, and efficiency.

---

## System Optimization

- **Unsloth**: A guide for training R1 reasoning models locally using GRPO. It highlights VRAM efficiency improvements and supports a variety of model architectures.

---

## SFT Methods

- **LIMO: Less is More for Reasoning**: A study revealing that complex mathematical reasoning can be effectively elicited with surprisingly few examples, challenging the assumption that massive data is required for sophisticated reasoning.
- **SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training**: An investigation comparing supervised fine-tuning (SFT) and reinforcement learning (RL), showing that RL—especially with outcome-based rewards—generalizes better across rule-based textual and visual variants, while SFT often overfits the training data.
- **LIMA: Less Is More for Alignment**: Introduces a 65B parameter LLaMa model fine-tuned with only 1,000 carefully curated prompt-response pairs. Without relying on reinforcement learning or human preference modeling, LIMA achieves strong performance and generalization.

---

## Papers

- **Demystifying Long Chain-of-Thought Reasoning in LLMs**: This study systematically investigates the mechanisms behind long chain-of-thought reasoning, outlining key insights regarding SFT, compute scaling, reward shaping, and inherent model abilities like error correction.
- **LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs**: Proposes a comprehensive framework for advancing step-by-step visual reasoning. It introduces a visual reasoning benchmark with eight categories, a novel assessment metric, and the LlamaV-o1 model trained via a multi-step curriculum.
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**: Presents a novel architecture that scales test-time computation through iterative latent-space reasoning. The approach, demonstrated on a 3.5B parameter model trained on 800B tokens, improves performance on various reasoning benchmarks without requiring specialized training data.

---

欢迎大家参与讨论、提交 issue 或 pull request，一起推进 DeepSeek R1 相关技术的发展！
