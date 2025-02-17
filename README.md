# 🚀 Awesome-DeepSeek-R1-Reproduction

A curated collection of cutting-edge projects, benchmarks, and research papers dedicated to reproducing and advancing the DeepSeek R1 framework. This repository brings together innovative efforts in large language models (LLMs) and multimodal LLMs (MLLMs), providing state-of-the-art implementations, comprehensive evaluations, and meticulously crafted training recipes to elevate reasoning capabilities.

## 📑 Table of Contents

- [LLM Initiatives](#llm-initiatives)
- [MLLM Initiatives](#mllm-initiatives)
- [Benchmarks](#benchmarks)
- [System Optimization](#system-optimization)
- [SFT Approaches](#sft-approaches)
- [Key Papers](#key-papers)

---

## 🤖 LLM Initiatives

- **SimpleRL-reason**: A straightforward yet powerful reinforcement learning recipe designed to enhance the reasoning prowess of language models.
- **TinyZero**: An accessible and minimal reproduction of DeepSeek R1-Zero, focusing on countdown and multiplication tasks with reinforcement learning.
- **open-r1**: A fully open-source reproduction of DeepSeek-R1, complete with scripts for both training and evaluation using SFT and GRPO.
- **Logic-RL**: Implements DeepSeek R1 Zero on challenging logic puzzles, showcasing enhanced reasoning through rule-based reinforcement learning techniques.
- **DeepScaleR**: Surpasses the O1 preview with a 1.5B model by harnessing reinforcement learning and advanced scaling methods.
- **K1.5**: Outlines the training recipe and system architecture of Kimi K1.5, a multi-modal LLM that leverages long-context scaling and refined policy optimization to achieve state-of-the-art performance across diverse benchmarks.

---

## 🖼️ MLLM Initiatives

- **lmm-r1**: Extends the OpenRLHF framework to support multimodal reinforcement learning, enabling the reproduction of DeepSeek-R1 on multimodal tasks with impressive speed improvements.
- **R1-V**: Enhances the generalization capabilities of vision-language models (VLMs) with minimal resource overhead, showcasing the power of RLVR.
- **open-r1-multimodal**: A fork of open-r1, enriched with multimodal training capabilities and support for GRPO alongside other reinforcement learning algorithms.
- **R1-Multimodal-Journey**: Chronicles large-scale experiments and optimizations using vLLM, emphasizing early breakthroughs in complex geometry challenges with R1-inspired reinforcement learning.

---

## 📊 Benchmarks

- **MME-CoT**: A robust benchmark designed to evaluate chain-of-thought (CoT) reasoning in large multimodal models. It spans six domains and introduces novel metrics to assess reasoning quality, robustness, and efficiency.

---

## ⚙️ System Optimization

- **Unsloth**: A comprehensive guide to training R1 reasoning models locally with GRPO, focusing on VRAM efficiency and compatibility with various model architectures.

---

## 🔧 SFT Approaches

- **LIMO: Less is More for Reasoning**: A pioneering study demonstrating that complex mathematical reasoning can be effectively elicited with minimal examples, challenging the notion that massive datasets are required for sophisticated reasoning.
- **SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training**: An in-depth analysis contrasting supervised fine-tuning (SFT) with reinforcement learning (RL), revealing that RL—especially when coupled with outcome-based rewards—yields superior generalization across both textual and visual rule-based tasks.
- **LIMA: Less Is More for Alignment**: Introduces a 65B-parameter LLaMa model fine-tuned with only 1,000 meticulously curated prompt-response pairs. This approach, devoid of reinforcement learning or human preference modeling, achieves exceptional performance and generalization.

---

## 📚 Key Papers

- **Demystifying Long Chain-of-Thought Reasoning in LLMs**: This seminal study unpacks the intricate mechanics of long chain-of-thought reasoning, highlighting the roles of SFT, compute scaling, reward shaping, and intrinsic model capabilities like error correction.
- **LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs**: Proposes a comprehensive framework for advancing sequential visual reasoning, introducing a novel visual reasoning benchmark with eight categories, an innovative assessment metric, and the LlamaV-o1 model trained via a multi-step curriculum.
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**: Presents an avant-garde architecture that scales test-time computation by iteratively reasoning in latent space. Demonstrated on a 3.5B-parameter model trained on 800B tokens, this approach delivers significant performance gains on complex reasoning benchmarks without the need for specialized training data.

---

We invite you to **explore**, **contribute**, and **collaborate** in advancing the frontier of DeepSeek R1 technology. Join the conversation by submitting issues or pull requests, and help shape the future of reasoning in language and multimodal models!
