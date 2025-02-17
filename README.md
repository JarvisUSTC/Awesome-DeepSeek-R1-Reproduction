# 🚀 Awesome-DeepSeek-R1-Reproduction

A curated collection of cutting-edge projects, benchmarks, and research papers dedicated to reproducing and advancing the DeepSeek R1 framework. This repository brings together innovative efforts in large language models (LLMs) and multimodal LLMs (MLLMs), providing state-of-the-art implementations, comprehensive evaluations, and meticulously crafted training recipes to elevate reasoning capabilities.

## 📑 Table of Contents

- [LLM (Large Language Model) Related](#llm-large-language-model-related)
- [MLLM (Multimodal Large Language Model) Related](#mllm-multimodal-large-language-model-related)
- [Benchmarks](#benchmarks)
- [System Optimization](#system-optimization)
- [SFT Methods](#sft-methods)
- [Papers](#papers)

---

## 🤖 LLM (Large Language Model) Related

- [**SimpleRL-reason**](https://github.com/hkust-nlp/simpleRL-reason)  
  A simple reinforcement learning recipe to improve models' reasoning abilities.

- [**TinyZero**](https://github.com/Jiayi-Pan/TinyZero)  
  A clean, minimal, and accessible reproduction of DeepSeek R1-Zero focusing on countdown and multiplication tasks using reinforcement learning.

- [**open-r1**](https://github.com/huggingface/open-r1)  
  A fully open reproduction of DeepSeek-R1, including scripts for training and evaluating models with SFT and GRPO.

- [**Logic-RL**](https://github.com/Unakar/Logic-RL)  
  Reproduces DeepSeek R1 Zero on logic puzzles, demonstrating enhanced reasoning capabilities with rule-based RL.

- [**DeepScaleR**](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)  
  Discusses surpassing O1 preview with a 1.5B model by scaling reinforcement learning, focusing on advancements in model scaling and RL techniques.

- [**K1.5**](https://arxiv.org/abs/2501.12599)  
  Presents the training recipe and system design of Kimi K1.5, a multi-modal LLM trained with reinforcement learning. It highlights the effectiveness of long context scaling and improved policy optimization in achieving state-of-the-art reasoning performance across multiple benchmarks and modalities.

---

## 🖼️ MLLM (Multimodal Large Language Model) Related

- [**lmm-r1**](https://github.com/TideDra/lmm-r1)  
  Extends OpenRLHF to support LMM RL training for reproducing DeepSeek-R1 on multimodal tasks, achieving significant speedup compared to other implementations.

- [**R1-V**](https://github.com/Deep-Agent/R1-V)  
  Focuses on reinforcing super generalization ability in vision language models (VLMs) with minimal cost, demonstrating the effectiveness of RLVR.

- [**open-r1-multimodal**](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)  
  A fork of open-r1 to add multimodal model training capabilities, supporting GRPO and other RL algorithms for multimodal tasks.

- [**R1-Multimodal-Journey**](https://github.com/FanqingM/R1-Multimodal-Journey)  
  A journey to real multimodal R1, focusing on large-scale experiments and improvements in training efficiency using vLLM. It explores the effectiveness of R1-like reinforcement learning on challenging geometry problems and highlights the "aha moment" in early training stages.

---

## 📊 Benchmarks

- [**MME-CoT**](https://arxiv.org/abs/2502.09621)  
  A benchmark for evaluating chain-of-thought reasoning in large multimodal models, covering six domains and providing novel metrics for reasoning quality, robustness, and efficiency.

---

## ⚙️ System Optimization

- [**Unsloth**](https://unsloth.ai/blog/r1-reasoning)  
  Provides a guide to training R1 reasoning models locally using GRPO, highlighting improvements in VRAM efficiency and support for various models.

---

## 🔧 SFT Methods

- [**LIMO: Less is More for Reasoning**](https://arxiv.org/abs/2502.03387)  
  Presents a fundamental discovery that challenges the understanding of how complex reasoning emerges in large language models. It demonstrates that complex mathematical reasoning abilities can be effectively elicited with surprisingly few examples, challenging the assumption of massive data requirements and the belief that supervised fine-tuning primarily leads to memorization rather than generalization.

- [**SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training**](http://arxiv.org/abs/2501.17161)  
  This paper studies the difference between supervised fine-tuning (SFT) and reinforcement learning (RL) in terms of generalization and memorization. The study shows that RL, especially when trained with an outcome-based reward, generalizes across both rule-based textual and visual variants, while SFT tends to memorize training data and struggles to generalize out-of-distribution scenarios.

- [**LIMA: Less Is More for Alignment**](http://arxiv.org/abs/2305.11206)  
  This paper introduces LIMA, a 65B parameter LLaMa language model fine-tuned with only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. LIMA demonstrates strong performance and generalization, suggesting that large language models can achieve high-quality output with minimal instruction tuning data.

---

## 📚 Papers

- [**Demystifying Long Chain-of-Thought Reasoning in LLMs**](http://arxiv.org/abs/2502.03373)  
  This study systematically investigates the mechanics of long chain-of-thought (CoT) reasoning in large language models (LLMs), identifying key factors that enable models to generate long CoT trajectories. The authors present four main findings: (1) Supervised fine-tuning (SFT) is not strictly necessary but simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL, with noisy, web-extracted solutions showing strong potential for out-of-distribution tasks; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute.

- [**LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLM**](http://arxiv.org/abs/2501.06186)  
  This paper proposes a comprehensive framework for advancing step-by-step visual reasoning in large language models (LMMs). It introduces a visual reasoning benchmark with eight categories, a novel metric for assessing visual reasoning quality, and a new multimodal visual reasoning model named LlamaV-o1. The model is trained using a multi-step curriculum learning approach and outperforms existing open-source models in multi-step reasoning tasks.

- [**Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**](http://arxiv.org/abs/2502.05171)  
  This paper introduces a novel language model architecture that scales test-time computation by implicitly reasoning in latent space. The model iterates a recurrent block, allowing it to unroll to arbitrary depth at test-time. This approach does not require specialized training data, can work with small context windows, and can capture reasoning types not easily represented in words. The authors scale a proof-of-concept model to 3.5 billion parameters and 800 billion tokens, showing improved performance on reasoning benchmarks.

---

We invite you to **explore**, **contribute**, and **collaborate** in advancing the frontier of DeepSeek R1 technology. Join the conversation by submitting issues or pull requests, and help shape the future of reasoning in language and multimodal models!
