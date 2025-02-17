# 🚀 Awesome-DeepSeek-R1-Reproduction

A curated collection of cutting-edge projects, benchmarks, and research papers dedicated to reproducing and advancing the DeepSeek R1 framework. This repository brings together innovative efforts in large language models (LLMs) and multimodal LLMs (MLLMs), offering state-of-the-art implementations, comprehensive evaluations, and meticulously designed training recipes to enhance reasoning capabilities.

---

## 📑 Table of Contents

- [LLM (Large Language Model) Projects](#llm-large-language-model-projects)
- [MLLM (Multimodal Large Language Model) Projects](#mllm-multimodal-large-language-model-projects)
- [Benchmarks](#benchmarks)
- [System Optimization](#system-optimization)
- [SFT Methods](#sft-methods)
- [Research Papers](#research-papers)

---

## 🤖 LLM (Large Language Model) Projects

- **[SimpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)**  
  A straightforward reinforcement learning recipe designed to enhance model reasoning abilities.

- **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)**  
  A minimal and accessible reproduction of DeepSeek R1-Zero, focusing on reinforcement learning for countdown and multiplication tasks.

- **[open-r1](https://github.com/huggingface/open-r1)**  
  A fully open-source reproduction of DeepSeek-R1, complete with scripts for training and evaluation using SFT and GRPO.

- **[Logic-RL](https://github.com/Unakar/Logic-RL)**  
  A reproduction of DeepSeek R1 Zero applied to logic puzzles, leveraging rule-based reinforcement learning for improved reasoning.

- **[DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)**  
  A discussion on surpassing the O1 preview with a 1.5B model through reinforcement learning and model scaling advancements.

- **[K1.5](https://arxiv.org/abs/2501.12599)**  
  Presents the training recipe and system design of Kimi K1.5, a multimodal LLM trained with reinforcement learning, emphasizing long-context scaling and improved policy optimization.

---

## 🖼️ MLLM (Multimodal Large Language Model) Projects

- **[lmm-r1](https://github.com/TideDra/lmm-r1)**  
  Extends OpenRLHF to support LMM RL training for DeepSeek-R1 multimodal tasks, achieving significant speedups.

- **[R1-V](https://github.com/Deep-Agent/R1-V)**  
  Focuses on reinforcing generalization in vision-language models (VLMs) with minimal cost using RLVR.

- **[open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)**  
  A fork of open-r1 adding multimodal training capabilities, supporting GRPO and other RL algorithms.

- **[R1-Multimodal-Journey](https://github.com/FanqingM/R1-Multimodal-Journey)**  
  Explores R1-like reinforcement learning for complex multimodal tasks, focusing on training efficiency and real-world applications.

---

## 📊 Benchmarks

- **[MME-CoT](https://arxiv.org/abs/2502.09621)**  
  A benchmark for evaluating chain-of-thought (CoT) reasoning in multimodal models, spanning six domains with novel metrics.

---

## ⚙️ System Optimization

- **[Unsloth](https://unsloth.ai/blog/r1-reasoning)**  
  A guide for training R1 reasoning models locally using GRPO, with a focus on VRAM efficiency and model compatibility.

---

## 🔧 SFT Methods

- **[LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)**  
  Challenges the assumption that massive datasets are required for complex reasoning, showing that fewer examples can suffice.

- **[SFT Memorizes, RL Generalizes](http://arxiv.org/abs/2501.17161)**  
  A comparative study highlighting how reinforcement learning yields superior generalization compared to supervised fine-tuning (SFT).

- **[LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206)**  
  Introduces LIMA, a 65B LLaMa model fine-tuned with only 1,000 curated prompts, achieving strong performance with minimal instruction tuning.

---

## 📚 Research Papers

- **[Demystifying Long Chain-of-Thought Reasoning in LLMs](http://arxiv.org/abs/2502.03373)**  
  Investigates the emergence of long chain-of-thought reasoning, revealing key insights into supervised fine-tuning, training compute, and reward shaping.

- **[LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs](http://arxiv.org/abs/2501.06186)**  
  Presents a framework for sequential visual reasoning, introducing benchmarks, evaluation metrics, and a novel training approach.

- **[Scaling up Test-Time Compute with Latent Reasoning](http://arxiv.org/abs/2502.05171)**  
  Proposes a recurrent-depth approach to enhance test-time reasoning efficiency, allowing scalable, implicit latent-space reasoning.

---

We warmly invite you to **explore**, **contribute**, and **collaborate** in advancing the frontier of DeepSeek R1 technology. Feel free to submit issues or pull requests and help shape the future of reasoning in language and multimodal models!

