# 🚀 Awesome-DeepSeek-R1-Reproduction

A curated collection of cutting-edge projects, benchmarks, and research papers dedicated to reproducing and advancing the DeepSeek R1 framework. This repository brings together innovative efforts in large language models (LLMs) and multimodal LLMs (MLLMs), offering state-of-the-art implementations, comprehensive evaluations, and meticulously designed training recipes to enhance reasoning capabilities.

---

## 📑 Table of Contents

- [🚀 Awesome-DeepSeek-R1-Reproduction](#-awesome-deepseek-r1-reproduction)
  - [📑 Table of Contents](#-table-of-contents)
  - [🤖 LLM (Large Language Model) Projects](#-llm-large-language-model-projects)
  - [🖼️ MLLM (Multimodal Large Language Model) Projects](#️-mllm-multimodal-large-language-model-projects)
  - [📊 Benchmarks](#-benchmarks)
  - [⚙️ System Optimization](#️-system-optimization)
  - [🔧 SFT Methods](#-sft-methods)
  - [📚 Research Papers](#-research-papers)
  - [🌟 About Vision2Mind](#-about-vision2mind)

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

- **[X-R1](https://github.com/dhcode-cpp/X-R1)**  
Aims to build an easy-to-use, low-cost training framework based on end-to-end reinforcement learning to accelerate the development of Scaling Post-Training. Inspired by DeepSeek-R1 and open-r1, X-R1 focuses on minimal-cost training for 0.5B R1-Zero models, supporting LoRA and larger models up to 32B parameters.

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

- **[r1-onevision](https://yangyi-vai.notion.site/r1-onevision)**
  A multimodal reasoning model that bridges multimodal capabilities and deep reasoning, achieving state-of-the-art performance on challenging benchmarks by using a formal language-driven visual reasoning process and Rule-based Reinforcement Learning (ongoing), and has developed a new benchmark called R1-Onevision-bench.

- **[open-r1-video](https://github.com/Wang-Xiaodong1899/Open-R1-Video)**
  A multimodal video reasoning model that extends the open-r1 framework to video understanding tasks.

- **[Seg-Zero](https://github.com/dvlab-research/Seg-Zero)**
  Reasoning-chain guided segmentation framework via GRPO.

- **[Visual Reinforcement Fine-tuning](https://arxiv.org/abs/2503.01785)**  
  A framework for visual reinforcement learning that combines visual grounding and reinforcement learning to enhance MLLM performance in visual tasks.

- **[MM-Eureka](https://github.com/ModalMinds/MM-EUREKA)**  
  Present MM-Eureka and MM-Eureka-Zero, a series of multimodal reasoning models that successfully extend large-scale rule-based reinforcement learning to multimodal reasoning.

- **[LMM-R1](http://arxiv.org/abs/2503.07536)**
  Experiments on Qwen2.5-VL-Instruct-3B, and proposes a two-stage rule-based reinforcement learning method to empower 3B LLMs with strong reasoning abilities.

- **[R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model](http://arxiv.org/abs/2503.05132)**
  Experiments on Qwen2-VL-Base-2B and present challenges when training on instruct model.

- **[Boosting the Generalization and Reasoning of Vision Language Models with  Curriculum Reinforcement Learning](http://arxiv.org/abs/2503.07065)**
  Proposes two approaches, Curriculum Reinforcement Learning and Rejected Sampling-based Self-improvement, to enhance the generalization and reasoning of vision language models.

- **[DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding](http://arxiv.org/abs/2503.12797)**
  Proposes DeepPerception, a novel cognitive visual perception framework that enhances MLLMs' reasoning capabilities for knowledge-intensive visual grounding tasks.
  
---

## 📊 Benchmarks

- **[MME-CoT](https://arxiv.org/abs/2502.09621)**  
  A benchmark for evaluating chain-of-thought (CoT) reasoning in multimodal models, spanning six domains with novel metrics.

- **[EnigmaEVAL](https://arxiv.org/pdf/2502.08859)**
  A dataset derived from puzzle competitions, containing 1184 puzzles that test models' implicit knowledge synthesis and multi-step deductive reasoning, revealing current models' limitations in unstructured and lateral reasoning tasks, with state-of-the-art models achieving extremely low accuracy on these puzzles.

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

- **[Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)**
  Detailed Introduction to the Training Process of the Kimi Multimodal Inference Model.

- **[LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs](http://arxiv.org/abs/2501.06186)**  
  Presents a framework for sequential visual reasoning, introducing benchmarks, evaluation metrics, and a novel training approach.

- **[Scaling up Test-Time Compute with Latent Reasoning](http://arxiv.org/abs/2502.05171)**  
  Proposes a recurrent-depth approach to enhance test-time reasoning efficiency, allowing scalable, implicit latent-space reasoning.

- **[MM-RLHF](https://arxiv.org/pdf/2502.10391)**
  Introduces MM-RLHF, a dataset containing 120k fine-grained, human-annotated preference comparison pairs, and proposes key innovations to improve reward models and alignment algorithms.

- **[ReasonFlux](https://arxiv.org/pdf/2502.06772v1)**
  Introduces ReasonFlux, a hierarchical LLM reasoning framework that optimizes the reasoning search space through structured thought templates, achieving state-of-the-art performance in mathematical reasoning benchmarks.

- **[The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://github.com/AlexCuadron/Overthinking)**
  Introduces Overthinking, a systematic evaluation framework that automatically rates overthinking behavior in large language models, focusing on detecting when models prefer their internal reasoning chain over interacting with the environment.
---

## 🌟 About Vision2Mind

**Vision2Mind** is a pioneering startup dedicated to advancing the frontiers of artificial intelligence, with a particular focus on vision-language models and multimodal reasoning. Our mission is to develop cutting-edge technologies that bridge the gap between human cognition and artificial intelligence, enabling more intuitive and effective human-machine interactions.

At Vision2Mind, we are committed to:

- **Innovation**: Pushing the boundaries of AI research to create novel solutions for complex problems.
- **Excellence**: Delivering high-quality, reliable, and scalable AI models and applications.
- **Collaboration**: Working with the global AI community to foster open-source development and knowledge sharing.
- **Impact**: Making a tangible difference in industries such as healthcare, education, and finance through AI-driven innovations.

We invite researchers, developers, and enthusiasts to join us in our journey to transform the AI landscape. For more information, visit our website: [https://vision2mind.wegic.app/home](https://vision2mind.wegic.app/home).

---

We warmly invite you to **explore**, **contribute**, and **collaborate** in advancing the frontier of DeepSeek R1 technology. Feel free to submit issues or pull requests and help shape the future of reasoning in language and multimodal models!
