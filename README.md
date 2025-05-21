<div align="center">
<h1>  UniVG-R1:
Reasoning Guided Universal Visual Grounding with Reinforcement Learning </h1>
<div align=center>
<img width="650" alt="image" src="figs/teaser.jpg">
</div>
<br>
<a href='https://arxiv.org/abs/2505.14231'><img src='https://img.shields.io/badge/Arxiv-2505.14231-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://amap-ml.github.io/UniVG-R1-page/'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a> -->

<br>
<div>
<a href="https://sulebai.github.io/">Sule Bai</a><sup>1,2</sup>,
<a href="https://scholar.google.com/citations?user=-pfkprkAAAAJ&hl=zh-CN&oi=ao" target="_blank">Mingxing Li</a><sup>2</sup>,
<a href="https://yongliu20.github.io/">Yong Liu</a><sup>1</sup>,
<a href="" target="_blank">Jing Tang</a><sup>2</sup>,
<a href="https://zhang9302002.github.io/">Haoji Zhang</a><sup>1</sup>,
<a href="" target="_blank">Lei Sun</a><sup>2</sup>,
<a href="https://cxxgtxy.github.io/">Xiangxiang Chu</a><sup>2</sup>,
<a href="https://andytang15.github.io/">Yansong Tang</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>Tsinghua University
    <sup>2</sup>ALibaba Group
</div>
</div>

## ToDo
- [ ] The code, models, and data will be released soon after a legal inspection.
- [x] We release our paper in [arxiv](https://arxiv.org/abs/2505.14231).

## 📖 Overview
1. We propose UniVG-R1, a reasoning guided MLLM for universal visual grounding, which employs GRPO with a cold-start initialization to effectively enhance reasoning capabilities across multimodal contexts.
2. A high-quality CoT dataset is introduced, encompassing diverse tasks, each meticulously annotated with detailed reasoning chains to facilitate advanced reasoning-based grounding.
3. We identify a difficulty bias in GRPO training, and propose a difficulty-aware weight adjustment strategy. Experiments validate that GRPO equipped with this strategy consistently enhance the model performance.
4. Extensive experiments demonstrate that our model achieves state-of-the-art performance across multiple grounding benchmarks, showcasing its versatility and generalizability.
<div align=center>
<img width="650" alt="image" src="figs/pipeline.jpg">
</div>

## 📈Results
<div align=center>
<img width="650" alt="image" src="figs/result1.png">
</div>
<div align="center">
Performance on the MIG-Bench.
</div>
<div align=center>
<img width="650" alt="image" src="figs/result2.png">
</div>
<div align="center">
Zero-shot performance on several reasoning grounding benchmarks.
</div>

## 🌹 Acknowledgement
Our work is primarily based on [Migician](https://github.com/thunlp/Migician), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We are sincerely grateful for their excellent works.

