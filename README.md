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
<a href='https://huggingface.co/GD-ML/UniVG-R1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/GD-ML/UniVG-R1-data'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://huggingface.co/spaces/SuleBai/UniVG-R1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>

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

## 🎯 ToDo
- [x] We release the demo, check it [here](https://huggingface.co/spaces/SuleBai/UniVG-R1). 🔥🔥🔥
- [x] We release our code.
- [x] We release our model and dataset in the Hugging Face.
- [x] We release our paper in [arxiv](https://arxiv.org/abs/2505.14231).

## 📖 Overview
1. We propose UniVG-R1, a reasoning guided MLLM for universal visual grounding, which employs GRPO with a cold-start initialization to effectively enhance reasoning capabilities across multimodal contexts.
2. A high-quality CoT dataset is introduced, encompassing diverse tasks, each meticulously annotated with detailed reasoning chains to facilitate advanced reasoning-based grounding.
3. We identify a difficulty bias in GRPO training, and propose a difficulty-aware weight adjustment strategy. Experiments validate that GRPO equipped with this strategy consistently enhance the model performance.
4. Extensive experiments demonstrate that our model achieves state-of-the-art performance across multiple grounding benchmarks, showcasing its versatility and generalizability.
<div align=center>
<img width="650" alt="image" src="figs/pipeline.jpg">
</div>

## 🛠️ Installation
Our code is based on [VLM-R1](https://github.com/om-ai-lab/VLM-R1), please follow their instrcutions.
```bash
conda create -n univg-r1 python=3.10
conda activate univg-r1
bash setup.sh
```

## 💪🏻 Training
#### 📚 Stage 1 CoT-SFT

1. Download the [Stage 1 CoT-SFT Data](https://huggingface.co/datasets/GD-ML/UniVG-R1-data/blob/main/stage1_cotsft.json) and the [MGrounding-630k dataset](https://github.com/thunlp/Migician).
2. We use the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning. Please follow their instructions to set up the environment.

```bash
llamafactory-cli train sft/stage1_cotsft.yaml
```
#### 📚 Stage 2 GRPO

1. Download the [Stage 2 RL Data](https://huggingface.co/datasets/GD-ML/UniVG-R1-data/blob/main/stage2_rl.json) and the [MGrounding-630k dataset](https://github.com/thunlp/Migician/tree/main).

```bash
bash src/open-r1-multimodal/run_scripts/run_grpo_univg.sh
```

## 📊 Evaluation
For evaluation on the [MIG-bench](https://github.com/thunlp/Migician), please refer to their instructions to set up the environment.

And we provide the UniVG-R1 scripts in the `eval` folder. Please replace them in the [Migician](https://github.com/thunlp/Migician) codebase.

## 📈 Results
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

## 📚 Citation

If you find our paper and code helpful for your research, please consider starring our repository ⭐ and citing our work ✏️.
```bibtex
@article{bai2025univg,
  title={UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning},
  author={Bai, Sule and Li, Mingxing and Liu, Yong and Tang, Jing and Zhang, Haoji and Sun, Lei and Chu, Xiangxiang and Tang, Yansong},
  journal={arXiv preprint arXiv:2505.14231},
  year={2025}
}
```

