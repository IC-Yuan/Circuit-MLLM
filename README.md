<div align="center">

# Circuit-MLLM

### Topological Logic-Guided Latent-Space Visual Reasoning for Circuit Schematic Understanding

**Jinyuan Deng, Yuqi Jiang, Wenjing Huang, Xin Li, Qi Sun, and Cheng Zhuo**<br>
Zhejiang University, Hangzhou, China

[![Project](https://img.shields.io/badge/Project-Circuit--MLLM-1f6feb)](https://github.com/IC-Yuan/Circuit-MLLM)
[![arXiv](https://img.shields.io/badge/arXiv-2609.15668-b31b1b)](https://arxiv.org/abs/2609.15668)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab)](https://www.python.org/)
[![Backbone](https://img.shields.io/badge/Backbone-Qwen2.5--VL-8a2be2)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

</div>

## Timeline

- **2026-09-11**: Reorganized the public release with reproducible path configuration, launch scripts, and architecture/result figures.
- **2026-09-13**: Added the staged-release note.
- **Coming soon**: Clean up and selectively release the remaining core topology-guided training implementation, followed by finalized dataset/checkpoint instructions and a compact end-to-end reproduction recipe.

## Overview

Circuit schematics contain long wires, junctions, branches, and irregular spatial layouts. Cropping can break connectivity, while ordinary patch order does not match the order used to reason through a circuit. Circuit-MLLM addresses these issues with three components:

1. **Latent-space circuit knowledge mining.** HAWP captures holistic wireframes and junctions, DeepLSD captures line segments and local topology, and DINOv2 supplies patch-level semantics. Their aligned features are projected and fused with the MLLM visual features.
2. **Topology-guided sequencing.** A pixel-level topological logic mask follows the root-component, wire-path, target-component order. Valid patch features are sorted by this mask and pooled into a fixed number of latent targets.
3. **Text-latent joint supervision.** Cross-entropy trains the surrounding text while a cosine alignment objective supervises generated latent tokens with the topology-ordered visual targets.

<p align="center">
  <img src="assets/circuit_mllm_overview.png" width="92%" alt="Circuit-MLLM overall architecture">
</p>

## Results

**Table 1. Performance across all Circuit-MLLM-bench task categories.**

<p align="center">
  <img src="assets/benchmark_results.png" width="92%" alt="Circuit-MLLM-bench results">
</p>

**Table 2. Performance on topology analysis tasks by difficulty.**

<p align="center">
  <img src="assets/topology_results.png" width="92%" alt="Topology task results by difficulty">
</p>

## Installation

The reference environment uses Linux, Python 3.11, CUDA 12.4, PyTorch 2.6.0, and FlashAttention 2.7.4. A recent CUDA GPU is required for training.

```bash
git clone https://github.com/IC-Yuan/Circuit-MLLM.git
cd Circuit-MLLM

conda create -n circuit-mllm python=3.11 -y
conda activate circuit-mllm

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install the bundled, project-modified Transformers package and visual experts.
pip install -e ./transformers
pip install -e ./hawp
pip install -e ./DeepLSD
```

Download the HAWP and DeepLSD checkpoints:

```bash
bash scripts/download_expert_weights.sh
```

DINOv2 defaults to `facebook/dinov2-giant` and is downloaded through Hugging Face. Set `DINOV2_MODEL` to a local directory for offline use.

## Data

Training expects one JSONL object per line. Image paths may be absolute, or relative to `CIRCUIT_DATA_ROOT`. A topology-guided example has the following shape:

```json
{
  "text_input": "What devices are connected to M1?",
  "image_input": ["images/example.png"],
  "sequence_plan": [
    {"type": "text", "content": "Trace the branches from M1."},
    {
      "type": "latent",
      "helper_image": "helpers/example_step_1.png",
      "mask_vector_file": "masks/example_step_1.npy"
    },
    {"type": "text", "content": "The final answer is ..."}
  ]
}
```

Recommended layout:

```text
circuit_data/
├── combine/
│   └── train.jsonl
├── images/
├── helpers/
└── masks/
```

The `helper_image` and `mask_vector_file` entries are training-only supervision. The mask is a NumPy `float32` array aligned with the corresponding helper image.

## Configuration

Copy the example and edit only local resource paths:

```bash
cp .env.example .env
```

Required settings:

| Variable | Purpose | Example |
|---|---|---|
| `CIRCUIT_DATA_ROOT` | Root for relative training paths | `/path/to/circuit_data` |
| `DATA_PATH` | Training JSONL | `/path/to/circuit_data/combine/train.jsonl` |
| `MODEL_NAME` | Hugging Face model ID or local model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `CUDA_VISIBLE_DEVICES` | GPUs exposed to training | `0,1,2,3` |
| `NUM_PROCESSES` | Number of training processes | `4` |
| `OUTPUT_ROOT` | Checkpoints and logs | `./outputs` |

Expert checkpoint paths already default to the locations created by `scripts/download_expert_weights.sh`. `.env` is ignored by Git, so machine-specific paths stay local.

## Training

Launch topology-guided, multi-expert training with:

```bash
bash run_training_circuit_sequence_circuit_expert.sh
```

Important defaults are `LATENT_SIZE=4`, `SIM_WEIGHT=0.6`, batch size 1 per device, gradient accumulation 8, and DeepSpeed ZeRO-2. Override any value through `.env` or the shell:

```bash
LATENT_SIZE=8 SIM_WEIGHT=0.4 MASK_NOISE_RATIO=0.05 \
  bash run_training_circuit_sequence_circuit_expert.sh
```

The launcher writes under `OUTPUT_ROOT`, validates the expert checkpoints, and automatically resumes from the latest checkpoint in the output directory.

## Evaluation

Evaluation JSONL files are grouped in one directory. Images referenced by relative paths are resolved under a separate image root:

```bash
bash scripts/evaluate.sh \
  connection_identification_ours \
  /path/to/circuit-mllm-checkpoint \
  /path/to/evaluation-jsonl-directory \
  /path/to/evaluation-images \
  0
```

Supported task names are:

```text
connection_identification[_ours]
connection_judge[_ours]
element_classification[_ours]
total_counting[_ours]
type_wise_counting[_ours]
```

Results are saved to `outputs/evaluation/<task>/` unless `OUTPUT_DIR` is set.

Example connection-identification question:

<p align="center">
  <img src="assets/evaluation_connection_example.png" width="360" alt="Circuit schematic with M1, M2, M3, current source I1, and Vout">
</p>

**Question:** Which of the following are connected to the upper terminal of current source I1? Select all that apply: A. M1; B. M2; C. M3; D. Vout.

**Answer:** B, C, D (M2, M3, and Vout). The corresponding evaluation record is in [examples/evaluation/connection_identification.jsonl](examples/evaluation/connection_identification.jsonl).

## Analysis

Circuit-MLLM's latent attention moves from a queried root component, through connected wires, toward downstream targets. This ordering is more consistent with circuit connectivity than ordinary raster scanning.

<p align="center">
  <img src="assets/attention_analysis.png" width="88%" alt="Latent-space attention comparison">
</p>

<p align="center">
  <img src="assets/latent_shift_analysis.png" width="92%" alt="Latent-space visual shift analysis">
</p>

## Acknowledgements

This repository builds on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Transformers](https://github.com/huggingface/transformers), [HAWP](https://github.com/cherubicXN/hawp), [DeepLSD](https://github.com/cvg/DeepLSD), [ILVR](https://github.com/UMass-Embodied-AGI/), and [Mirage](https://github.com/UMass-Embodied-AGI/Mirage). We thank the authors of these projects.

## Citation

Please cite the paper:

```bibtex
@misc{deng2026circuitmllm,
  title  = {Circuit-MLLM: Topological Logic-Guided Latent-Space Visual Reasoning for Circuit Schematic Understanding},
  author = {Deng, Jinyuan and Jiang, Yuqi and Huang, Wenjing and Li, Xin and Sun, Qi and Zhuo, Cheng},
  year   = {2026},
  eprint = {2609.15668},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url    = {https://arxiv.org/abs/2609.15668}
}
```
