# Environment Setup

This project runs in two environments:

- **Mac** — for local development, pipeline debugging, and small-scale tests.
- **HPC cluster** — for the full experimental runs on Llama-3.1-8B.

The dependencies are split across three files so that both environments share the same core packages but install PyTorch differently.

| File | Purpose |
|------|---------|
| `requirements.txt` | Core dependencies, identical across Mac and HPC |
| `requirements-mac.txt` | CPU-only PyTorch for Mac (Apple Silicon or Intel) |
| `requirements-hpc.txt` | Quantization support for GPU clusters (PyTorch installed separately with CUDA build) |

---

## Setup on Mac (local development)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
pip install -r requirements.txt
```

Do not try to load Llama-3.1-8B on your Mac. It will not fit in RAM on most machines and will be unusably slow. Use your Mac for:

- Pipeline development and unit tests with `gpt2-small` as a stand-in.
- LLM-as-judge prompt iteration (API calls, no local model needed).
- Analysis and plotting once activation CSVs and results are downloaded from the cluster.
- Writing.

Expect to do ~80% of your code development on Mac and ~20% on the cluster running the actual experiments.

---

## Setup on the HPC cluster

Check what CUDA version is available on your cluster:

```bash
nvidia-smi
nvcc --version
module avail cuda
```

Load the appropriate modules (syntax varies by cluster; ask your sysadmin if unsure):

```bash
module load python/3.11
module load cuda/12.1
```

Create and activate the environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install PyTorch with the CUDA build matching the cluster:

```bash
# For CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install -r requirements-hpc.txt
pip install -r requirements.txt
```

Verify GPU visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Should print `True` and a number ≥ 1.

---

## GPU memory guidance for Llama-3.1-8B

| Precision | VRAM needed | Recommended GPUs |
|-----------|-------------|------------------|
| bf16 / fp16 | ~16 GB | A100, H100, RTX 4090 |
| 8-bit (bitsandbytes) | ~10 GB | A40, RTX 3090 |
| 4-bit (bitsandbytes) | ~5 GB | RTX 3080, V100 16GB |

If your cluster has V100 16GB or similar, use 4-bit quantization by loading the model with `load_in_4bit=True` in the HuggingFace `from_pretrained` call.

---

## Workflow across the two environments

1. **Develop on Mac** — write and test the activation extraction, analysis, and plotting code against `gpt2-small`.
2. **Push to GitHub.**
3. **Pull on the cluster** — switch to Llama-3.1-8B in the config, run the full extraction and composition sweep.
4. **Download results** — pull the numerical results (CSVs, NPZ files of activations if needed) back to your Mac.
5. **Analyze and plot on Mac** — iterate on figures and writing locally.

This two-environment pattern is the standard workflow in ML research labs. It maximizes your time on the expensive cluster (only for compute-heavy steps) and keeps iteration fast.

---

## Keeping the environments in sync

All three team members should use the same versions pinned in the requirements files. If you need to add a new package:

1. Add it to the appropriate requirements file with a pinned version.
2. Commit the change.
3. The other team members run `pip install -r requirements.txt` on their next pull.

Do not use `pip freeze > requirements.txt` — this pulls in every transitive dependency with exact versions and makes the file unreadable. Only add packages you explicitly import.


# Data
Contrastive datasets for `evil`, `sycophancy`, and `hallucination` are 
adapted from the Persona Vectors dataset released by Chen et al. (2025):

> Chen, Arditi, Sleight, Evans, Lindsey (2025). *Persona Vectors: Monitoring 
> and Controlling Character Traits in Language Models*. Anthropic.  
> https://github.com/safety-research/persona_vectors

Datasets for the remaining behaviors were generated using their prompt 
template with Claude 3.7 Sonnet.

