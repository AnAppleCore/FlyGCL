# FlyGCL: A Comprehensive Framework for Continual Learning Research

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

FlyGCL is a comprehensive research framework for **Class-Incremental Learning (CIL)** and **Generalized Continual Learning (GCL)** that implements state-of-the-art methods with a focus on prompt-based learning using Vision Transformers (ViT). The framework supports various continual learning scenarios, particularly the challenging "Si-Blurry" setting where data arrives in streaming fashion with both disjoint and overlapping samples.

## 🌟 Key Features

- **🔬 Comprehensive Method Coverage**: Implements 15+ state-of-the-art continual learning methods
- **🎯 Prompt-Based Learning**: Specialized support for ViT-based prompt learning methods (L2P, DualPrompt, CODA-P, MVP)
- **📊 Si-Blurry Setting**: Realistic continual learning scenarios with configurable data overlap
- **🚀 Online Learning**: True streaming data simulation with configurable evaluation periods
- **💾 Memory Management**: Efficient episodic memory handling for replay-based methods
- **📈 Robust Evaluation**: Multi-seed experiments with comprehensive metrics (A_auc, A_last)
- **🔧 Distributed Training**: Multi-GPU support with PyTorch DistributedDataParallel
- **📁 Modular Design**: Easy to extend with new methods and datasets

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Implemented Methods](#-implemented-methods)
- [Experiment Configuration](#-experiment-configuration)
- [Usage Examples](#-usage-examples)
- [Code Structure](#-code-structure)
- [Results and Evaluation](#-results-and-evaluation)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

## 🚀 Installation

### Prerequisites

- **CUDA**: 11.7 or compatible
- **Python**: 3.10+
- **GPU Memory**: ≥16GB recommended for ViT experiments

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/FlyGCL.git
cd FlyGCL
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate flygcl
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Manual Installation

If the conda environment fails, install dependencies manually:

```bash
conda create -n flygcl python=3.10
conda activate flygcl

# Core dependencies
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
pip install timm==0.9.12 pytorch-lightning==2.2.1
pip install numpy pandas matplotlib scikit-learn
pip install omegaconf datasets huggingface_hub

# Optional: for profiling and visualization
pip install torchinfo torchmetrics
```

## ⚡ Quick Start

### 1. Setup Datasets

Download and organize datasets in the following structure:
```
/data/datasets/
├── CIFAR/          # CIFAR-100
├── imagenet-r/     # ImageNet-R
├── CUB200_2011/    # CUB-200
└── ...
```

### 2. Run Your First Experiment

```bash
# Run L2P on CIFAR-100 with default settings
python main.py --mode L2P --dataset cifar100 --data_dir /data/datasets/CIFAR

# Run MVP with contrastive learning
python main.py --mode mvp --dataset cifar100 --use_mask --use_contrastiv --use_afs --use_mcr
```

### 3. Run Comprehensive Baselines

```bash
# Run all baseline methods on CIFAR-100
bash scripts/run_baselines.sh 0 1 cifar100

# Parameters: GPU_ID SEED DATASET
bash scripts/run_baselines.sh 1 42 imagenet-r
```

## 📊 Datasets

### Supported Datasets

| Dataset | Classes | Train Size | Test Size | Resolution | Path Configuration |
|---------|---------|------------|-----------|------------|-------------------|
| **CIFAR-100** | 100 | 50,000 | 10,000 | 32×32 | `/data/datasets/CIFAR` |
| **ImageNet-R** | 200 | ~120,000 | ~6,000 | 224×224 | `/data/datasets/imagenet-r` |
| **CUB-200** | 200 | ~6,000 | ~5,800 | 224×224 | `/data/datasets/CUB200_2011` |
| **TinyImageNet** | 200 | 100,000 | 10,000 | 64×64 | `/data/datasets/TinyImageNet` |
| **ImageNet-100** | 100 | ~130,000 | 5,000 | 224×224 | `/data/datasets/ImageNet100` |

### Dataset Download Instructions

<details>
<summary>Click to expand dataset download instructions</summary>

**CIFAR-100**: Automatically downloaded by torchvision

**ImageNet-R**:
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xf imagenet-r.tar -C /data/datasets/
```

**CUB-200**:
```bash
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz -C /data/datasets/
```

</details>

## 🧠 Implemented Methods

### Simple Baseline Methods

| Method | Description | Key Features |
|--------|-------------|--------------|
| **Sequential Fine-tuning** | Standard fine-tuning approach | Full model update, catastrophic forgetting baseline |
| **Linear Probe** | Freeze backbone, train classifier only | Parameter-efficient, limited adaptability |

### Prompt-Based Continual Learning Methods

| Method | Paper | Key Innovation | Implementation |
|--------|-------|----------------|----------------|
| **L2P** | [Learning to Prompt](https://arxiv.org/abs/2112.06905) | Learnable prompt selection | `methods/L2P.py` |
| **DualPrompt** | [DualPrompt](https://arxiv.org/abs/2204.04799) | E-prompts + G-prompts | `methods/dualprompt.py` |
| **CODA-Prompt** | [CODA-Prompt](https://arxiv.org/abs/2211.13860) | Domain-adaptive prompting | `methods/codaprompt.py` |
| **MVP** | [Multi-Visual Prompting](https://arxiv.org/abs/2306.12842) | Contrastive prompting + AFS/MCR | `methods/mvp.py` |

### Memory-Based Methods

| Method | Description | Memory Strategy |
|--------|-------------|-----------------|
| **Experience Replay (ER)** | Store and replay past samples | Random sampling |
| **DER++** | Dark Experience Replay++ | Logit distillation + replay |
| **Rainbow Memory** | Advanced memory management | Gradient-based selection |

### Regularization-Based Methods

| Method | Key Technique | Implementation |
|--------|---------------|----------------|
| **LwF** | Learning without Forgetting | Knowledge distillation |
| **EWC** | Elastic Weight Consolidation | Fisher information matrix |

## 📋 Experiment Configuration

### Si-Blurry Setting

The framework implements the **Si-Blurry** continual learning setting:

- **Disjoint Class Ratio (m)**: Percentage of task-specific classes (default: 50%)
- **Blurry Sample Ratio (n)**: Percentage of overlapping samples (default: 10%)
- **Tasks**: Number of incremental tasks (default: 5)
- **Online Learning**: 1 epoch per sample stream

### Key Configuration Parameters

```python
# Core settings
--dataset cifar100              # Dataset choice
--n_tasks 5                     # Number of tasks
--m 50                          # Disjoint class ratio (%)
--n 10                          # Blurry sample ratio (%)
--memory_size 500               # Episodic memory size

# Training settings
--batchsize 64                  # Batch size
--lr 0.005                      # Learning rate
--num_epochs 1                  # Epochs per task (online learning)
--online_iter 3                 # Updates per sample

# Model settings
--model_name vit_finetune       # ViT-B/16 backbone
--opt_name adam                 # Optimizer choice
--use_amp                       # Mixed precision training

# Method-specific
--use_mask                      # Enable logit masking (MVP)
--use_contrastiv               # Enable contrastive loss (MVP)
--use_afs                      # Adaptive Feature Scaling (MVP)
--use_mcr                      # Minor-Class Reinforcement (MVP)
```

## 💻 Usage Examples

### Basic Usage

```bash
# Train L2P on CIFAR-100
python main.py \
    --mode L2P \
    --dataset cifar100 \
    --n_tasks 5 \
    --m 50 --n 10 \
    --seeds 1 2 3 4 5 \
    --data_dir /data/datasets/CIFAR \
    --note "l2p_baseline_experiment"

# Train MVP with all enhancements
python main.py \
    --mode mvp \
    --dataset imagenet-r \
    --use_mask --use_contrastiv --use_afs --use_mcr \
    --alpha 0.5 --gamma 1.0 --margin 0.5 \
    --data_dir /data/datasets/imagenet-r
```

### Advanced Configuration

```bash
# Memory-efficient settings for large datasets
python main.py \
    --mode DualPrompt \
    --dataset imagenet-r \
    --batchsize 32 \
    --memory_size 1000 \
    --eval_period 500 \
    --use_amp \
    --n_worker 8

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --mode L2P \
    --dataset cub200 \
    --distributed \
    --world_size 2
```

### Batch Experiments

```bash
# Run all baselines on CIFAR-100
bash scripts/run_baselines.sh 0 1 cifar100

# Run specific method with multiple seeds
for seed in 1 2 3 4 5; do
    python main.py --mode L2P --dataset cifar100 --rnd_seed $seed
done
```

## 📁 Code Structure

```
FlyGCL/
├── 📁 configuration/           # Configuration management
│   ├── config.py              # Argument parsing and defaults
│   └── logging.conf           # Logging configuration
├── 📁 datasets/               # Dataset implementations
│   ├── __init__.py           # Dataset registry
│   ├── CIFAR100.py           # CIFAR-100 dataset
│   ├── Imagenet_R.py         # ImageNet-R dataset
│   └── ...                   # Other dataset implementations
├── 📁 methods/                # Continual learning methods
│   ├── __init__.py           # Method registry
│   ├── _trainer.py           # Base trainer class
│   ├── L2P.py               # Learning to Prompt
│   ├── dualprompt.py        # DualPrompt implementation
│   ├── mvp.py               # Multi-Visual Prompting
│   ├── codaprompt.py        # CODA-Prompt
│   └── ...                  # Other method implementations
├── 📁 models/                 # Model architectures
│   ├── vit.py               # Vision Transformer implementations
│   ├── L2P.py               # L2P-specific model components
│   ├── dualprompt.py        # DualPrompt model components
│   └── ...                  # Other model files
├── 📁 utils/                  # Utility functions
│   ├── memory.py            # Memory management utilities
│   ├── data_loader.py       # Data loading utilities
│   ├── train_utils.py       # Training helper functions
│   └── ...                  # Other utilities
├── 📁 scripts/               # Experiment scripts
│   ├── run_baselines.sh     # Main baseline script
│   ├── run_baselines_l2p_50_10.sh
│   └── ...                  # Other experiment scripts
├── 📁 results/               # Experiment results and logs
├── 📁 pretrained_prompt/     # Pre-trained prompt weights
├── main.py                   # Main entry point
├── environment.yml           # Conda environment specification
└── README.md                # This documentation
```

### Key Components

**🔧 Core Framework (`methods/_trainer.py`)**
- Base trainer class with distributed training support
- Online learning simulation with configurable evaluation
- Memory management and data streaming
- Mixed precision training and optimization

**🧠 Method Implementations (`methods/`)**
- Each method inherits from `_Trainer`
- Implements method-specific `online_step()` and `online_train()`
- Modular design for easy extension

**🏗️ Model Architecture (`models/`)**
- ViT backbone with prompt integration
- Method-specific model components
- Pre-trained weight loading and initialization

**💾 Memory Management (`utils/memory.py`)**
- Efficient episodic memory storage
- Class-balanced sampling strategies
- Loss-based sample selection

## 📈 Results and Evaluation

### Metrics

- **A_auc**: Area under the accuracy curve (higher is better)
- **A_last**: Final task accuracy (higher is better)
- **Per-task accuracy**: Individual task performance tracking
- **Memory efficiency**: Memory usage vs. performance trade-offs

### Result Analysis

Results are automatically saved in the `results/` directory with the following structure:

```
results/
├── logs/
│   └── {dataset}/
│       └── {method}_{dataset}_{note}/
│           ├── seed_{seed}_log.txt    # Training logs
│           └── ...
└── {method}_{dataset}_{timestamp}.json   # Experiment results
```

### Viewing Results

```bash
# Monitor running experiment
tail -f results/logs/cifar100/L2P_cifar100_baseline/seed_1_log.txt

# View completed results
ls -la results/ | grep baseline

# Parse results with custom scripts
python utils/parse_results.py --result_dir results/
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Adding a New Method

1. **Create method file**: `methods/your_method.py`
2. **Inherit from `_Trainer`**: Implement required methods
3. **Register method**: Add to `methods/__init__.py`
4. **Add model components**: If needed in `models/`
5. **Test implementation**: Run basic experiments

```python
# Template for new method
from methods._trainer import _Trainer

class YourMethod(_Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Method-specific initialization
    
    def online_step(self, images, labels, idx):
        # Implement online learning step
        pass
    
    def online_train(self, data):
        # Implement training logic
        pass
```

### Adding a New Dataset

1. **Create dataset file**: `datasets/your_dataset.py`
2. **Follow existing patterns**: Inherit from appropriate base class
3. **Register dataset**: Add to `datasets/__init__.py`
4. **Update statistics**: Add to `utils/data_loader.py`

### Code Style

- Follow PEP 8 guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write unit tests for new functionality

## 🐛 Troubleshooting

### Common Issues and Solutions

**🚨 CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
python main.py --batchsize 32  # or even 16

# Solution 2: Use gradient accumulation
python main.py --online_iter 1 --batchsize 64

# Solution 3: Enable mixed precision
python main.py --use_amp
```

**📂 Dataset Path Issues**
```bash
# Check dataset structure
ls -la /data/datasets/CIFAR

# Update paths in scripts
export DATA_DIR="/your/custom/path"
python main.py --data_dir $DATA_DIR
```

**🔧 Environment Issues**
```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

**📊 Performance Issues**
```bash
# Enable profiling
python main.py --profile

# Reduce workers if I/O bound
python main.py --n_worker 4

# Use smaller input resolution
python main.py --transforms none  # Remove AutoAugment
```

### Getting Help

1. **Check logs**: Review detailed logs in `results/logs/`
2. **Issues**: Open GitHub issues with logs and configuration
3. **Discussions**: Use GitHub Discussions for questions
4. **Documentation**: Refer to method-specific papers for details

## 📚 Citation

If you use FlyGCL in your research, please cite:

```bibtex
@misc{flygcl2024,
  title={FlyGCL: A Comprehensive Framework for Continual Learning Research},
  author={Your Name and Collaborators},
  year={2024},
  howpublished={\url{https://github.com/your-username/FlyGCL}},
}
```

### Related Papers

Please also consider citing the original papers for the methods you use:

- **L2P**: Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022
- **DualPrompt**: Wang et al., "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning", ECCV 2022
- **CODA-Prompt**: Smith et al., "CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning", CVPR 2023
- **MVP**: Jia et al., "Multi-Visual Prompting for Rehearsal-free Continual Learning", arXiv 2023

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Prompt Learning Community**: For foundational work on prompt-based learning
- **Continual Learning Researchers**: For establishing evaluation protocols
- **PyTorch Team**: For the excellent deep learning framework
- **Timm Library**: For pre-trained Vision Transformer models

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

**🔔 Watch for updates and new methods! 🔔**

</div>