# FlyGCL: A Comprehensive Framework for Continual Learning Research

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

FlyGCL is a comprehensive research framework for **Class-Incremental Learning (CIL)** and **Generalized Continual Learning (GCL)** that implements state-of-the-art methods with a focus on prompt-based learning using Vision Transformers (ViT). The framework supports various continual learning scenarios, particularly the challenging "Si-Blurry" setting where data arrives in streaming fashion with both disjoint and overlapping samples.

## 🌟 Key Features

- **🔬 Comprehensive Method Coverage**: Implements 6 state-of-the-art continual learning methods with focus on prompt-based approaches
- **🎯 Prompt-Based Learning**: Specialized support for ViT-based prompt learning methods (L2P, DualPrompt, CODA-P, MVP)
- **📊 Si-Blurry Setting**: Realistic continual learning scenarios with configurable data overlap and class distribution
- **🚀 Online Learning**: True streaming data simulation with configurable evaluation periods
- **📈 Robust Evaluation**: Multi-seed experiments with comprehensive metrics and distributed training support
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

2. **Create conda environment and install dependencies:**
```bash
conda create -n flygcl python=3.10
conda activate flygcl

# Core dependencies
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
pip install timm pytorch-lightning
pip install numpy pandas matplotlib scikit-learn
pip install omegaconf datasets huggingface_hub
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## ⚡ Quick Start

### 1. Setup Datasets

Download and organize datasets in the following structure:
```
/data/datasets/
├── CIFAR/          # CIFAR-100
├── imagenet-r/     # ImageNet-R
├── CUB200_2011/    # CUB-200
├── TinyImageNet/   # TinyImageNet
└── ...
```

### 2. Run Your First Experiment

```bash
# Run L2P on CIFAR-100 with default settings
python main.py --method l2p --dataset cifar100 --data_dir /data/datasets/CIFAR

# Run MVP with enhanced features
python main.py --method mvp --dataset cifar100 --no_batchmask
```

### 3. Run Comprehensive Baselines

```bash
# Run all baseline methods on CIFAR-100
bash scripts/run_baselines.sh 0 1 cifar100

# Parameters: GPU_ID SEED DATASET [EXTRA_NOTE]
bash scripts/run_baselines.sh 1 "1 2 3" imagenet-r baseline_test
```

## 📊 Datasets

### Supported Datasets

| Dataset | Classes | Train Size | Test Size | Resolution | Configuration Key |
|---------|---------|------------|-----------|------------|-------------------|
| **CIFAR-10** | 10 | 50,000 | 10,000 | 32×32 | `cifar10` |
| **CIFAR-100** | 100 | 50,000 | 10,000 | 32×32 | `cifar100` |
| **TinyImageNet** | 200 | 100,000 | 10,000 | 64×64 | `tinyimagenet` |
| **ImageNet-R** | 200 | ~120,000 | ~6,000 | 224×224 | `imagenet-r` |
| **CUB-200** | 200 | ~6,000 | ~5,800 | 224×224 | `cub200` |
| **CUB-175** | 175 | ~5,000 | ~5,000 | 224×224 | `cub175` |
| **CARS-196** | 196 | ~8,000 | ~8,000 | 224×224 | `cars196` |
| **ImageNet-100** | 100 | ~130,000 | 5,000 | 224×224 | `imagenet100` |
| **ImageNet-900** | 900 | ~1.2M | ~45,000 | 224×224 | `imagenet900` |
| **Places365** | 365 | ~1.8M | ~36,500 | 224×224 | `places365` |
| **GTSRB** | 43 | ~39,000 | ~12,600 | 32×32 | `gtsrb` |
| **WikiArt** | 195 | ~80,000 | ~10,000 | 224×224 | `wikiart` |

### Dataset Download Instructions

<details>
<summary>Click to expand dataset download instructions</summary>

**CIFAR-10/100**: Automatically downloaded by torchvision

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

**TinyImageNet**:
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d /data/datasets/
mv /data/datasets/tiny-imagenet-200 /data/datasets/TinyImageNet
```

</details>

## 🧠 Implemented Methods

### Prompt-Based Continual Learning Methods

| Method | Paper | Key Innovation | Implementation |
|--------|-------|----------------|----------------|
| **L2P** | [Learning to Prompt](https://arxiv.org/abs/2112.06905) | Learnable prompt selection with pool | `methods/l2p.py` |
| **DualPrompt** | [DualPrompt](https://arxiv.org/abs/2204.04799) | E-prompts + G-prompts | `methods/dualprompt.py` |
| **CODA-Prompt** | [CODA-Prompt](https://arxiv.org/abs/2211.13860) | Domain-adaptive prompting | `methods/codaprompt.py` |
| **MVP** | [Multi-Visual Prompting](https://arxiv.org/abs/2306.12842) | Contrastive prompting + AFS/MCR | `methods/mvp.py` |

### Additional Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **SLCA** | Sequential Learning with Class Adaptation | `methods/slca.py` |
| **FlyPrompt** | Framework-specific prompt method | `methods/flyprompt.py` |

## 📋 Experiment Configuration

### Si-Blurry Setting

The framework implements the **Si-Blurry** continual learning setting with precise control:

- **Disjoint Class Ratio (n)**: Percentage of task-specific classes (default: 50%)
- **Blurry Sample Ratio (m)**: Percentage of overlapping samples (default: 10%)
- **Tasks**: Number of incremental tasks (default: 5)
- **Online Learning**: Configurable epochs and iterations per sample stream

> **⚠️ Important**: Parameter `n` controls disjoint classes, `m` controls blurry samples. This differs from some documentation conventions.

### Key Configuration Parameters

```python
# Core Si-Blurry settings
--dataset cifar100              # Dataset choice
--n_tasks 5                     # Number of tasks
--n 50                          # Disjoint class ratio (%)
--m 10                          # Blurry sample ratio (%)
--rnd_NM                        # Randomly vary N/M across tasks

# Training settings
--batchsize 64                  # Batch size
--lr 0.005                      # Learning rate
--num_epochs 1                  # Epochs per task (online learning)
--online_iter 3                 # Updates per sample
--eval_period 1000              # Evaluation frequency

# Model settings
--backbone vit_base_patch16_224 # ViT-B/16 backbone
--opt_name adam                 # Optimizer choice
--sched_name default            # Scheduler choice
--use_amp                       # Mixed precision training

# Method-specific (MVP)
--no_batchmask                  # Disable batch-wise masking
```

### Advanced Configuration Options

```python
# Data processing
--transforms autoaug            # Data augmentation strategy
--n_worker 8                    # Data loading workers

# Distributed training
--world_size 2                  # Number of distributed processes
--distributed                   # Enable distributed training

# Evaluation
--topk 1                        # Top-k accuracy evaluation
--profile                       # Enable profiling mode
```

## 💻 Usage Examples

### Basic Usage

```bash
# Train L2P on CIFAR-100
python main.py \
    --method l2p \
    --dataset cifar100 \
    --n_tasks 5 \
    --n 50 --m 10 \
    --seeds 1 2 3 4 5 \
    --data_dir /data/datasets/CIFAR \
    --note "l2p_baseline_experiment"

# Train MVP with enhanced features
python main.py \
    --method mvp \
    --dataset imagenet-r \
    --no_batchmask \
    --data_dir /data/datasets/imagenet-r \
    --note "mvp_enhanced"
```

### Advanced Configuration

```bash
# Memory-efficient settings for large datasets
python main.py \
    --method dualprompt \
    --dataset imagenet-r \
    --batchsize 32 \
    --eval_period 500 \
    --use_amp \
    --n_worker 8

# Varying N/M ratio across tasks
python main.py \
    --method codaprompt \
    --dataset cub200 \
    --rnd_NM \
    --n_tasks 10 \
    --note "varying_nm_experiment"
```

### Batch Experiments

```bash
# Run all baselines on CIFAR-100 with specific seeds
bash scripts/run_baselines.sh 0 "1 2 3 4 5" cifar100 comprehensive_test

# Run method-specific experiments
bash scripts/run_baselines_mvp.sh 1 "1 2 3" imagenet-r mvp_test
bash scripts/run_baselines_l2p.sh 2 "1 2 3" cub200 l2p_test
```

## 📁 Code Structure

```
FlyGCL/
├── 📁 configuration/           # Configuration management
│   ├── config.py              # Argument parsing and defaults
│   └── logging.conf           # Logging configuration
├── 📁 datasets/               # Dataset implementations
│   ├── __init__.py           # Dataset registry
│   ├── CIFAR100.py           # (Handled by torchvision)
│   ├── Imagenet_R.py         # ImageNet-R dataset
│   ├── CUB200.py             # CUB-200 dataset
│   ├── TinyImageNet.py       # TinyImageNet dataset
│   ├── CARS196.py            # Stanford Cars dataset
│   ├── WIKIART.py            # WikiArt dataset
│   ├── GTSRB.py              # German Traffic Sign Recognition
│   └── OnlineIterDataset.py  # Online iteration wrapper
├── 📁 methods/                # Continual learning methods
│   ├── __init__.py           # Method registry
│   ├── _trainer.py           # Base trainer class
│   ├── l2p.py               # Learning to Prompt
│   ├── dualprompt.py        # DualPrompt implementation
│   ├── mvp.py               # Multi-Visual Prompting
│   ├── codaprompt.py        # CODA-Prompt
│   ├── slca.py              # Sequential Learning with Class Adaptation
│   └── flyprompt.py         # Framework-specific method
├── 📁 models/                 # Model architectures
│   ├── __init__.py          # Model registry
│   ├── vit.py               # Vision Transformer implementations
│   ├── l2p.py               # L2P-specific model components
│   ├── dualprompt.py        # DualPrompt model components
│   ├── mvp.py               # MVP model components
│   ├── codaprompt.py        # CODA-Prompt model components
│   └── layers.py            # Custom layer implementations
├── 📁 utils/                  # Utility functions
│   ├── onlinesampler.py     # Si-Blurry sampling logic
│   ├── data_loader.py       # Data loading utilities
│   ├── train_utils.py       # Training helper functions
│   ├── memory.py            # Memory management utilities
│   ├── buffer.py            # Buffer management
│   └── augment.py           # Data augmentation utilities
├── 📁 scripts/               # Experiment scripts
│   ├── run_baselines.sh     # Main baseline script
│   ├── run_baselines_l2p.sh # L2P-specific experiments
│   ├── run_baselines_mvp.sh # MVP-specific experiments
│   ├── run_baselines_dualprompt.sh # DualPrompt experiments
│   ├── run_baselines_codaprompt.sh # CODA-Prompt experiments
│   └── common_baselines.sh  # Common experiment utilities
├── 📁 results/               # Experiment results and logs
├── 📁 pretrained_prompt/     # Pre-trained prompt weights
├── main.py                   # Main entry point
├── run.sh                    # Custom experiment runner
└── README.md                # This documentation
```

### Key Components

**🔧 Core Framework (`methods/_trainer.py`)**
- Base trainer class with distributed training support
- Online learning simulation with Si-Blurry sampling
- Memory management and streaming data handling
- Mixed precision training and optimization
- Comprehensive evaluation with configurable periods

**🧠 Method Implementations (`methods/`)**
- Each method inherits from `_Trainer`
- Implements method-specific `online_step()` and `online_train()`
- Modular design for easy extension and comparison

**🏗️ Model Architecture (`models/`)**
- ViT backbone with frozen weights (except classifier)
- Method-specific prompt integration strategies
- Pre-trained weight loading and initialization

**📊 Si-Blurry Sampling (`utils/onlinesampler.py`)**
- Sophisticated class and sample distribution control
- Support for varying N/M ratios across tasks
- Distributed training compatible sampling

## 📈 Results and Evaluation

### Metrics

- **A_auc**: Area under the accuracy curve across all tasks
- **A_last**: Final task accuracy after all training
- **Per-task accuracy**: Individual task performance tracking
- **Memory efficiency**: Memory usage vs. performance analysis

### Result Analysis

Results are automatically saved with structured logging:

```
results/
├── logs/
│   └── {dataset}/
│       └── {method}_{dataset}_{note}/
│           ├── seed_{seed}_log.txt    # Detailed training logs
│           └── ...
└── {method}_{dataset}_{timestamp}.json   # Structured results
```

### Viewing Results

```bash
# Monitor running experiment
tail -f results/logs/cifar100/l2p_cifar100_baseline/seed_1_log.txt

# View completed results
ls -la results/ | grep baseline

# Parse results for analysis
python utils/generate_csv_from_logs.ipynb  # Jupyter notebook for analysis
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
2. **Follow existing patterns**: Check existing implementations
3. **Register dataset**: Add to `datasets/__init__.py`
4. **Update statistics**: Add to `utils/data_loader.py`

## 🐛 Troubleshooting

### Common Issues and Solutions

**🚨 CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
python main.py --batchsize 32  # or even 16

# Solution 2: Reduce online iterations
python main.py --online_iter 1

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

# Check timm installation
python -c "import timm; print(timm.__version__)"
```

**📊 Si-Blurry Configuration**
```bash
# Ensure proper N/M values
python main.py --n 50 --m 10  # 50% disjoint, 10% blurry overlap

# Enable random N/M variation
python main.py --rnd_NM
```

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