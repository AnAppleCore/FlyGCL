# Baseline Experiments Guide

This guide explains how to run standardized baseline experiments according to the paper's configuration.

## 📋 Experiment Configuration

### Si-Blurry Setting
- **Disjoint Class Ratio (m)**: 50%
- **Blurry Sample Ratio (n)**: 10%
- **Tasks**: 5
- **Evaluation Metrics**: AaUC (↑), ALast (↑)

### Datasets Supported
- **CIFAR-100**: `/data/datasets/CIFAR`
- **ImageNet-R**: `/data/datasets/imagenet-r`
- **CUB-200**: `/data/datasets/CUB200_2011`

## 🚀 Running Experiments

### 1. Baseline Methods (Table 2 Configuration)

```bash
# Run all baseline methods on CIFAR-100 with GPU 0 and seed 1
bash scripts/run_baselines.sh 0 1 cifar100

# Run on ImageNet-R with GPU 1 and seed 42
bash scripts/run_baselines.sh 1 42 imagenet-r

# Run on CUB-200 with GPU 0 and seed 123
bash scripts/run_baselines.sh 0 123 cub200
```

**Parameters:**
- `$1`: GPU ID (default: 0)
- `$2`: Random seed (default: 1)
- `$3`: Dataset name (default: cifar100)

### 2. MePo Method (Table 3 Configuration)

```bash
# Run MePo experiments
bash scripts/run_mepo.sh 0 1 cifar100
```

**Note**: MePo meta-learning phases are preserved for compatibility but not executed as per your requirements.

## 📊 Implemented Baseline Methods

### Simple Methods
| Method | Backbone | Optimizer | Training Config |
|--------|----------|-----------|----------------|
| Seq FT | ViT-B/16 | SGD | Full model fine-tuning |
| Linear Probe | ViT-B/16 | Adam | Backbone frozen, output layer only |

### PTMs-based CL Methods
| Method | Backbone | Prompt Type | Optimizer | Batch Size |
|--------|----------|-------------|-----------|------------|
| CODA-P | ViT-B/16 | Prompt Tuning | Adam | 64 |
| L2P | ViT-B/16 | Prompt Tuning | Adam | 64 |
| DualPrompt | ViT-B/16 | Prefix Tuning | Adam | 64 |

### PTMs-based GCL Methods
| Method | Backbone | Features | Optimizer | Batch Size |
|--------|----------|----------|-----------|------------|
| MVP | ViT-B/16 | Contrastive Loss + Logit Masking | Adam | 64 |

## ⚙️ Key Configuration Parameters

### Common Settings
- **Batch Size**: 64
- **Learning Rate**: 0.005 (output layer)
- **Training Epochs**: 1 (online learning)
- **Backbone**: ViT-B/16 (frozen for PTMs methods)

### Feature Alignment (MePo)
- **Sup-21K**: α = 0.5
- **Sup-21/1K**: α = 0.7
- **iBOT-21K**: α = 0.7

## 📁 Results Structure

Results are saved in `results/` directory with naming pattern:
```
{METHOD}_{DATASET}_baseline_standard_{TIMESTAMP}.log
```

Example:
```
results/DualPrompt_cifar100_baseline_standard_20241201_143022.log
```

## 🔧 Manual GPU Selection

To specify a different GPU:
```bash
# Use GPU 2
export CUDA_VISIBLE_DEVICES=2
bash scripts/run_baselines.sh 2 1 cifar100
```

## 📝 Notes

1. **Prefix Tuning**: DualPrompt uses prefix tuning (length=5, layers 1-5) as specified in the paper
2. **L2P**: Uses prompt tuning (not prefix), which matches the "Deep L2P" configuration
3. **Replay-free**: All PTMs methods run without replay memory
4. **Online Learning**: All methods use 1 epoch training as per online learning setting
5. **MISA**: Implementation not found in current codebase - may need separate implementation

## 🐛 Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size if encountering OOM errors
2. **Data Path**: Ensure datasets are properly located in `/data/datasets/`
3. **Missing Methods**: Some methods (like MISA) may need additional implementation

### Checking Results
```bash
# View latest results
ls -la results/ | grep baseline_standard

# Monitor running experiment
tail -f results/DualPrompt_cifar100_baseline_standard_*.log
```

## 📈 Expected Output

Each experiment will generate:
- **AaUC**: Average accuracy under curve
- **ALast**: Average final accuracy
- **Per-task performance**: Accuracy for each of the 5 tasks
- **Training logs**: Detailed training progress 