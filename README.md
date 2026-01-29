# TATGSL: Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyTorch 2.1.0+](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/) [![Transformers 4.49.0+](https://img.shields.io/badge/Transformers-4.49.0+-orange.svg)](https://huggingface.co/transformers) [![DGL 1.1.2+](https://img.shields.io/badge/DGL-1.1.2+-green.svg)](https://www.dgl.ai/)

This repository contains the official implementation of the paper **"Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification"**.

## 📋 Overview

TATGSL is a novel framework for **semi-supervised text classification** that jointly optimizes node representations and graph topology through task-aware structure learning. Unlike static graph construction methods that suffer from semantic-task mismatch, TATGSL introduces a unified optimization objective that dynamically refines the graph to align with classification tasks while maintaining computational efficiency.

**Key Contributions:**

- ✅ **Joint graph-node optimization**: Alternating optimization of node representations and graph topology
- ✅ **Unified objective integration**: Coherent combination of classification loss, task-aware metric learning, and homophily-consistency loss
- ✅ **Confidence-aware co-training**: Dynamic graph refinement with error propagation prevention
- ✅ **Memory-efficient design**: Sparse adjacency matrix implementation with modest memory usage
- ✅ **Strong robustness**: State-of-the-art performance across multiple text classification benchmarks

## 🚀 Performance Highlights

TATGSL achieves **state-of-the-art performance** on five text classification benchmarks while maintaining **modest memory usage** and demonstrating **strong robustness** against label noise and sparse supervision.

## 📁 Project Structure

```
TATGSL/
├── README.md                           # This file
├── data/                               # Dataset directory
│   ├── raw/                            # Raw un-processed data
│   │   ├── 20ng.labels.txt             # 20 Newsgroups dataset labels
│   │   ├── 20ng.texts.txt              # 20 Newsgroups dataset raw texts
│   │   ├── AGNews.labels.txt           # AG News dataset labels
│   │   ├── AGNews.texts.txt            # AG News dataset raw texts
│   │   ├── R8.labels.txt               # Reuters-8 dataset labels
│   │   ├── R8.texts.txt                # Reuters-8 dataset raw texts
│   │   ├── ohsumed.labels.txt          # Ohsumed medical dataset labels
│   │   ├── ohsumed.texts.txt           # Ohsumed medical dataset raw texts
│   │   ├── snippets.labels.txt         # Search Engine snippets dataset labels
│   │   └── snippets.texts.txt          # Search Engine snippets dataset raw texts
│   └── split/                          # Preprocessed dataset splits
│       ├── 20ng-t10v10.json            # 20 Newsgroups dataset (10 samples per class for training and validation)
│       ├── AGNews-t10v10.json          # AG News dataset
│       ├── R8-t10v10.json              # Reuters-8 dataset
│       ├── ohsumed-t10v10.json         # Ohsumed medical dataset
│       └── snippets-t10v10.json        # Search Engine snippets dataset
├── plm/                                # Pre-trained language models (auto-downloaded)
│   └── all-MiniLM-L6-v2/               # Sentence-BERT model
└── src/                                # TATGSL core implementation
    ├── __init__.py
    ├── main.py                         # Main entry point
    ├── graph_classifier.py             # GNN-based classifier
    ├── graph_learner.py                # Graph structure learner
    ├── graph_sampler.py                # Graph sampling utilities
    ├── modules.py                      # Neural network modules
    ├── sbert_trainer.py                # Sentence-BERT training wrapper
    ├── text_filter.py                  # Text preprocessing utilities
    └── utils.py                        # Utility functions
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TATGSL.git
cd TATGSL

# Create and activate conda environment (recommended)
conda create -n tatgsl python=3.11.0
conda activate tatgsl

# Install PyTorch (adjust according to your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install DGL with CUDA support
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install other dependencies
pip install -r requirements.txt
```

### Data Preparation

The datasets are already preprocessed and split in the `data/split/` directory. Each JSON file contains:

- `train_texts`, `train_labels`: Training data with limited labels
- `val_texts`, `val_labels`: Validation data
- `test_texts`, `test_labels`: Test data
- `unlabeled_texts`: Unlabeled data for semi-supervised learning

#### Dataset Preprocess

The original raw datasets are located in `./data/raw/`. Each dataset consists of two files:
- `{dataset_name}.texts.txt`: contains the raw text samples
- `{dataset_name}.labels.txt`: contains the corresponding labels

If you wish to preprocess the raw texts (e.g., cleaning, tokenization, normalization), you can run:

```bash
python src/text_filter.py
```

This script will process all datasets and save the cleaned versions to `./data/temp/`. Note that the provided pre-split datasets in `./data/split/` are already cleaned and ready to use.

#### Using Your Own Dataset

If you want to use a custom dataset, please follow these steps:

1. Place your raw text and label files in `./data/raw/` with the same naming format (e.g., `mydataset.texts.txt`, `mydataset.labels.txt`).
2. Run `main.py` with the `--do_split` flag for the first time:

```bash
python src/main.py --dataset mydataset --do_split
```

This will:
- Automatically preprocess and clean your raw texts
- Split the dataset into train/validation/test sets (with default ratios)
- Save the split data to `./data/split/` in JSON format

You only need to run with `--do_split` once for each new dataset. After that, the split files will be available for future runs without re-processing.

### Pre-trained Models

The Sentence-BERT model `all-MiniLM-L6-v2` will be automatically downloaded from Hugging Face on first run and saved to the `plm/` directory.

## ⚡ Running Experiments

### Basic Usage

```bash
# Run TATGSL with default parameters
python src/main.py --dataset ohsumed --gnn_type GSAGE --top_k 15
```

This command will:
1. Download the Sentence-BERT model if not already present
2. Load the specified dataset
3. Train TATGSL with GraphSAGE as the GNN backbone
4. Construct a sparse graph with top-15 neighbors per node
5. Evaluate on the test set and report results

### Advanced Training

For more control over the training process:

```bash
# Run with custom configurations
python src/main.py \
  --dataset AGNews \
  --gnn_type GAT \
  --top_k 20 \
  --epochs 50 \
  --train_per_label 15 \
  --eval_per_label 15 \
  --test_per_label 30 \
  --gca_learning_rate 0.005 \
  --gsl_learning_rate 0.0005 \
  --conf_thresh 0.8 \
  --cuda_id 0
```

## ⚙️ Configuration

### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset name | `ohsumed` | `ohsumed`, `20ng`, `AGNews`, `R8`, `snippets` |
| `--gnn_type` | GNN backbone | `GSAGE` | `GCN`, `GAT`, `GIN`, `GSAGE`, `APPNP`|
| `--readout` | Readout function | `mean` | `mean`, `max`, `centroid` |
| `--train_per_label` | Training samples per label | `10` | Integer > 0 |
| `--eval_per_label` | Validation samples per label | `10` | Integer > 0 |
| `--test_per_label` | Testing samples per label | `20` | Integer > 0 |
| `--epochs` | Main training epochs | `10` | Integer > 0 |
| `--finetune_epochs` | SBERT fine-tuning epochs | `5` | Integer > 0 |
| `--top_k` | Neighbors for graph sparsification | `15` | Integer > 0 |
| `--num_hop` | Label propagation hops | `2` | Integer > 0 |
| `--conf_thresh` | Confidence threshold τp | `0.7` | Float (0.0-1.0) |

### Model Architecture Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--in_channels` | Node feature dimension | `384` |
| `--gca_hidden_channels` | GCA hidden dimension | `128` |
| `--gsl_hidden_channels` | GSL hidden dimension | `256` |
| `--mlp_layer` | MLP layers in GSL | `2` |
| `--gca_dropout` | GCA dropout rate | `0.5` |
| `--gsl_dropout` | GSL dropout rate | `0.2` |

### Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ftn_learning_rate` | SBERT fine-tuning LR | `1e-5` |
| `--gca_learning_rate` | GCA learning rate | `0.01` |
| `--gsl_learning_rate` | GSL learning rate | `0.001` |

### Experimental Control Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--cpu` | Use CPU instead of GPU | `False` |
| `--use_dense` | Learn dense adjacency matrix | `False` |
| `--do_split` | Re-divide train/eval/test sets | `False` |
| `--no_finetune` | Skip SBERT fine-tuning | `False` |
| `--end2end` | End-to-end PLM training | `False` |
| `--no_gsl` | Skip graph structure learning | `False` |
| `--no_conf` | Disable confidence threshold | `False` |

### Path and System Parameters

| Parameter | Description | Default                                  |
|-----------|-------------|------------------------------------------|
| `--bert_name` | Sentence-BERT model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `--plm_path` | Pre-trained model path | `./plm`                                  |
| `--ftm_path` | Fine-tuned model path | `./ftm`                                  |
| `--data_path` | Dataset path | `./data`                                 |
| `--save_path` | Output save path | `./out`                                  |
| `--cuda_id` | GPU device ID | `0`                                      |
| `--seed` | Random seed | `123`                                    |

## 📊 Datasets

| Dataset | Classes | Domain | Train/Val/Test Samples per Class | Description |
|---------|---------|--------|----------------------------------|-------------|
| ohsumed | 23 | Medical | 10/10/20                         | Medical abstract classification |
| 20ng | 20 | News | 10/10/20                         | 20 Newsgroups text classification |
| AGNews | 4 | News | 10/10/20                         | AG News topic classification |
| R8 | 8 | News | 10/10/20                         | Reuters-21578 R8 subset |
| snippets | 8 | Web Snippets | 10/10/20                         | Web search result classification |

## 🔧 Advanced Usage

### Extending with New GNN Models

1. Add the new GNN model implementation to `src/modules.py`
2. Update the model initialization in `src/graph_classifier.py`
3. Add the model name to the choices in `src/main.py` argument parser

### Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# Run experiments for all datasets with optimal configurations
for dataset in ohsumed 20ng AGNews R8 snippets; do
  python -m src.main \
    --dataset $dataset \
    --gnn_type GSAGE \
    --top_k 15 \
    --epochs 10 \
    --finetune_epochs 5 \
    --gca_learning_rate 0.01 \
    --gsl_learning_rate 0.001 \
    --conf_thresh 0.7 \
    --seed 123
done
```

## 📈 Output Format

Results are printed to console and saved to `out/{dataset}_{timestamp}.log` with the following information:

- **Configuration**: All training arguments
- **Training Progress**: Per-epoch metrics including:
  - Graph connectivity flag
  - Graph homophily score
  - GCA training loss
  - GCA validation accuracy
  - GSL training loss
- **Final Results**: Classification report with precision, recall, F1-score
- **Macro/Micro Averages**: Overall performance metrics
- **Visualization**: Training curves saved as images

## 📝 Citation

If you use TATGSL in your research, please cite our paper:

```bibtex
@inproceedings{wang2026tatgsl,
  title={Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification},
  author={Shiyu Wang and Gang Zhou and Jicang Lu and Ningbo Huang and Qiankun Pi},
  booktitle={Proceedings of the 31st International Conference on Database Systems for Advanced Applications (DASFAA 2026)},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This implementation uses [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers/), and [DGL](https://www.dgl.ai/)
- Thanks to the authors of the benchmark datasets used in this work
- The Sentence-BERT implementation is based on [sentence-transformers](https://www.sbert.net/)

---

For questions or issues, please open an issue on GitHub.