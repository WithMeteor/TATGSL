# TATGSL

The implementation of TATGSL in the paper:

"Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification".

## Code Structure

```
.
├── README.md
├── data
│       └── split
│           ├── 20ng-t10v10.json
│           ├── AGNews-t10v10.json
│           ├── R8-t10v10.json
│           ├── ohsumed-t10v10.json
│           └── snippets-t10v10.json
├── requirements.txt
└── src
    ├── __init__.py
    ├── graph_classifier.py
    ├── graph_learner.py
    ├── graph_sampler.py
    ├── main.py
    ├── modules.py
    ├── sbert_trainer.py
    ├── text_filter.py
    └── utils.py
```

___

## Getting Started

### Stage-1: Preprocess and Split Text

Text data was preprocessed and divided in advance, and stored in the path `./data/split`

### Stage-2: Running Command

You can train the model with ___Sparse Adjacency Matrix___ by default.

At the first run, the SBERT model 'all-MILM-L6-v2' will be downloaded from [huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and saved in the path `./plm`

```bash
python -m src.main --dataset ohsumed --gnn_type GSAGE --top_k 15
```

___

## Requirements

- python 3.11+
- torch 2.1.0+
- transformers 4.49.0+
- sentence-transformers 3.4.1+
- dgl 1.1.2+
- scikit-learn 1.6.1+
- tqdm 4.67.1+
- numpy 1.25.2+
- datasets 3.5.0+
- networkx
- matplotlib