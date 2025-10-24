# TATGSL

The implementation of TATGSL in the paper:

"Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification".

## Usage

### Preprocess and Split Text

Text data was preprocessed and divided in advance, and stored in the path `./data/split`

___

### Train Command

You can train the model with ___Sparse Adjacency Matrix___ by default.

At the first run, the SBERT model 'all-MILM-L6-v2' will be downloaded from [huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and saved in the path `./plm`

```bash
python -m src.main --dataset ohsumed --gnn_type GSAGE --top_k 15
```
