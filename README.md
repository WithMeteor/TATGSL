# TATGSL

The implementation of TATGSL in the paper:

"Task-Aware Text Graph Structure Learning for Semi-Supervised Text Classification".

## Usage

### Install environment

```bash
pip install -e .
```
The `homo-text-graph` package will be installed.
You can uninstall it with command `pip uninstall homo-text-graph`.

If you don't want to install the environment, you can run the code by replacing
`python src/main.py --argument` with `python -m src.main --argument`.

### Preprocess Text
The raw text file and label file are prepared at `./data/raw`

```bash
python src/text_filter.py
```
The preprocessed text file and label file will be saved at `./data/temp/`
___
### Split Data
When running the training code for the first time, you need to specify the parameter "--do-split" to split data. 
The divided data will be saved at `./data/split/`

___
### Train Command

You can train the model with ___Sparse Adjacency Matrix___ by default.
```bash
python src/main.py --dataset 20ng --gnn_type GSAGE --top_k 15 --do_split
```
Or you can train the model with ___Dense Adjacency Matrix___ on small dataset.
```bash
python src/main.py --dataset R8 --gnn_type GSAGE --top_k 15 --do_split --use_dense
```