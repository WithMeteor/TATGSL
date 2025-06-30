from src.utils import matrix_to_edges, cal_homophily, topo_postprocess, get_similarity
import torch
import json
import dgl
import networkx as nx


class GraphSampler:
    def __init__(self,
                 encode_model,
                 data_path: str,
                 dataset_name: str,
                 train_per_label=10,
                 eval_per_label=10,
                 test_per_label=20,
                 top_k=32,
                 dense=False,
                 save_path=None):
        self.encoder = encode_model
        self.dataset_name = dataset_name
        self.top_k = top_k
        self.dense = dense
        self.save_path = save_path
        self.corpus = []
        self.all_label = None
        self.train_mask = None
        self.train_per_label = train_per_label
        self.eval_per_label = eval_per_label
        self.test_per_label = test_per_label
        self.train_size = 0
        self.eval_size = 0
        self.test_size = 0
        self.load_data(data_path)
        self.get_train_mask()

    def __str__(self):
        return f"GraphSampler(dataset={self.dataset_name}, top_k={self.top_k})"

    def load_data(self, data_path):
        print("Loading and splitting data...", end='', flush=True)
        data_splits = self.load_split_data(data_path, self.train_per_label, self.eval_per_label)
        all_labels = []
        for tl_pair in data_splits['train'] + data_splits['eval'] + data_splits['test'] + data_splits['unlabeled']:
            self.corpus.append(tl_pair['text'])
            all_labels.append(tl_pair['label'])
        self.all_label = torch.tensor(all_labels)
        # 记录数据集实际分割大小
        self.train_size = len(data_splits['train'])
        self.eval_size = len(data_splits['eval'])
        self.test_size = len(data_splits['test'])

        print("\rLoading and splitting data... Done!")

    def load_split_data(self, data_path, train_per_label, eval_per_label):
        with open(f"{data_path}/split/{self.dataset_name}-t{train_per_label}v{eval_per_label}.json", "r") as f:
            data_splits = json.load(f)
        return data_splits

    def lookup_text(self, text_ids):
        temp_list = []
        for tid in text_ids:
            temp_list.append(self.corpus[tid])
        return temp_list

    def get_train_mask(self):
        total_size = len(self.corpus)
        # 初始化全为 False 的 tensor
        train_mask = torch.zeros(total_size, dtype=torch.bool)
        # 将训练集对应的位置设为 True
        train_mask[:self.train_size] = True
        self.train_mask = train_mask

    def induce_subgraph(self, graph: dgl.DGLGraph, centroid_list, num_hops=2):
        subgraph_list = []

        for centroid_id in centroid_list:
            # 获取k-hop子图节点
            nodes = dgl.khop_in_subgraph(graph, centroid_id, k=num_hops)[0].ndata[dgl.NID]
            # 提取诱导子图(包含原始边权重)
            subgraph = graph.subgraph(nodes, relabel_nodes=True)

            # 获取中心节点在新导出的子图中的映射
            new_centroid_id = (subgraph.ndata[dgl.NID] == centroid_id).nonzero().item()

            subgraph_text = self.lookup_text(nodes)
            subgraph_label = self.all_label[centroid_id]
            subgraph_list.append((subgraph_label, new_centroid_id, subgraph, subgraph_text))

        return subgraph_list

    def save_input(self, save_dir, graph_topo, node_feat):
        """ 保存初始的结构和特征 """
        feat2save = node_feat.cpu()
        topo2save = graph_topo.cpu()
        node_num = self.all_label.size(0)
        train_mask = torch.zeros(node_num, dtype=torch.bool)
        train_mask[:self.train_size] = True
        eval_mask = torch.zeros(node_num, dtype=torch.bool)
        eval_mask[self.train_size:self.train_size + self.eval_size] = True
        test_mask = torch.zeros(node_num, dtype=torch.bool)
        test_mask[self.train_size + self.eval_size:self.train_size + self.eval_size + self.test_size] = True
        torch.save(feat2save, f'{save_dir}/graph.feat.pt')
        torch.save(topo2save.indices(), f'{save_dir}/graph.indi.pt')
        torch.save(topo2save.values(), f'{save_dir}/graph.wght.pt')
        torch.save(self.all_label, f'{save_dir}/graph.label.pt')
        torch.save(train_mask, f'{save_dir}/mask.train.pt')
        torch.save(eval_mask, f'{save_dir}/mask.eval.pt')
        torch.save(test_mask, f'{save_dir}/mask.test.pt')

    def build_knn_graph(self):
        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder.encode(self.corpus)
        graph_topo = get_similarity(embeddings)
        graph_topo = topo_postprocess(graph_topo, self.top_k, self.dense)
        self.save_input(self.save_path, graph_topo, embeddings)
        return graph_topo

    def sample_graph(self, graph_topo):

        graph_homo = cal_homophily(graph_topo, self.all_label, self.dense)
        edge_index, edge_weight = matrix_to_edges(graph_topo, self.dense)
        # 创建 DGL 图，将边权重存储为边特征
        graph = dgl.graph((edge_index[0], edge_index[1]))
        graph.edata['weight'] = edge_weight

        train_range = range(self.train_size)
        train_subgraph = self.induce_subgraph(graph, train_range)

        eval_range = range(self.train_size, self.eval_size + self.train_size)
        eval_subgraph = self.induce_subgraph(graph, eval_range)

        test_range = range(self.eval_size + self.train_size, self.test_size + self.eval_size + self.train_size)
        test_subgraph = self.induce_subgraph(graph, test_range)

        graph = dgl.to_networkx(graph.cpu())
        connect_flag = nx.is_connected(graph.to_undirected())
        return train_subgraph, eval_subgraph, test_subgraph, graph_homo, connect_flag
