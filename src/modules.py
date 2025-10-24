import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func
from transformers import AutoTokenizer, AutoModel
from src.utils import topo_postprocess, get_similarity
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, GINConv, APPNPConv


class SentenceBERT(nn.Module):
    def __init__(self, model_name, model_path, device):
        super(SentenceBERT, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        self.device = device
        self.model.to(device)

    def forward(self, input_texts):
        # Tokenize and get embeddings
        inputs = self.tokenizer(input_texts, padding=True, truncation=True,
                                return_tensors="pt", max_length=128).to(self.device)
        outputs = self.model(**inputs)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_texts, batch_size=32):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(input_texts), batch_size):
                batch = input_texts[i:i + batch_size]
                embeddings.append(self.forward(batch))
        return torch.cat(embeddings, dim=0)

    def save(self, save_path):

        # 保存模型
        self.model.save_pretrained(save_path)
        # 保存分词器
        self.tokenizer.save_pretrained(save_path)
        print(f"SBERT model saved to {save_path}")

    @classmethod
    def load(cls, model_name, model_path, device):
        """从指定目录加载模型和分词器"""
        # 检查目录是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not exits.")

        model = cls(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        # print(f"SBERT model loaded from {model_path}")
        return model


class GNNModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes,
                 gnn_type, num_heads=4, prop_k=10, prop_alpha=0.1,
                 aggregator_type='mean', dropout=0.5):
        """
        统一的GNN模型封装

        参数:
            gnn_type (str): GNN类型，可选 'GCN', 'GAT', 'GIN', 'GSAGE', 'APPNP'
            in_feats (int): 输入特征维度
            h_feats (int): 隐藏层特征维度
            num_classes (int): 输出类别数
            num_heads (int): GAT使用的注意力头数
            aggregator_type (str): GraphSAGE的聚合类型，可选 'mean', 'gcn', 'pool', 'lstm'
        """
        super(GNNModel, self).__init__()
        self.gnn_type = gnn_type
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.aggregator_type = aggregator_type
        self.dropout = nn.Dropout(p=dropout)

        # 根据类型初始化对应的GNN层
        if self.gnn_type == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats, weight=True)
            self.conv2 = GraphConv(h_feats, num_classes, weight=True)
        elif self.gnn_type == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats, num_heads)
            self.conv2 = GATConv(h_feats * num_heads, num_classes, 1)
        elif self.gnn_type == 'GIN':
            lin1 = nn.Linear(in_feats, h_feats)
            lin2 = nn.Linear(h_feats, num_classes)
            self.conv1 = GINConv(lin1, 'sum')
            self.conv2 = GINConv(lin2, 'sum')
        elif self.gnn_type == 'GSAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type)
            self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type)
        elif self.gnn_type == 'APPNP':
            self.lin1 = nn.Linear(in_feats, h_feats)
            self.lin2 = nn.Linear(h_feats, num_classes)
            self.conv = APPNPConv(prop_k, prop_alpha)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}. Choose from 'GCN', 'GAT', 'GIN', 'GSAGE', 'APPNP'.")

    def forward(self, g, in_feat, edge_weight=None):
        """
        前向传播

        参数:
            g (DGLGraph): 输入图
            in_feat (Tensor): 节点特征
            edge_weight (Tensor, optional): 边权重

        返回:
            Tensor: 模型输出
        """
        with g.local_scope():

            if self.gnn_type == 'GCN':
                h = self.conv1(g, in_feat, edge_weight=edge_weight)
                h = self.dropout(func.relu(h))
                h = self.conv2(g, h, edge_weight=edge_weight)
            elif self.gnn_type == 'GAT':
                h = self.conv1(g, in_feat)
                h = torch.flatten(h, 1)
                h = self.dropout(func.elu(h))
                h = self.conv2(g, h)
                h = torch.squeeze(h)
            elif self.gnn_type == 'GIN':
                h = self.conv1(g, in_feat, edge_weight=edge_weight)
                h = self.dropout(func.relu(h))
                h = self.conv2(g, h, edge_weight=edge_weight)
            elif self.gnn_type == 'GSAGE':
                h = self.conv1(g, in_feat, edge_weight=edge_weight)
                h = self.dropout(func.relu(h))
                h = self.conv2(g, h, edge_weight=edge_weight)
            elif self.gnn_type == 'APPNP':
                h = func.relu(self.lin1(in_feat))
                h = self.dropout(self.lin2(h))
                h = self.conv(g, h, edge_weight=edge_weight)
            return h

    def save(self, path, model_name=None):

        if model_name is None:
            model_name = f"{self.gnn_type}_model_in{self.in_feats}_h{self.h_feats}_out{self.num_classes}.pth"

        save_path = os.path.join(path, model_name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'gnn_type': self.gnn_type,
            'in_feats': self.in_feats,
            'h_feats': self.h_feats,
            'num_classes': self.num_classes,
            'num_heads': self.num_heads,
            'aggregator_type': self.aggregator_type
        }, str(save_path))

        print(f"GNN model saved to {save_path}")

    @classmethod
    def load(cls, path, model_name=None, map_location=None):
        if os.path.isdir(path) and model_name is None:
            # 如果path是目录且没有指定model_name，则查找目录中的第一个.pth文件
            pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
            if not pth_files:
                raise FileNotFoundError(f"No .pth files found in directory {path}")
            model_name = pth_files[0]
            path = os.path.join(path, model_name)
        elif os.path.isdir(path) and model_name is not None:
            path = os.path.join(path, model_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)

        # 创建模型实例
        model = cls(
            gnn_type=checkpoint['gnn_type'],
            in_feats=checkpoint['in_feats'],
            h_feats=checkpoint['h_feats'],
            num_classes=checkpoint['num_classes'],
            num_heads=checkpoint.get('num_heads', 4),
            aggregator_type=checkpoint.get('aggregator_type', 'mean')
        )

        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])

        # print(f"GNN model loaded from {path}")
        return model


class ClassNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, gnn_type, encode_model, end2end, readout, dropout):
        super(ClassNet, self).__init__()
        self.end2end = end2end
        self.readout = readout
        self.encoder = encode_model
        self.gnn = GNNModel(in_channels, hidden_channels, num_classes, gnn_type, dropout=dropout)

    def forward(self, centroid_id, subgraph, node_text):
        """
        :param centroid_id: label of each subgraph, shape: 1
        :param subgraph:
            edge adjacency of each subgraph, shape: [2, |E|]
            edge weight of each subgraph, shape: [|E|]
        :param node_text: text node attribute of each subgraph, shape: [|V|]
        :return:
        """
        if self.end2end:
            node_feat = self.encoder.encode(node_text)  # shape: [n, d]
        else:
            with torch.no_grad():
                node_feat = self.encoder.encode(node_text)

        # Apply GNN
        out = self.gnn(subgraph, node_feat, subgraph.edata['weight'])
        # 将中心节点表示替换为平均值池化表示，以作为子图的读出表示时，在Ohsumed上效果下降明显
        # Get center node representations
        if self.readout == 'mean':
            graph_logit = torch.mean(out, dim=0)
        elif self.readout == 'max':
            graph_logit = torch.max(out, dim=0).values
        elif self.readout == 'centroid':
            graph_logit = out[centroid_id]
        else:
            raise ValueError(f"Unsupported readout type: {self.readout}. Choose from 'mean', 'max', 'centroid'.")
        return graph_logit


class StrucNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, top_k=10, dropout=0.2):
        super(StrucNet, self).__init__()
        assert in_channels == out_channels, "Input and output dimensions must be equal"
        self.top_k = top_k
        self.num_layers = num_layers
        self.learner = nn.ModuleList()
        # 第一层
        self.learner.append(nn.Linear(in_channels, hidden_channels))
        self.learner.append(nn.ReLU())
        self.learner.append(nn.Dropout(dropout))
        # 中间层
        for _ in range(num_layers - 2):
            self.learner.append(nn.Linear(hidden_channels, hidden_channels))
            self.learner.append(nn.ReLU())
            self.learner.append(nn.Dropout(dropout))
        # 最后一层
        self.learner.append(nn.Linear(hidden_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.learner):
            if isinstance(layer, nn.Linear):
                if i == len(self.learner) - 1:  # 最后一个线性层采用全0初始化
                    init.zeros_(layer.weight)
                    init.zeros_(layer.bias)
                else:
                    init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    init.zeros_(layer.bias)

    def forward(self, graph_feat, dense=False):
        identity = graph_feat
        for layer in self.learner:
            graph_feat = layer(graph_feat)
        graph_feat = identity + graph_feat  # 采用残差连接机制，确保训练初始特征不变
        # for layer in self.learner:
        #     graph_feat = layer(graph_feat)
        graph_topo = get_similarity(graph_feat)
        graph_topo = topo_postprocess(graph_topo, self.top_k, dense)
        return graph_topo
