import os
import dgl
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from src.utils import matrix_to_edges
from src.modules import ClassNet, GNNModel, SentenceBERT


class GraphClassifier:
    def __init__(self, in_channels, hidden_channels, num_classes,
                 gnn_type, encode_model, end2end, readout, dropout, lr, device):
        super(GraphClassifier, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.model = ClassNet(in_channels, hidden_channels, num_classes,
                              gnn_type, encode_model, end2end, readout, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def __str__(self):
        return (f"GraphClassifier(in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, num_classes={self.num_classes})")

    def run(self, train_subgraphs, eval_subgraphs, graph_topo, all_text, dense):
        train_loss = self._train(train_subgraphs, self.optimizer, self.criterion)
        eval_acc = self._eval(eval_subgraphs)
        node_logits, node_feat = self._propagate(graph_topo, all_text, dense)
        return train_loss, eval_acc, node_logits, node_feat

    def _train(self, subgraphs, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()
        total_loss = 0
        progress_bar = tqdm(subgraphs, desc=f"Training process")
        for subgraph in progress_bar:
            centroid_label = subgraph[0].to(self.device)  # shape: [1]
            centroid_logit = self.model(subgraph[1], subgraph[2].to(self.device), subgraph[3])
            # Get Loss
            loss = criterion(centroid_logit, centroid_label)
            total_loss += loss.item()
            loss.backward()
            progress_bar.set_postfix({'loss': loss.item()})
        optimizer.step()
        return total_loss / len(subgraphs)

    def _eval(self, subgraphs):
        self.model.eval()
        correct = 0
        progress_bar = tqdm(subgraphs, desc=f"Evaluating process")
        with torch.no_grad():
            for subgraph in progress_bar:
                centroid_label = subgraph[0].item()  # shape: [1]
                centroid_logit = self.model(subgraph[1], subgraph[2].to(self.device), subgraph[3])
                # Get Prediction
                centroid_pred = centroid_logit.argmax(dim=0)
                correct += (centroid_pred.item() == centroid_label)
                progress_bar.set_postfix({'correct': correct})
        return correct / len(subgraphs)

    def _propagate(self, graph_topo, all_text, dense):
        """
        执行全图传播，获取全图节点的 logits
        :param graph_topo:
        :param all_text:
        :param dense:
        :return:
        """
        with torch.no_grad():
            node_feat = self.model.encoder.encode(all_text)
            edge_index, edge_weight = matrix_to_edges(graph_topo, dense)
            graph = dgl.graph((edge_index[0], edge_index[1]))
            node_logits = self.model.gnn(graph, node_feat, edge_weight)
        return node_logits, node_feat

    def test(self, subgraphs):
        self.model.eval()
        preds, labels = [], []
        progress_bar = tqdm(subgraphs, desc=f"Testing process")
        with torch.no_grad():
            for subgraph in progress_bar:
                centroid_label = subgraph[0].item()  # shape: [1]
                centroid_logit = self.model(subgraph[1], subgraph[2].to(self.device), subgraph[3])
                # Get Prediction
                centroid_pred = centroid_logit.argmax(dim=0)
                preds.append(centroid_pred.item())
                labels.append(centroid_label)
        return (
            metrics.classification_report(labels, preds, digits=4, zero_division=0),
            metrics.precision_recall_fscore_support(labels, preds, average='macro', zero_division=0),
            metrics.precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
        )

    @staticmethod
    def save_model(path, best_model):
        os.makedirs(path, exist_ok=True)
        best_model.encoder.save(path)
        best_model.gnn.save(path)

    def load_model(self, path):
        self.model.encoder = SentenceBERT.load(path, self.device)
        self.model.gnn = GNNModel.load(path).to(self.device)
