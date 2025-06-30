import torch
import torch.nn as nn
from src.modules import StrucNet
import torch.nn.functional as func
from src.utils import get_pseudo_label


class GraphLearner(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_hop, num_layers, top_k, dropout, lr, device):
        super(GraphLearner, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_hop = num_hop
        self.num_layers = num_layers
        self.top_k = top_k
        self.model = StrucNet(in_channels, hidden_channels, out_channels,
                              num_layers, top_k, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.device = device

    def __str__(self):
        return (f"GraphLearner(in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, "
                f"num_hop={self.num_hop}, num_layers={self.num_layers}, top_k={self.top_k})")

    def run(self, node_logits, node_feat, all_label, train_mask, dense):
        all_label = all_label.to(self.device)
        train_mask = train_mask.to(self.device)
        pseudo_label = get_pseudo_label(node_logits, all_label, train_mask)
        train_loss = self._train(pseudo_label, node_feat, self.optimizer, self.criterion, dense)
        graph_topo = self._eval(node_feat, dense)
        # graph_topo = graph_topo * (1 - alpha) + learned_topo * alpha
        return train_loss, graph_topo

    def _train(self, pseudo_label, node_feat, optimizer, criterion, dense):
        self.model.train()
        optimizer.zero_grad()
        graph_topo = self.model(node_feat, dense)
        q_dist = pseudo_label
        for _ in range(self.num_hop):
            if dense:
                q_dist = torch.matmul(graph_topo, q_dist)  # for dense matrix
            else:
                q_dist = torch.sparse.mm(graph_topo, q_dist)  # for sparse matrix
        q_dist = func.log_softmax(q_dist, dim=-1)
        # Get Loss
        loss = criterion(q_dist, pseudo_label)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _eval(self, node_feat, dense):
        self.model.eval()
        with torch.no_grad():
            graph_topo = self.model(node_feat, dense)
        return graph_topo
