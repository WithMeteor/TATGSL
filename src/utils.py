from collections import defaultdict
import torch.nn.functional as func
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
import numpy as np
import warnings
import logging
import random
import torch
import json
import os

warnings.filterwarnings("ignore", category=UserWarning, message="Sparse CSR tensor support is in beta state.*")
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

data_args = {
    'ohsumed':
        {
            # 'train_size': 3357,
            # 'test_size': 4043,
            # 'valid_size': 336,
            "num_classes": 23
         },
    '20ng':
        {
            # 'train_size': 11314,
            # 'test_size': 7532,
            # 'valid_size': 1131,
            "num_classes": 20
         },
    'R8':
        {
            # 'train_size': 5485,
            # 'test_size': 2189,
            # 'valid_size': 548,
            "num_classes": 8
         },
    'R52':
        {
            # 'train_size': 6532,
            # 'test_size': 2568,
            # 'valid_size': 653,
            "num_classes": 52
         },
    'AGNews':
        {
            # 'train_size': 6000,
            # 'test_size': 3000,
            # 'valid_size': 600,
            "num_classes": 4
         },
    'snippets':
        {
            # 'train_size': 10060,
            # 'test_size': 2280,
            # 'valid_size': 1006,
            "num_classes": 8
         },
}


def logging_config(args, ablation=False, sensitive=False, search=False, trial_id=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if search:
        save_dir = (f'{args.save_path}/{trial_id}-{args.dataset}-{args.gnn_type}-'
                    f'{args.gca_hidden_channels}-{args.gsl_hidden_channels}-'
                    f'{args.top_k}-{args.num_hop}')
    elif sensitive:
        save_dir = f'{args.save_path}/{args.dataset}-{args.train_per_label}-{args.top_k}-{args.num_hop}-{args.iter}'
    elif ablation:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f'{args.save_path}/{args.dataset}-{args.no_finetune}-{not args.end2end}-{args.no_gsl}-{time_str}'
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f'{args.save_path}/{args.dataset}-{args.gnn_type}-{time_str}'

    os.makedirs(save_dir, exist_ok=True)

    # 1. 创建文件Handler，写入i.log
    file_handler = logging.FileHandler(f'{save_dir}/run.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # 2. 创建控制台Handler，输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只输出INFO及以上级别
    console_handler.setFormatter(log_formatter)

    # 将两个Handler添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, save_dir


def split_data(dataset, data_path,
               train_per_label, eval_per_label, test_per_label,
               random_seed: int = 42):
    # 设置随机种子
    random.seed(random_seed)

    text_list = []
    with open(f"{data_path}/temp/{dataset}.texts.remove.txt", "r") as f:
        for text in f.readlines():
            text_list.append(text.strip())
    label_list = np.load(f"{data_path}/temp/{dataset}.targets.npy").tolist()

    # 将样本按类别分组
    class_samples = defaultdict(list)
    for text, label in zip(text_list, label_list):
        class_samples[label].append(text)

    # 初始化结果字典
    data_splits = {
        'train': [],  # 训练集
        'eval': [],  # 验证集
        'test': [],  # 测试集
        'unlabeled': []  # 无标签集
    }

    # 对每个类别进行处理
    for label, samples in class_samples.items():
        # 打乱样本顺序
        random.shuffle(samples)

        # 训练集采样
        train_samples = samples[:train_per_label] if len(samples) >= train_per_label else samples
        data_splits['train'].extend([{'text': text, 'label': label} for text in train_samples])

        # 剩余样本
        remaining_samples = samples[len(train_samples):]

        # 如果该类样本数不足 train_per_label，则不划分验证集和测试集
        if len(samples) <= train_per_label:
            continue

        # 划分验证集和测试集
        # 如果剩余样本数不足 eval_per_label + test_per_label，则均匀划分验证集和测试集，不设置无标签集
        if len(remaining_samples) <= eval_per_label + test_per_label:
            split_point = len(remaining_samples) // 2
            valid_samples = remaining_samples[:split_point]
            test_samples = remaining_samples[split_point:]
            unlabeled_samples = []
        else:  # 否则按既定方式划分，剩余的作为无标签集
            valid_samples = remaining_samples[:eval_per_label]
            test_samples = remaining_samples[eval_per_label:eval_per_label+test_per_label]
            unlabeled_samples = remaining_samples[eval_per_label+test_per_label:]

        data_splits['eval'].extend([{'text': text, 'label': label} for text in valid_samples])
        data_splits['test'].extend([{'text': text, 'label': label} for text in test_samples])
        data_splits['unlabeled'].extend([{'text': text, 'label': label} for text in unlabeled_samples])

    save_path = f"{data_path}/split/"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/{dataset}-t{train_per_label}v{eval_per_label}.json", "w") as f:
        json.dump(data_splits, f, indent=4)


def matrix_to_edges(matrix, dense: bool):
    """
    将稠密/稀疏邻接矩阵转化为边邻接表和权重列表
    :param matrix:
    :param dense:
    :return:
        edge_index: shape: [2, |E|]
        edge_weight: shape: [|E|]
    """
    if dense:
        edge_index = matrix.nonzero().t()  # for dense matrix
        edge_weight = matrix[edge_index[0], edge_index[1]]
    else:
        edge_index = matrix.indices()  # for sparse matrix
        edge_weight = matrix.values()
    return edge_index, edge_weight


def cal_homophily(graph_topo, node_label, dense: bool):
    """
    计算拓扑结构的同质性，需要提供节点标签
    :param graph_topo:
    :param node_label:
    :param dense:
    :return:
    """
    graph_topo = graph_topo.detach().cpu()
    if not dense:
        graph_topo_csr = graph_topo.to_sparse_csr()
        graph_crow = graph_topo_csr.crow_indices()
        graph_cols = graph_topo_csr.col_indices()
    else:
        graph_crow, graph_cols = None, None

    n = graph_topo.size(0)
    total_same_label = 0
    total_neighbors = 0
    for i in range(n):
        if dense:  # for dense matrix
            neighbors = (graph_topo[i] > 0).nonzero().squeeze(-1)
        else:  # for sparse matrix
            start, end = graph_crow[i], graph_crow[i + 1]
            neighbors = graph_cols[start: end]
        neighbors = neighbors[neighbors != i]  # 排除自环
        same_label = (node_label[neighbors] == node_label[i]).sum().item()
        total_same_label += same_label
        total_neighbors += len(neighbors)
    graph_homo = total_same_label / total_neighbors if total_neighbors > 0 else 0.0
    return graph_homo


def get_similarity(hidden: torch.Tensor):
    embeddings = func.normalize(hidden, dim=1, p=2)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    return sim_matrix


def top_k_sparsify(dense_matrix, k, dense):
    if dense:
        _, indices = dense_matrix.topk(k=int(k), dim=-1)
        mask = torch.zeros(dense_matrix.shape).to(dense_matrix.device)
        mask[torch.arange(dense_matrix.shape[0]).view(-1, 1), indices] = 1.

        mask.requires_grad = False
        dense_matrix = dense_matrix * mask
    else:
        values, indices = dense_matrix.topk(k=int(k), dim=-1)
        rows = torch.arange(dense_matrix.shape[0]).unsqueeze(1).expand(-1, k).flatten().to(dense_matrix.device)
        cols = indices.flatten()
        values = values.flatten()
        dense_matrix = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, dense_matrix.shape)
    return dense_matrix


def normalize_graph(graph_topo, dense):
    if dense:
        inv_sqrt_degree = 1. / (torch.sqrt(graph_topo.sum(dim=1, keepdim=False)) + 1e-10)
        graph_topo = inv_sqrt_degree[:, None] * graph_topo * inv_sqrt_degree[None, :]
    else:
        degrees = torch.sparse.sum(graph_topo, dim=1).to_dense()  # (n,)
        inv_sqrt_degree = 1.0 / (torch.sqrt(degrees) + 1e-10)  # (n,)
        n = graph_topo.size(0)
        indices = torch.arange(n).unsqueeze(0).repeat(2, 1).to(graph_topo.device)
        diag = torch.sparse_coo_tensor(indices, inv_sqrt_degree, (n, n)).to(graph_topo.device)
        graph_topo = torch.sparse.mm(graph_topo, diag)
        graph_topo = torch.sparse.mm(diag, graph_topo)
    return graph_topo


def topo_postprocess(graph_topo, top_k, dense=False):
    """
    对稠密/稀疏邻接矩阵进行稀疏化、对称化、对称归一化
    :param graph_topo:
    :param top_k:
    :param dense:
    :return:
    """
    graph_topo = top_k_sparsify(graph_topo, k=top_k+1, dense=dense)
    """
    为什么先执行 ReLU 激活，再执行对称化，KL散度计算会报错？
    解释：当 ReLU 先作用于稀疏矩阵时，会将负值置零，导致矩阵的非零位置（稀疏结构）发生改变。
    此时再进行对称化 (A + A.T)/2，可能会意外引入新的非零元素。
    例如，非线性激活后 A[i,j]=0 但 A[j,i]>0，转置相加后 A[i,j] 变为非零，
    这会破坏稀疏矩阵的梯度链（某些边在反向传播时突然出现或消失），导致 KLDivLoss 计算梯度时出现张量维度不一致。
    尽管输入 criterion 计算 loss 的张量形状相同，但稀疏格式的底层索引可能已混乱
    因此，我们改进后处理过程，先做对称化、再执行非线性激活
    """
    graph_topo = (graph_topo + graph_topo.T) / 2
    graph_topo = func.relu(graph_topo)
    graph_topo = normalize_graph(graph_topo, dense=dense)

    return graph_topo


def get_pseudo_label(logits, all_label, train_mask):
    num_classes = torch.max(all_label).item() + 1
    pseudo_label = torch.argmax(logits, dim=-1, keepdim=True)
    pseudo_label[train_mask, 0] = all_label[train_mask]
    one_hot_label = torch.zeros(all_label.shape[0], num_classes).to(logits.device)
    return one_hot_label.scatter(1, pseudo_label, 1).detach()


def count_confident_rate(logits: torch.Tensor, confidence_threshold=0.9):
    """
    统计模型预测中有把握的对象数量

    参数:
        logits (torch.Tensor): 模型输出的logits，形状为 N×C
        confidence_threshold (float): 置信度阈值(0-1之间)

    返回:
        float: 有把握预测的对象数量比例
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    confident_mask = max_probs > confidence_threshold
    confident_count = torch.sum(confident_mask).item()
    return confident_count / probs.size(0)


def save_graph(topo2save: torch.Tensor, save_dir):
    torch.save(topo2save.indices(), f'{save_dir}/learned.graph.indi.pt')
    torch.save(topo2save.values(), f'{save_dir}/learned.graph.wght.pt')


class Record:
    def __init__(self, save_path):
        # 初始化四个指标的记录列表
        self.gca_train_loss = []
        self.gca_eval_acc = []
        self.gsl_train_loss = []
        self.gsl_eval_homo = []
        self.save_path = save_path

    def add_record(self, gca_train_loss, gca_eval_acc, gsl_train_loss, gsl_eval_homo):
        """添加一个epoch的记录"""
        self.gca_train_loss.append(gca_train_loss)
        self.gca_eval_acc.append(gca_eval_acc)
        self.gsl_train_loss.append(gsl_train_loss)
        self.gsl_eval_homo.append(gsl_eval_homo)

    def visualize(self):
        """可视化训练过程中的指标变化"""
        epochs = range(1, len(self.gca_train_loss) + 1)

        # 创建图形和子图布局
        plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # 第一张图：训练损失（双y轴）
        ax0 = plt.subplot(gs[0])
        ax0.set_title('Training Loss')

        # GCA训练损失（左轴）
        color = 'tab:blue'
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('GCA Loss', color=color)
        line1, = ax0.plot(epochs, self.gca_train_loss, color=color, label='GCA Loss')
        ax0.tick_params(axis='y', labelcolor=color)

        # 自动调整y轴范围，留出10%的空白
        y_min = min(self.gca_train_loss)
        y_max = max(self.gca_train_loss)
        margin = (y_max - y_min) * 0.1
        ax0.set_ylim(y_min - margin, y_max + margin)

        # GSL训练损失（右轴）
        ax0_r = ax0.twinx()
        color = 'tab:orange'
        ax0_r.set_ylabel('GSL Loss', color=color)
        line2, = ax0_r.plot(epochs, self.gsl_train_loss, color=color, label='GSL Loss')
        ax0_r.tick_params(axis='y', labelcolor=color)

        # 自动调整右y轴范围
        y_min = min(self.gsl_train_loss)
        y_max = max(self.gsl_train_loss)
        margin = (y_max - y_min) * 0.1
        ax0_r.set_ylim(y_min - margin, y_max + margin)

        # 添加图例
        lines = [line1, line2]
        ax0.legend(lines, [l.get_label() for l in lines], loc='upper right')

        # 第二张图：评估指标（双y轴）
        ax1 = plt.subplot(gs[1])
        ax1.set_title('Evaluation Metrics')

        # GCA评估准确率（左轴）
        color = 'tab:green'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('GCA Accuracy', color=color)
        line3, = ax1.plot(epochs, self.gca_eval_acc, color=color, label='GCA Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)

        # 自动调整y轴范围
        y_min = min(self.gca_eval_acc)
        y_max = max(self.gca_eval_acc)
        margin = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - margin, y_max + margin)

        # GSL评估同质性（右轴）
        ax1_r = ax1.twinx()
        color = 'tab:red'
        ax1_r.set_ylabel('GSL Homogeneity', color=color)
        line4, = ax1_r.plot(epochs, self.gsl_eval_homo, color=color, label='GSL Homogeneity')
        ax1_r.tick_params(axis='y', labelcolor=color)

        # 自动调整右y轴范围
        y_min = min(self.gsl_eval_homo)
        y_max = max(self.gsl_eval_homo)
        margin = (y_max - y_min) * 0.1
        ax1_r.set_ylim(y_min - margin, y_max + margin)

        # 添加图例
        lines = [line3, line4]
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{self.save_path}/record.png')
