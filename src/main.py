import copy
import torch
import argparse
from src.sbert_trainer import SBertTrainer
from src.graph_sampler import GraphSampler
from src.graph_learner import GraphLearner
from src.graph_classifier import GraphClassifier
from src.utils import data_args, logging_config, split_data, save_graph, Record


def parse_args():
    parser = argparse.ArgumentParser(description="Training Argument of TATGSL.")

    # 添加参数
    parser.add_argument('--dataset', type=str, default='ohsumed', choices=[
        'ohsumed', '20ng', 'R8', 'AGNews', 'snippets'], help='Name of the dataset')
    parser.add_argument('--gnn_type', type=str, default='GSAGE', choices=[
        'GCN', 'GAT', 'GIN', 'GSAGE', 'APPNP', 'CPGNN', 'GPRGNN', 'H2GNN'], help='Type of GNN (GCN, GAT, GIN, GSAGE, APPNP)')
    parser.add_argument('--readout', type=str, default='mean', choices=[
        'mean', 'max', 'centroid'], help='Readout function of GNN')
    parser.add_argument('--train_per_label', type=int, default=10, help='Training sample per label')
    parser.add_argument('--eval_per_label', type=int, default=10, help='Evaluating sample per label')
    parser.add_argument('--test_per_label', type=int, default=20, help='Testing sample per label')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=5, help='Number of finetune epochs')
    parser.add_argument('--in_channels', type=int, default=384, help='Dimension of node features')
    parser.add_argument('--gca_hidden_channels', type=int, default=128, help='Number of hidden channels for GCA')
    parser.add_argument('--gsl_hidden_channels', type=int, default=256, help='Number of hidden channels for GSL')
    parser.add_argument('--ftn_learning_rate', type=float, default=1e-5, help='Learning rate for Fine-tune')
    parser.add_argument('--gca_learning_rate', type=float, default=0.01, help='Learning rate for GCA')
    parser.add_argument('--gsl_learning_rate', type=float, default=0.001, help='Learning rate for GSL')
    parser.add_argument('--gca_dropout', type=float, default=0.5, help='Dropout rate for GCA')
    parser.add_argument('--gsl_dropout', type=float, default=0.2, help='Dropout rate for GSL')
    parser.add_argument('--mlp_layer', type=int, default=2, help='Number of mlp layers')
    parser.add_argument('--top_k', type=int, default=15, help='Top K value of graph sparsification')
    parser.add_argument('--num_hop', type=int, default=2, help='Number of label propagation hops')
    parser.add_argument('--cpu', action='store_true', help='Use cpu')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device index')
    parser.add_argument('--bert_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sbert name')
    parser.add_argument('--plm_path', type=str, default='plm', help='Pretrained sbert path')
    parser.add_argument('--ftm_path', type=str, default='ftm', help='Fine-tuned sbert model')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data')
    parser.add_argument('--save_path', type=str, default='./out', help='Path to save the model and log')
    parser.add_argument('--use_dense', action='store_true', help='Learn dense graph adj matrix')
    parser.add_argument('--do_split', action='store_true', help='Re divide train, eval and test set')
    parser.add_argument('--no_finetune', action='store_true', help='Do not finetune sbert model')
    parser.add_argument('--end2end', action='store_true', help='Do end2end training for plm')
    parser.add_argument('--no_gsl', action='store_true', help='Do not train gsl model')
    parser.add_argument('--conf_thresh', type=float, default=0.7, help='Confidence threshold τp')
    parser.add_argument('--no_conf', action='store_true', help='Not use confidence-threshold for GSL')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--iter', type=int, default=0, help='Running iteration')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger, save_dir = logging_config(args)
    logger.info(args)
    if args.do_split:  # 划分数据集
        logger.info(f"Split training set, evaluating set and testing set with"
                    f" {args.train_per_label}, {args.eval_per_label}, and {args.test_per_label} samples.")
        split_data(args.dataset, args.data_path,
                   args.train_per_label, args.eval_per_label, args.test_per_label,
                   args.seed)
    num_classes = data_args[args.dataset]['num_classes']  # 类别数
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    sbt = SBertTrainer(
        model_name=args.bert_name,
        dataset_name=args.dataset,
        plm_path=args.plm_path,
        ftm_path=args.ftm_path,
        data_path=args.data_path,
        device=device,
        logger=logger,
        train_per_label=args.train_per_label,
        eval_per_label=args.eval_per_label,
        test_per_label=args.test_per_label,
        lr=args.ftn_learning_rate,
        epochs=args.finetune_epochs,
        random_seed=args.seed
    )
    if args.no_finetune:
        encode_model = sbt.plm()
    else:
        logger.info("SBERT fine-tune start.")
        encode_model = sbt.run()
    del sbt
    gsp = GraphSampler(
        encode_model=encode_model,
        data_path=args.data_path,
        dataset_name=args.dataset,
        train_per_label=args.train_per_label,
        eval_per_label=args.eval_per_label,
        test_per_label=args.test_per_label,
        top_k=args.top_k,
        dense=args.use_dense,
        save_path=save_dir
    )
    gca = GraphClassifier(
        in_channels=args.in_channels,
        hidden_channels=args.gca_hidden_channels,
        num_classes=num_classes,
        gnn_type=args.gnn_type,
        encode_model=encode_model,
        end2end=args.end2end,
        readout=args.readout,
        dropout=args.gca_dropout,
        lr=args.gca_learning_rate,
        device=device
    )
    gsl = GraphLearner(
        in_channels=args.in_channels,
        hidden_channels=args.gsl_hidden_channels,
        out_channels=args.in_channels,
        num_hop=args.num_hop,
        top_k=args.top_k,
        num_layers=args.mlp_layer,
        dropout=args.gsl_dropout,
        lr=args.gsl_learning_rate,
        use_conf=not args.no_conf,
        conf_thresh=args.conf_thresh,
        device=device
    )
    test_subgraphs = None
    gca_best_model = None
    gca_best_acc = 0
    recorder = Record(save_dir)
    graph_topo = gsp.build_knn_graph()
    best_graph = None
    for epoch in range(args.epochs):
        train_subgraphs, eval_subgraphs, test_subgraphs, graph_homo, connect_flag = gsp.sample_graph(graph_topo)
        logger.info("Text Graph node classification start.")
        gca_train_loss, gca_eval_acc, node_logits, node_feat = gca.run(
            train_subgraphs, eval_subgraphs, graph_topo, gsp.corpus, args.use_dense)
        logger.info("Text Graph structure learning start.")
        if args.no_gsl:
            gsl_train_loss = -1
        else:
            gsl_train_loss, graph_topo = gsl.run(
                node_logits, node_feat, gsp.all_label, gsp.train_mask, args.use_dense)
        logger.info(f'Epoch: {epoch + 1}, '
                    f'Connected Graph: {connect_flag}, '
                    f'GSL Eval Homo: {graph_homo:.4f}, '
                    f'GCA Train Loss: {gca_train_loss:.4f}, '
                    f'GCA Eval Acc: {gca_eval_acc:.4f}, '
                    f'GSL Train Loss: {gsl_train_loss:.4f}')
        recorder.add_record(gca_train_loss, gca_eval_acc, gsl_train_loss, graph_homo)
        if gca_eval_acc > gca_best_acc:
            gca_best_acc = gca_eval_acc
            gca_best_model = copy.deepcopy(gca.model)
            best_graph = graph_topo.detach().cpu()
    gca.save_model(save_dir, gca_best_model)
    gca.load_model(args.bert_name, save_dir)
    save_graph(best_graph, save_dir)
    cls_report, macro_score, micro_score = gca.test(test_subgraphs)
    logger.info("Test Precision, Recall and F1-Score...")
    logger.info(cls_report)
    logger.info("Macro average Test Precision, Recall and F1-Score...")
    logger.info(macro_score)
    logger.info("Micro average Test Precision, Recall and F1-Score...")
    logger.info(micro_score)
    recorder.visualize()
    recorder.save()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()
