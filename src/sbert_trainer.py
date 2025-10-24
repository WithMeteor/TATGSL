import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as func
from collections import defaultdict
from src.modules import SentenceBERT
from torch.utils.data import Dataset, DataLoader


class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'anchor': self.data[idx]['anchor'],
            'positive': self.data[idx]['positive'],
            'negative': self.data[idx]['negative']
        }


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_pos = func.cosine_similarity(anchor, positive)
        distance_neg = func.cosine_similarity(anchor, negative)
        losses = func.relu(distance_neg - distance_pos + self.margin)
        return losses.mean()


class TripletEvaluator:
    def __init__(self, anchors, positives, negatives, device):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.device = device

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            # Get embeddings
            anchor_emb = model.encode(self.anchors)
            positive_emb = model.encode(self.positives)
            negative_emb = model.encode(self.negatives)

            # Calculate similarities
            pos_sim = func.cosine_similarity(anchor_emb, positive_emb).cpu().numpy()
            neg_sim = func.cosine_similarity(anchor_emb, negative_emb).cpu().numpy()

            # Calculate accuracy
            accuracy = np.mean(pos_sim > neg_sim)

            # Calculate mean similarities
            mean_pos_sim = pos_sim.mean()
            mean_neg_sim = neg_sim.mean()

        return {
            'cosine_accuracy': float(accuracy),
            'mean_pos_cosine': float(mean_pos_sim),
            'mean_neg_cosine': float(mean_neg_sim)
        }


class SBertTrainer:
    def __init__(self, model_name,
                 dataset_name,
                 plm_path,
                 ftm_path,
                 data_path,
                 device,
                 logger,
                 train_per_label=10,
                 eval_per_label=10,
                 test_per_label=20,
                 epochs=5,
                 lr=1e-5,
                 batch_size=16,
                 random_seed=42):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.plm_path = plm_path
        self.ftm_path = ftm_path
        self.data_path = data_path
        self.train_per_label = train_per_label
        self.eval_per_label = eval_per_label
        self.test_per_label = test_per_label
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_seed = random_seed
        self.logger = logger

    def load_data(self):
        with open(f"{self.data_path}/split/{self.dataset_name}-t{self.train_per_label}v{self.eval_per_label}.json", "r") as f:
            data = json.load(f)
        return data['train'], data['eval'], data['test']

    @staticmethod
    def prepare_triplet_data(data, random_seed: int = 42):
        # 设置随机种子
        random.seed(random_seed)
        class_dict = defaultdict(list)
        for example in data:
            class_dict[example["label"]].append(example["text"])

        classes = [c for c in class_dict.keys() if len(class_dict[c]) >= 2]
        class_num = len(classes)
        if class_num < 2:
            raise ValueError("至少需要两个类别且每个类别至少有两个样本")

        triplets = []
        for class_idx in classes:
            class_samples = class_dict[class_idx]
            other_classes = [c for c in classes if c != class_idx]

            used_negatives = set()

            for i in range(len(class_samples)):
                for j in range(i + 1, len(class_samples)):
                    anchor = class_samples[i]
                    positive = class_samples[j]

                    neg_class_id = len(triplets) % (class_num - 1)
                    neg_class = other_classes[neg_class_id]
                    neg_candidates = class_dict[neg_class]

                    available_negs = [n for n in neg_candidates if n not in used_negatives]
                    if not available_negs:
                        used_negatives = set()
                        available_negs = neg_candidates

                    negative = random.choice(available_negs)
                    used_negatives.add(negative)

                    triplets.append({
                        "anchor": anchor,
                        "positive": positive,
                        "negative": negative
                    })

        if not triplets:
            raise ValueError("无法生成有效的三元组，请检查数据分布")

        random.shuffle(triplets)
        return triplets

    @staticmethod
    def collate_fn(batch):
        anchors = [item['anchor'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]
        return {
            'anchors': anchors,
            'positives': positives,
            'negatives': negatives
        }

    def plm(self):
        return SentenceBERT(self.model_name, self.plm_path, self.device)

    def train(self, train_data, eval_data):
        model = SentenceBERT(self.model_name, self.plm_path, self.device)
        loss_fn = TripletLoss(margin=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        train_dataset = TripletDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=self.collate_fn)

        evaluator = TripletEvaluator(
            anchors=[ex["anchor"] for ex in eval_data],
            positives=[ex["positive"] for ex in eval_data],
            negatives=[ex["negative"] for ex in eval_data],
            device=self.device
        )

        best_accuracy = 0
        os.makedirs(f"{self.ftm_path}/{self.model_name}-{self.dataset_name}", exist_ok=True)

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch in progress_bar:
                optimizer.zero_grad()

                # Get embeddings
                anchor_emb = model(batch['anchors'])
                positive_emb = model(batch['positives'])
                negative_emb = model(batch['negatives'])

                # Compute loss
                loss = loss_fn(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            # Evaluation
            eval_results = evaluator.evaluate(model)
            eval_accuracy = eval_results['cosine_accuracy']
            self.logger.info(f"Epoch {epoch + 1} - SBERT Train Loss: {total_loss / len(train_loader):.4f}")
            self.logger.info(f"SBERT Eval Accuracy: {eval_accuracy:.4f}, "
                             f"Pos Sim: {eval_results['mean_pos_cosine']:.4f}, "
                             f"Neg Sim: {eval_results['mean_neg_cosine']:.4f}")

            # Save best model
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                torch.save(model.state_dict(),
                           f"{self.ftm_path}/{self.model_name}-{self.dataset_name}/best_model.pt")
                self.logger.info("Saved best SBERT model!")

        # Load best model for return
        model.load_state_dict(torch.load(f"{self.ftm_path}/{self.model_name}-{self.dataset_name}/best_model.pt"))
        return model

    def test(self, model, test_data):
        evaluator = TripletEvaluator(
            anchors=[ex["anchor"] for ex in test_data],
            positives=[ex["positive"] for ex in test_data],
            negatives=[ex["negative"] for ex in test_data],
            device=self.device
        )
        test_results = evaluator.evaluate(model)

        self.logger.info("SBERT Test Results:")
        self.logger.info(f"Accuracy: {test_results['cosine_accuracy']:.4f}")
        self.logger.info(f"Mean Positive Cosine: {test_results['mean_pos_cosine']:.4f}")
        self.logger.info(f"Mean Negative Cosine: {test_results['mean_neg_cosine']:.4f}")

        with open(f"{self.ftm_path}/{self.model_name}-{self.dataset_name}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    def run(self):
        train_data, eval_data, test_data = self.load_data()
        print(len(train_data), len(eval_data), len(test_data))
        train_triplets = self.prepare_triplet_data(train_data, self.random_seed)
        eval_triplets = self.prepare_triplet_data(eval_data, self.random_seed)
        test_triplets = self.prepare_triplet_data(test_data, self.random_seed)
        # print('Train triplets num:', len(train_triplets))
        # print('Eval triplets num:', len(eval_triplets))
        # print('Test triplets num:', len(test_triplets))

        model = self.train(train_triplets, eval_triplets)
        self.test(model, test_triplets)
        return model
