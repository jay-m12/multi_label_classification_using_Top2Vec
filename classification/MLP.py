import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import os


class MLP:
    def __init__(self, topic_size, X_path, Y_path, GROUND_TRUTH, OUTPUT_DIR, TITLE_PATH):
        self.topic_size = topic_size
        self.X_path = X_path
        self.Y_path = Y_path
        self.GROUND_TRUTH = GROUND_TRUTH
        self.OUTPUT_DIR = OUTPUT_DIR
        self.TITLE_PATH = TITLE_PATH

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.data = [None] * 16  
        self.X, self.Y, self.X_train, self.X_test, self.Y_train_df, self.Y_test_df, \
        self.Y_train_filtered, self.Y_test_filtered, self.test_document_ids, \
        self.model, self.optimal_thresholds, self.ground_truth_df, self.Y_pred, \
        self.y_proba_matrix, self.Y_pred_full, self.Y_proba_full = self.data

    def load_data(self):
        self.X = pd.read_csv(self.X_path, header=0)
        self.Y = pd.read_csv(self.Y_path, header=0)
        self.ground_truth_df = pd.read_csv(self.GROUND_TRUTH, encoding='utf-8-sig')

        self.title_df = pd.read_csv(self.TITLE_PATH, header=0)
        self.test_db_key = self.X['Document ID'].values
        self.title = self.title_df['연구보고서'].values

        self.X['Embedding Vector'] = self.X['Embedding Vector'].astype(str).apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )


        self.X_train = self.X.iloc[:730].copy()
        self.Y_train_df = self.Y.iloc[:730].copy()
        self.X_test = self.X.iloc[730:].copy()
        self.Y_test_df = self.Y.iloc[730:].copy()


        single_class_cols = [col for col in self.Y.columns if self.Y_train_df[col].nunique() == 1]
        self.Y_train_filtered = self.Y_train_df.drop(columns=single_class_cols)
        self.Y_test_filtered = self.Y_test_df.drop(columns=single_class_cols)

        print(f"제거된 컬럼 수: {len(single_class_cols)}/{self.Y.shape[1]}")
        print("제거된 컬럼 목록:", single_class_cols)

        self.test_document_ids = self.X_test['Document ID'].values

        self.X_train = self.X_train.drop(columns=['Document ID'])
        self.X_test = self.X_test.drop(columns=['Document ID'])
        self.X_train = np.stack(self.X_train['Embedding Vector'].values)
        self.X_test = np.stack(self.X_test['Embedding Vector'].values)

        self.Y_train = self.Y_train_filtered.values
        self.Y_test = self.Y_test_filtered.values


        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        train_dataset = TensorDataset(torch.FloatTensor(self.X_train), torch.FloatTensor(self.Y_train))
        test_dataset = TensorDataset(torch.FloatTensor(self.X_test), torch.FloatTensor(self.Y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def create_model(self, input_dim, output_dim):
        class MLPModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MLPModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, output_dim)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.sigmoid(self.fc2(x))
                return x

        return MLPModel(input_dim, output_dim)

    def train_model(self):
        input_dim, output_dim = self.X_train.shape[1], self.Y_train.shape[1]
        self.model = self.create_model(input_dim, output_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(130):  # 50 Epochs
            self.model.train()
            total_loss = 0
            for batch_X, batch_Y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_Y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # print(f"Epoch [{epoch + 1}/50], Loss: {total_loss / len(self.train_loader):.4f}")

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            self.y_proba_matrix = self.model(torch.FloatTensor(self.X_test)).numpy()
            self.Y_pred_proba = self.y_proba_matrix 
        
        self.optimal_thresholds = []
        for i in range(self.Y_test.shape[1]):
            if np.sum(self.Y_test[:, i]) == 0:
                self.optimal_thresholds.append(0.3)
                continue
            fpr, tpr, thresholds = roc_curve(self.Y_test[:, i], self.y_proba_matrix[:, i])
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            self.optimal_thresholds.append(float(thresholds[optimal_idx]))

        self.Y_pred = (self.y_proba_matrix >= np.array(self.optimal_thresholds)).astype(int)

    def calculate_hit_at_k(self, y_true, y_proba, k):
            hits = 0
            for true, proba in zip(y_true, y_proba):
                top_k_indices = np.argsort(proba)[-k:][::-1]
                if any(true[i] == 1 for i in top_k_indices):
                    hits += 1
            return hits / len(y_true)
    
    def calculate_performance_metrics(self):

        f1_micro = f1_score(self.Y_test, self.Y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(self.Y_test, self.Y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(self.Y_test, self.Y_pred, average="weighted", zero_division=0)


        precision_micro = precision_score(self.Y_test, self.Y_pred, average="micro", zero_division=0)
        recall_micro = recall_score(self.Y_test, self.Y_pred, average="micro", zero_division=0)
        precision_macro = precision_score(self.Y_test, self.Y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(self.Y_test, self.Y_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(self.Y_test, self.Y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(self.Y_test, self.Y_pred, average="weighted", zero_division=0)

        print('\n')
        print('========[MLP 중분류 성능]========')
        print('------------------[F1 Score]-------------------')
        print(f"Micro F1 Score: {f1_micro:.4f}")
        print(f"Macro F1 Score: {f1_macro:.4f}")
        print(f"Weighted F1 Score: {f1_weighted:.4f}")
        print('--------------[Precision / Recall]--------------')
        print(f"Micro Precision: {precision_micro:.4f}")
        print(f"Micro Recall: {recall_micro:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall: {recall_macro:.4f}")
        print(f"Weighted Precision: {precision_weighted:.4f}")
        print(f"Weighted Recall: {recall_weighted:.4f}")


        optimal_thresholds_df = pd.DataFrame({
            "class_name": self.Y_train_filtered.columns.tolist(),
            "optimal_threshold": self.optimal_thresholds
        })
        

        auc_scores = []
        for i in range(self.Y_test.shape[1]):
            if np.sum(self.Y_test[:, i]) == 0:
                auc_scores.append(None)
                continue

            try:
                auc = roc_auc_score(self.Y_test[:, i], self.Y_pred_proba[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(None)

        valid_auc_scores = [score for score in auc_scores if score is not None]
        average_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0.0

        print('-------------------[AUC]--------------------')
        print(f"Average AUC: {average_auc:.4f}")


        
        hit_1 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 1)
        hit_3 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 3)
        hit_5 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 5)
        
        print('------------------[Hit@K]-------------------')
        print(f"Hit@1: {hit_1:.4f}")
        print(f"Hit@3: {hit_3:.4f}")
        print(f"Hit@5: {hit_5:.4f}")


        optimal_thresholds_df.to_csv(f"{self.OUTPUT_DIR}/optimal_thresholds_mlp.csv", index=False)
        print(f"\nOptimal thresholds saved to '{self.OUTPUT_DIR}/optimal_thresholds_mlp.csv'.")


    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        self.calculate_performance_metrics()


if __name__ == "__main__":
    TOPIC_SIZE = 'minor'
    X_PATH = '/home/women/doyoung/Top2Vec/embedding/output/gpt_document_embeddings_900.csv'
    Y_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_gpt_{TOPIC_SIZE}.csv'
    TITLE_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/gpt_gt.csv'
    GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/gpt_{TOPIC_SIZE}_GT.csv'
    OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/MLP/{TOPIC_SIZE}'

    mlp = MLP(
        topic_size=TOPIC_SIZE,
        X_path=X_PATH,
        Y_path=Y_PATH,
        GROUND_TRUTH=GROUND_TRUTH,
        OUTPUT_DIR=OUTPUT_DIR,
        TITLE_PATH=TITLE_PATH
    )
    mlp.run()
