import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class Top2VecMLP:
    def __init__(self, topic_size, X_path, Y_path, TEST900_PATH, GROUND_TRUTH, OUTPUT_DIR, TITLE900_PATH, test_db_key,
                 test_size=0.2, random_state=42, batch_size=32, num_epochs=300, patience=10):
        self.topic_size = topic_size
        self.X_path = X_path
        self.Y_path = Y_path
        self.TEST900_PATH = TEST900_PATH
        self.GROUND_TRUTH = GROUND_TRUTH
        self.OUTPUT_DIR = OUTPUT_DIR
        self.TITLE900_PATH = TITLE900_PATH
        self.test_db_key = test_db_key
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience  
        self.best_val_loss = float('inf')  
        self.epochs_without_improvement = 0  

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.X, self.Y, self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test, \
        self.scaler, self.model, self.criterion, self.optimizer, self.train_loader, self.val_loader, \
        self.optimal_thresholds, self.Y_pred, self.Y_pred_proba = [None] * 17

        self.ground_truth_df = pd.read_csv(GROUND_TRUTH, encoding='utf-8-sig')

    def load_data(self):
        X = pd.read_csv(self.X_path, header=0)
        Y = pd.read_csv(self.Y_path, header=0)
        self.Y_columns = Y.columns
        self.Y_shape = Y.shape

        X['Embedding Vector'] = X['Embedding Vector'].astype(str).apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )

        test_mask = (X['Document ID'] >= 138) & (X['Document ID'] <= 158)
        self.X_test = X[test_mask]
        self.Y_test = Y[test_mask].values

        X_train_val = X[~test_mask]
        Y_train_val = Y[~test_mask]

        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=self.test_size, random_state=self.random_state)

        X_train = X_train.drop(columns=['Document ID'])
        X_val = X_val.drop(columns=['Document ID'])
        self.X_test = self.X_test.drop(columns=['Document ID'])

        self.X_train = np.stack(X_train['Embedding Vector'].values)
        self.X_val = np.stack(X_val['Embedding Vector'].values)
        self.X_test = np.stack(self.X_test['Embedding Vector'].values)

        self.Y_train = Y_train.values
        self.Y_val = Y_val.values

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        train_dataset = TensorDataset(torch.FloatTensor(self.X_train), torch.FloatTensor(self.Y_train))
        val_dataset = TensorDataset(torch.FloatTensor(self.X_val), torch.FloatTensor(self.Y_val))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def create_model(self, n_inputs, n_outputs):
        class MultiLabelClassifier(nn.Module):
            def __init__(self, n_inputs, n_outputs):
                super(MultiLabelClassifier, self).__init__()
                self.hidden = nn.Linear(n_inputs, 20)
                self.output = nn.Linear(20, n_outputs)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.hidden(x))
                x = self.sigmoid(self.output(x))
                return x
        return MultiLabelClassifier(n_inputs, n_outputs)
    

    def train_model(self, save_path):
        n_inputs, n_outputs = self.X_train.shape[1], self.Y_train.shape[1]
        self.model = self.create_model(n_inputs, n_outputs)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())

        train_loss_list = []
        val_loss_list = []
        epoch_list = []

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    outputs = self.model(batch_X)
                    val_loss = self.criterion(outputs, batch_y)
                    total_val_loss += val_loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = total_val_loss / len(self.val_loader)

            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
            epoch_list.append(epoch + 1)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0  
                torch.save(self.model.state_dict(), f"{self.OUTPUT_DIR}/mlp_model.pth")
                print(f"Model improved. Model saved at epoch {epoch + 1}")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break 

        self.plot_loss(train_loss_list, val_loss_list, save_path)



    def calculate_optimal_thresholds(self):
        self.model.eval()
        with torch.no_grad():
            self.Y_pred_proba = self.model(torch.FloatTensor(self.X_test)).numpy()

        self.optimal_thresholds = []
        for i in range(self.Y_test.shape[1]):
            if np.sum(self.Y_test[:, i]) == 0:
                self.optimal_thresholds.append(0.3)
                continue
            fpr, tpr, thresholds = roc_curve(self.Y_test[:, i], self.Y_pred_proba[:, i])
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = float(thresholds[optimal_idx])
            self.optimal_thresholds.append(optimal_threshold)

        self.Y_pred = (self.Y_pred_proba >= np.array(self.optimal_thresholds)).astype(int)

    def calculate_performance_metrics(self):
        f1_micro = f1_score(self.Y_test, self.Y_pred, average="micro")
        print(f"Micro F1 Score: {f1_micro:.4f}")
        f1_macro = f1_score(self.Y_test, self.Y_pred, average="macro")
        print(f"Macro F1 Score: {f1_macro:.4f}")
        f1_weighted = f1_score(self.Y_test, self.Y_pred, average="weighted")
        print(f"Weighted F1 Score: {f1_weighted:.4f}")

        optimal_thresholds_df = pd.DataFrame({
            "class_name": self.Y_columns.tolist(),
            "optimal_threshold": self.optimal_thresholds
        })
        optimal_thresholds_df.to_csv(f"{self.OUTPUT_DIR}/optimal_thresholds_mlp.csv", index=False)
        print(f"\nOptimal thresholds saved to '{self.OUTPUT_DIR}/optimal_thresholds_mlp.csv'.")

    
    def calculate_hit_at_k(self, y_true, y_proba, k):
        hits = 0
        for true, proba in zip(y_true, y_proba):
            top_k_indices = np.argsort(proba)[-k:][::-1]
            if any(true[i] == 1 for i in top_k_indices):
                hits += 1
        return hits / len(y_true)

    def calculate_hits(self):
        hit_1 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 1)
        hit_3 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 3)
        hit_5 = self.calculate_hit_at_k(self.Y_test, self.Y_pred_proba, 5)

        print(f"Hit@1: {hit_1:.4f}")
        print(f"Hit@3: {hit_3:.4f}")
        print(f"Hit@5: {hit_5:.4f}")

    def predict_label(self, row, column_names):
        return ', '.join(column_names[row == 1])

    def predict_label_with_proba(self, row, proba_row, column_names):
        labels_with_proba = [
            (column_names[i], proba_row[i])
            for i in range(len(row)) if row[i] == 1
        ]
        labels_with_proba.sort(key=lambda x: x[1], reverse=True)
        return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])

    def load_test900_data(self):
        self.test900_df = pd.read_csv(self.TEST900_PATH, header=0)

        self.test900_df['Embedding Vector'] = self.test900_df['Embedding Vector'].astype(str).apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )
        test900_ids = self.test900_df['Document ID'].values
        self.X_test_900 = self.scaler.transform(np.stack(self.test900_df.drop(columns=['Document ID'])['Embedding Vector'].values))
        
        self.model.eval()
        with torch.no_grad():
            Y_pred_proba_900 = self.model(torch.FloatTensor(self.X_test_900)).numpy()

        Y_pred_900 = (Y_pred_proba_900 >= np.array(self.optimal_thresholds)).astype(int)

        return test900_ids, Y_pred_900, Y_pred_proba_900

    def load_titles(self, title_file):
        title_dict = {}
        with open(title_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '-' in line:
                    try:
                        db_key, title = line.split('-', 1)
                        title_dict[db_key] = title
                    except ValueError:
                        print(f"Skipping line due to unexpected format: {line}")
        return title_dict

    def predict_all_labels_with_proba(self, proba_row, column_names):
        labels_with_proba = list(zip(column_names, proba_row))
        labels_with_proba.sort(key=lambda x: x[1], reverse=True)
        return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])

    def save_test900_results(self):
        test900_ids, Y_pred_900, Y_pred_proba_900 = self.load_test900_data()
        Y_pred_full_900 = np.zeros((Y_pred_900.shape[0], self.Y_shape[1]))
        Y_pred_full_900[:, :] = Y_pred_900

        title_dict = self.load_titles(self.TITLE900_PATH)

        label_res_with_all_df_900 = pd.DataFrame({
            'DB Key': test900_ids,
            'Title': [title_dict.get(str(db_key), 'Unknown') for db_key in test900_ids],
            'Model': 'Top2Vec-MLP',
            'Labels': [self.predict_all_labels_with_proba(Y_pred_proba_900[i], self.Y_columns)
                       for i in range(len(test900_ids))]
        })

        label_res_with_all_df_900_path = f"{self.OUTPUT_DIR}/test_900_all_labels.csv"
        label_res_with_all_df_900.to_csv(label_res_with_all_df_900_path, index=False, encoding='utf-8-sig')

        print(f'900개 테스트 데이터 모든 라벨 확률 예측 결과 저장 경로: {label_res_with_all_df_900_path}')

        label_res_with_prob_df_900 = pd.DataFrame({
            'DB Key': test900_ids,
            'Title': [title_dict.get(str(db_key), 'Unknown') for db_key in test900_ids],
            'Model': 'Top2Vec-MLP',
            'Labels': [self.predict_label_with_proba(Y_pred_900[i], Y_pred_proba_900[i], self.Y_columns)
                       for i in range(len(test900_ids))]
        })
        label_res_with_prob_df_900_path = f"{self.OUTPUT_DIR}/test_900_predictions_with_prob.csv"
        label_res_with_prob_df_900.to_csv(label_res_with_prob_df_900_path, index=False, encoding='utf-8-sig')

        print(f'900개 테스트 데이터 예측 결과 저장 경로: {label_res_with_prob_df_900_path}')

    def plot_loss(self, train_loss_list, val_loss_list, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
        plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f'Loss plot saved to {save_path}')

    def save_results(self):
        lable_res_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-MLP',
            'Labels': [self.predict_label(row, self.Y_columns) for row in self.Y_pred]
        })
        lable_res_df = pd.concat([lable_res_df, self.ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'],
                                                                                            ascending=[True, True]).reset_index(
            drop=True)
        lable_res_path = f"{self.OUTPUT_DIR}/mlp_predicted_labels.csv"
        lable_res_df.to_csv(lable_res_path, index=False, encoding='utf-8-sig')

        print(f'각 문서의 라벨 예측 결과 저장 경로: {lable_res_path}')

        lable_res_with_prob_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-MLP',
            'Labels': [self.predict_label_with_proba(row, proba_row, self.Y_columns)
                    for row, proba_row in zip(self.Y_pred, self.Y_pred_proba)]
        })

        label_res_with_prob_df = pd.concat([lable_res_with_prob_df, self.ground_truth_df], axis=0).sort_values(
            by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
        lable_res_with_prob_path = f"{self.OUTPUT_DIR}/mlp_predicted_labels_with_prob.csv"
        label_res_with_prob_df.to_csv(lable_res_with_prob_path, index=False, encoding='utf-8-sig')

        print(f'각 문서의 라벨 및 확률 예측 결과 저장 경로: {lable_res_with_prob_path}')

        total_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-MLP',
            'Labels': [self.predict_all_labels_with_proba(proba_row, self.Y_columns)
                    for proba_row in self.Y_pred_proba]
        })
        total_df = pd.concat([total_df, self.ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'],
                                                                                    ascending=[True, True]).reset_index(
            drop=True)
        all_labels_results_path = f"{self.OUTPUT_DIR}/mlp_predicted_all_labels.csv"
        total_df.to_csv(all_labels_results_path, index=False, encoding='utf-8-sig')

        print(f'각 문서에 대한 모든 라벨의 확률값 결과의 경로: {all_labels_results_path}')


    def run(self):
        self.load_data()
        save_path = f'{self.OUTPUT_DIR}/loss_plot.png'
        self.train_model(save_path)
        self.calculate_optimal_thresholds()
        self.calculate_performance_metrics()
        self.calculate_hits()
        self.save_results()  
        self.save_test900_results()


if __name__ == "__main__":
    TOPIC_SIZE = 'minor'
    X_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_163.csv'
    Y_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_{TOPIC_SIZE}.csv'
    TEST900_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_900.csv'
    GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/{TOPIC_SIZE}_GT.csv'
    OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/MLP/{TOPIC_SIZE}'
    TITLE900_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/title_900.txt'
    test_db_key = [453073, 453074, 453075, 453076, 453077, 453078, 453079, 453082, 453083, 453084, 453093,
                    453095, 453096, 453097, 452970, 453102, 453104, 453105, 453110, 453114, 453116]

    mlp = Top2VecMLP(
        TOPIC_SIZE, X_PATH, Y_PATH, TEST900_PATH, GROUND_TRUTH, OUTPUT_DIR, TITLE900_PATH, test_db_key
    )
    mlp.run()
