import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score

class Top2VecRandomForest:
    def __init__(self, topic_size, X_path, Y_path, TEST900_PATH, GROUND_TRUTH, OUTPUT_DIR, TITLE900_PATH, test_db_key):
        self.topic_size = topic_size
        self.X_path = X_path
        self.Y_path = Y_path
        self.TEST900_PATH = TEST900_PATH
        self.GROUND_TRUTH = GROUND_TRUTH
        self.OUTPUT_DIR = OUTPUT_DIR
        self.TITLE900_PATH = TITLE900_PATH
        self.test_db_key = test_db_key

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.data = [None] * 16
        self.X, self.Y, self.X_train, self.X_test, self.Y_train_df, self.Y_test_df, \
        self.Y_train_filtered, self.Y_test_filtered, self.test_document_ids, \
        self.model, self.optimal_thresholds, self.ground_truth_df, self.Y_pred, \
        self.y_proba_matrix, self.Y_pred_full, self.Y_proba_full = self.data

    def load_data(self):
        self.X = pd.read_csv(self.X_path, header=0)
        self.Y = pd.read_csv(f'{self.Y_path}', header=0)
        self.ground_truth_df = pd.read_csv(self.GROUND_TRUTH, encoding='utf-8-sig')

        self.X['Embedding Vector'] = self.X['Embedding Vector'].astype(str).apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )

        test_mask = (self.X['Document ID'] >= 138) & (self.X['Document ID'] <= 158)
        self.X_test = self.X[test_mask].copy()
        self.Y_test_df = self.Y.loc[test_mask].copy()
        self.X_train = self.X[~test_mask].copy()
        self.Y_train_df = self.Y.loc[~test_mask].copy()

        # 단일 클래스 컬럼 식별 및 제거
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

    def train_model(self):
        base_model = RandomForestClassifier(class_weight="balanced")
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(self.X_train, self.Y_train)
        self.Y_pred_proba = self.model.predict_proba(self.X_test)

    def calculate_optimal_thresholds(self):
        self.optimal_thresholds = []

        for i in range(self.Y_test.shape[1]):
            if np.sum(self.Y_test[:, i]) == 0:
                self.optimal_thresholds.append(0.3)
                continue

            n_classes = self.Y_pred_proba[i].shape[1]

            if n_classes >= 2:
                class_prob = self.Y_pred_proba[i][:, 1]
            else:
                class_prob = self.Y_pred_proba[i][:, 0]

            fpr, tpr, thresholds = roc_curve(self.Y_test[:, i], class_prob)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = float(thresholds[optimal_idx])
            self.optimal_thresholds.append(optimal_threshold)

        self.Y_pred = np.array([
            ((proba[:, 1] >= self.optimal_thresholds[i]) if proba.shape[1] > 1
             else (proba[:, 0] >= self.optimal_thresholds[i])).astype(int)
            for i, proba in enumerate(self.Y_pred_proba)
        ]).T

    def calculate_performance_metrics(self):
        f1_micro = f1_score(self.Y_test, self.Y_pred, average="micro")
        f1_macro = f1_score(self.Y_test, self.Y_pred, average="macro")
        f1_weighted = f1_score(self.Y_test, self.Y_pred, average="weighted")

        print(f"Micro F1 Score (Optimal Threshold): {f1_micro:.4f}")
        print(f"Macro F1 Score (Optimal Threshold): {f1_macro:.4f}")
        print(f"Weighted F1 Score (Optimal Threshold): {f1_weighted:.4f}")

        precision_micro = precision_score(self.Y_test, self.Y_pred, average="micro", zero_division=0)
        recall_micro = recall_score(self.Y_test, self.Y_pred, average="micro", zero_division=0)
        precision_macro = precision_score(self.Y_test, self.Y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(self.Y_test, self.Y_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(self.Y_test, self.Y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(self.Y_test, self.Y_pred, average="weighted", zero_division=0)
        
        print('-------[Precision/Recall]-------')
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
        optimal_thresholds_df.to_csv(f"{self.OUTPUT_DIR}/optimal_thresholds_rf.csv", index=False)
        print(f"\nOptimal thresholds saved to '{self.OUTPUT_DIR}/optimal_thresholds_rf.csv'.")

    def calculate_auc(self):
        auc_scores = []
        for i in range(self.Y_test.shape[1]):
            if np.sum(self.Y_test[:, i]) == 0:
                auc_scores.append(None)
                continue

            n_classes = self.Y_pred_proba[i].shape[1]
            class_prob = self.Y_pred_proba[i][:, 1] if n_classes >= 2 else self.Y_pred_proba[i][:, 0]

            try:
                auc = roc_auc_score(self.Y_test[:, i], class_prob)
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(None)

        valid_auc_scores = [score for score in auc_scores if score is not None]
        average_auc = np.mean(valid_auc_scores)
        print(f"Average AUC: {average_auc:.4f}")

    def prepare_predictions(self):
        self.y_proba_matrix = np.hstack([proba[:, -1].reshape(-1, 1) if proba.shape[1] > 1 else proba for proba in self.Y_pred_proba])

        self.Y_pred_full = np.zeros((self.Y_pred.shape[0], len(self.Y.columns)))
        self.Y_pred_full[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.Y_pred
        self.Y_proba_full = np.zeros((self.Y_pred.shape[0], len(self.Y.columns)))
        self.Y_proba_full[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.y_proba_matrix

    def calculate_hit_at_k(self, y_true, y_proba, k):
        hits = 0
        for true, proba in zip(y_true, y_proba):
            top_k_indices = np.argsort(proba)[-k:][::-1]
            if any(true[i] == 1 for i in top_k_indices):
                hits += 1
        return hits / len(y_true)

    def calculate_hits(self):
        hit_1 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 1)
        hit_3 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 3)
        hit_5 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 5)

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
        self.X_test_900 = np.stack(self.test900_df.drop(columns=['Document ID'])['Embedding Vector'].values)

        self.Y_pred_proba_900 = self.model.predict_proba(self.X_test_900)

        self.Y_pred_900 = np.array([
            ((proba[:, 1] >= self.optimal_thresholds[i]) if proba.shape[1] > 1
             else (proba[:, 0] >= self.optimal_thresholds[i])).astype(int)
            for i, proba in enumerate(self.Y_pred_proba_900)
        ]).T

        self.y_proba_matrix_900 = np.hstack([proba[:, -1].reshape(-1, 1) if proba.shape[1] > 1 else proba for proba in self.Y_pred_proba_900])

        self.Y_pred_full_900 = np.zeros((self.Y_pred_900.shape[0], len(self.Y.columns)))
        self.Y_pred_full_900[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.Y_pred_900

        self.Y_proba_full_900 = np.zeros((len(test900_ids), len(self.Y.columns)))  # 문서 개수와 라벨 개수에 맞춰 초기화
        self.Y_proba_full_900[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.y_proba_matrix_900

    def load_titles(self, title_file):
        title_dict = {}
        with open(title_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '-' in line:  # '-'이 포함된 줄만 처리
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

    def save_results(self):
        lable_res_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-RandomForest',
            'Labels': [self.predict_label(row, self.Y.columns) for row in self.Y_pred_full]
        })
        lable_res_df = pd.concat([lable_res_df, self.ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'],
                                                                                             ascending=[True, True]).reset_index(
            drop=True)
        lable_res_path = f"{self.OUTPUT_DIR}/rf_predicted_labels.csv"
        lable_res_df.to_csv(lable_res_path, index=False, encoding='utf-8-sig')

        print(f'각 문서의 라벨 예측 결과 저장 경로: {lable_res_path}')

        lable_res_with_prob_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-RandomForest',
            'Labels': [self.predict_label_with_proba(row, proba_row, self.Y.columns)
                       for row, proba_row in zip(self.Y_pred_full, self.Y_proba_full)]
        })

        label_res_with_prob_df = pd.concat([lable_res_with_prob_df, self.ground_truth_df], axis=0).sort_values(
            by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
        lable_res_with_prob_path = f"{self.OUTPUT_DIR}/rf_predicted_labels_with_prob.csv"
        label_res_with_prob_df.to_csv(lable_res_with_prob_path, index=False, encoding='utf-8-sig')

        print(f'각 문서의 라벨 및 확률 예측 결과 저장 경로: {lable_res_with_prob_path}')

        test_document_ids = self.X.loc[((self.X['Document ID'] >= 138) & (self.X['Document ID'] <= 158)), 'Document ID'].values
        total_df = pd.DataFrame({
            'DB Key': self.test_db_key,
            'Model': 'Top2Vec-RandomForest',
            'Labels': [self.predict_all_labels_with_proba(proba_row, self.Y.columns)
                       for proba_row in self.Y_proba_full]
        })
        total_df = pd.concat([total_df, self.ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'],
                                                                                     ascending=[True, True]).reset_index(
            drop=True)
        all_labels_results_path = f"{self.OUTPUT_DIR}/rf_predicted_all_labels.csv"
        total_df.to_csv(all_labels_results_path, index=False, encoding='utf-8-sig')

        print(f'각 문서에 대한 모든 라벨의 확률값 결과의 경로: {all_labels_results_path}')

        test900_ids = self.test900_df['Document ID'].values
        self.title_dict = self.load_titles(self.TITLE900_PATH)

        label_res_with_all_df_900 = pd.DataFrame({
            'DB Key': test900_ids,
            'Title': [self.title_dict.get(str(db_key), 'Unknown') for db_key in test900_ids],
            'Model': 'Top2Vec-RandomForest',
            'Labels': [self.predict_all_labels_with_proba(proba_row, self.Y.columns)
                       for proba_row in self.Y_proba_full_900]
        })

        label_res_with_all_df_900_path = f"{self.OUTPUT_DIR}/test_900_all_labels.csv"
        label_res_with_all_df_900.to_csv(label_res_with_all_df_900_path, index=False, encoding='utf-8-sig')

        print(f'900개 테스트 데이터 모든 라벨 확률 예측 결과 저장 경로: {label_res_with_all_df_900_path}')

        label_res_with_prob_df_900 = pd.DataFrame({
            'DB Key': test900_ids,
            'Title': [self.title_dict.get(str(db_key), 'Unknown') for db_key in test900_ids],
            'Model': 'Top2Vec-RandomForest',
            'Labels': [self.predict_label_with_proba(self.Y_pred_full_900[i], self.Y_proba_full_900[i], self.Y.columns)
                       for i in range(len(test900_ids))]
        })
        label_res_with_prob_df_900_path = f"{self.OUTPUT_DIR}/test_900_predictions_with_prob.csv"
        label_res_with_prob_df_900.to_csv(label_res_with_prob_df_900_path, index=False, encoding='utf-8-sig')

        print(f'900개 테스트 데이터 예측 결과 저장 경로: {label_res_with_prob_df_900_path}')

    def run(self):
        self.load_data()
        self.train_model()
        self.calculate_optimal_thresholds()
        self.calculate_performance_metrics()
        self.calculate_auc()
        self.prepare_predictions()
        self.calculate_hits()
        self.load_test900_data()
        self.save_results()

if __name__ == "__main__":
    TOPIC_SIZE = 'major'
    X_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_163.csv'
    Y_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_{TOPIC_SIZE}.csv'
    TEST900_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_900.csv'
    GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/{TOPIC_SIZE}_GT.csv'
    OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/RandomForest/{TOPIC_SIZE}'
    TITLE900_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/title_900.txt'
    test_db_key = [453073, 453074, 453075, 453076, 453077, 453078, 453079, 453082, 453083, 453084, 453093,
                    453095, 453096, 453097, 452970, 453102, 453104, 453105, 453110, 453114, 453116]

    evaluator = Top2VecRandomForest(
        TOPIC_SIZE, X_PATH, Y_PATH, TEST900_PATH, GROUND_TRUTH, OUTPUT_DIR, TITLE900_PATH, test_db_key
    )
    evaluator.run()
