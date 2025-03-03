import numpy as np
import pandas as pd
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score


class Top2VecLogisticRegression:
    def __init__(self, topic_size, X_path, Y_path, TITLE_PATH,  GROUND_TRUTH, OUTPUT_DIR):
        self.topic_size = topic_size
        self.X_path = X_path
        self.Y_path = Y_path
        self.TITLE_PATH = TITLE_PATH
        self.GROUND_TRUTH = GROUND_TRUTH
        self.OUTPUT_DIR = OUTPUT_DIR

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

        self.title_df = pd.read_csv(TITLE_PATH, header=0)
        self.test_db_key = self.X['Document ID'].values
        self.title = self.title_df['연구보고서'].values

        self.X['Embedding Vector'] = self.X['Embedding Vector'].astype(str).apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )

        self.X_train = self.X.iloc[:730].copy()
        self.Y_train_df = self.Y.iloc[:730].copy()
        self.X_test = self.X.iloc[730:].copy() 
        self.Y_test_df = self.Y.iloc[730:].copy()

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
        base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
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

    def adjust_probabilities_based_on_title(self):
        major_minor_mapping = {
            '가족': ['가족정책', '돌봄', '저출산', '일생활균형_가족'],
            '건강': ['건강'],
            '교육': ['교육정책', '직업교육', '성평등교육', '평생교육'],
            '국제협력': ['국제협력', 'ODA'],
            '노동': ['고용', '경력단절', '일생활균형_노동', '노동정책', '인적자원개발'],
            '법제도': ['법제도'],
            '사회통합': ['남성', '다문화', '미래', '이주민', '취약계층', '통일'],
            '성주류화': ['성주류화정책', '성별영향평가', '성인지예산', '성인지통계'],
            '성평등사회': ['성평등문화', '성평등의식', '성평등정책'],
            '안전': ['안전'],
            '젠더폭력': ['젠더폭력', '인권'],
            '대표성': ['공공대표성', '민간대표성', '정치대표성']
        }

        minor_keywords = [item for sublist in major_minor_mapping.values() for item in sublist]
        major_keywords = list(major_minor_mapping.keys())

        for i, title in enumerate(self.title[730:]):  
            title = str(title)  

            for minor_keyword in minor_keywords:
                if minor_keyword in title:  
                    if minor_keyword in self.Y_train_filtered.columns:
                        col_index = self.Y_train_filtered.columns.get_loc(minor_keyword)
                        prev_prob = self.Y_pred_proba[col_index][i, 1]
                        self.Y_pred_proba[col_index][i, 1] += 0.3
                        # print(f"[Doc {i+730}] '{title}' ▶ 중분류 '{minor_keyword}' 확률 업데이트: {prev_prob:.4f} → 1.0000")

            for major_keyword in major_keywords:
                if major_keyword in title:  
                    for minor_keyword in major_minor_mapping[major_keyword]:
                        if minor_keyword in self.Y_train_filtered.columns:
                            col_index = self.Y_train_filtered.columns.get_loc(minor_keyword)
                            prev_prob = self.Y_pred_proba[col_index][i, 1]
                            self.Y_pred_proba[col_index][i, 1] += 0.3
                            # print(f"[Doc {i+730}] '{title}' ▶ 대분류 '{major_keyword}' → 중분류 '{minor_keyword}' 확률 증가: {prev_prob:.4f} → {self.Y_pred_proba[col_index][i, 1]:.4f}")

        self.Y_pred = np.array([
        ((proba[:, 1] >= self.optimal_thresholds[i]) if proba.shape[1] > 1 
         else (proba[:, 0] >= self.optimal_thresholds[i])).astype(int)
        for i, proba in enumerate(self.Y_pred_proba)
        ]).T



        

    def calculate_hit_at_k(self, y_true, y_proba, k):
            hits = 0
            for true, proba in zip(y_true, y_proba):
                top_k_indices = np.argsort(proba)[-k:][::-1]
                if any(true[i] == 1 for i in top_k_indices):
                    hits += 1
            return hits / len(y_true)

    def calculate_performance_metrics(self):
        print(self.Y_pred[0])
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
        print('========[제목 고려 LG 중분류 성능]=======')
        print('------------------[F1 score]-------------------')
        print(f"Micro F1 Score: {f1_micro:.4f}")
        print(f"Macro F1 Score: {f1_macro:.4f}")
        print(f"Weighted F1 Score: {f1_weighted:.4f}")
        print('--------------[Precision/Recall]---------------')
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
        
        # AUC
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
        print('-------------------[AUC]--------------------')
        print(f"Average AUC: {average_auc:.4f}")

        hit_1 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 1)
        hit_3 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 3)
        hit_5 = self.calculate_hit_at_k(self.Y_test, self.y_proba_matrix, 5)
        print('------------------[hit@k]-------------------')
        print(f"Hit@1: {hit_1:.4f}")
        print(f"Hit@3: {hit_3:.4f}")
        print(f"Hit@5: {hit_5:.4f}")

        optimal_thresholds_df.to_csv(f"{self.OUTPUT_DIR}/optimal_thresholds_lr.csv", index=False)
        print(f"\nOptimal thresholds saved to '{self.OUTPUT_DIR}/optimal_thresholds_lr.csv'.")
        

    def prepare_predictions(self):
        self.y_proba_matrix = np.hstack([proba[:, -1].reshape(-1, 1) if proba.shape[1] > 1 else proba for proba in self.Y_pred_proba])

        self.Y_pred_full = np.zeros((self.Y_pred.shape[0], len(self.Y.columns)))
        self.Y_pred_full[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.Y_pred
        self.Y_proba_full = np.zeros((self.Y_pred.shape[0], len(self.Y.columns)))
        self.Y_proba_full[:, [self.Y.columns.get_loc(col) for col in self.Y_train_filtered.columns]] = self.y_proba_matrix

    
    def predict_label(self, row, column_names):
        return ', '.join(column_names[row == 1])

    def predict_label_with_proba(self, row, proba_row, column_names):
        labels_with_proba = [
            (column_names[i], proba_row[i])
            for i in range(len(row)) if row[i] == 1
        ]
        labels_with_proba.sort(key=lambda x: x[1], reverse=True)
        return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])


    def predict_all_labels_with_proba(self, proba_row, column_names):
        labels_with_proba = list(zip(column_names, proba_row)) 
        labels_with_proba.sort(key=lambda x: x[1], reverse=True) 
        return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba]) 

    def save_results(self):
        # ----- 소분류(predicted labels) 결과 파일 생성 -----
        # 예측 결과 DataFrame (Label 컬럼은 빈 문자열로 채움)
        pred_minor_df = pd.DataFrame({
            'DB Key': self.test_db_key[730:],
            'Title': self.title[730:],
            'Model': 'Top2Vec-LogisticRegression',
            'Labels': [self.predict_label(row, self.Y.columns) for row in self.Y_pred_full],
        })
        # GT DataFrame (파일에서 로드한 순서를 그대로 사용)
        gt_minor_df = self.ground_truth_df.copy()
        gt_minor_df = gt_minor_df[730:]
        gt_minor_df.rename(columns={'Label': 'Labels'}, inplace=True)
        gt_minor_df.insert(1, 'Title', self.title[730:])

        # 각 DB Key에 대해 예측 결과 행 다음에 GT 행이 나오도록 교차 결합
        interleaved_minor = []
        for i in range(len(self.title[730:])):
            interleaved_minor.append(pred_minor_df.iloc[i])
            interleaved_minor.append(gt_minor_df.iloc[i])
        combined_minor_df = pd.DataFrame(interleaved_minor)
        
        lable_res_path = f"{self.OUTPUT_DIR}/lr_predicted_labels.csv"
        combined_minor_df.to_csv(lable_res_path, index=False, encoding='utf-8-sig')
        print(f'각 문서의 소분류 라벨 예측 결과 저장 경로: {lable_res_path}')
        
        
        # ----- 소분류(predicted labels with probability) 결과 파일 생성 -----
        pred_minor_prob_df = pd.DataFrame({
            'DB Key': self.test_db_key[730:],
            'Title': self.title[730:],
            'Model': 'Top2Vec-LogisticRegression',
            'Labels': [self.predict_label_with_proba(row, proba_row, self.Y.columns)
                    for row, proba_row in zip(self.Y_pred_full, self.Y_proba_full)],
        })
        # 동일한 GT DataFrame 사용
        interleaved_minor_prob = []
        for i in range(len(self.title[730:])):
            interleaved_minor_prob.append(pred_minor_prob_df.iloc[i])
            interleaved_minor_prob.append(gt_minor_df.iloc[i])
        combined_minor_prob_df = pd.DataFrame(interleaved_minor_prob)
        
        lable_res_with_prob_path = f"{self.OUTPUT_DIR}/lr_predicted_labels_with_prob.csv"
        combined_minor_prob_df.to_csv(lable_res_with_prob_path, index=False, encoding='utf-8-sig')
        print(f'각 문서의 라벨 및 확률 예측 결과 저장 경로: {lable_res_with_prob_path}')
        
        
        # ----- 소분류(all labels with probability) 결과 파일 생성 -----
        pred_minor_all_df = pd.DataFrame({
            'DB Key': self.test_db_key[730:],
            'Title': self.title[730:],
            'Model': 'Top2Vec-LogisticRegression',
            'Labels': [self.predict_all_labels_with_proba(proba_row, self.Y.columns)
                    for proba_row in self.Y_proba_full],
        })
        interleaved_minor_all = []
        for i in range(len(self.title[730:])):
            interleaved_minor_all.append(pred_minor_all_df.iloc[i])
            interleaved_minor_all.append(gt_minor_df.iloc[i])
        combined_minor_all_df = pd.DataFrame(interleaved_minor_all)
        
        all_labels_results_path = f"{self.OUTPUT_DIR}/lr_predicted_all_labels.csv"
        combined_minor_all_df.to_csv(all_labels_results_path, index=False, encoding='utf-8-sig')
        print(f'각 문서에 대한 모든 라벨의 확률값 결과의 경로: {all_labels_results_path}')
        
        


    def run(self):
        self.load_data()
        self.train_model()
        self.calculate_optimal_thresholds()
        self.adjust_probabilities_based_on_title() 
        self.prepare_predictions()  
        self.calculate_performance_metrics()
        self.save_results()

if __name__ == "__main__":
    TOPIC_SIZE = 'minor'
    X_PATH = '/home/women/doyoung/Top2Vec/embedding/output/gpt_document_embeddings_900.csv'
    Y_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_gpt_{TOPIC_SIZE}.csv'
    TITLE_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/gpt_gt.csv'
    GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/gpt_{TOPIC_SIZE}_GT.csv'
    OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/LogisticRegression/title/{TOPIC_SIZE}'
    
    evaluator = Top2VecLogisticRegression(
        TOPIC_SIZE, X_PATH, Y_PATH, TITLE_PATH, GROUND_TRUTH, OUTPUT_DIR
    )
    evaluator.run()



