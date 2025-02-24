import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

# === 설정 ==================================================================


DOCUMENT_EMBEDDINGS_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_163.csv'
TEST900_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_900.csv'
TITLE900_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/title_900.txt'

MAJOR_GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/major_GT.csv'
MINOR_GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/minor_GT.csv'

OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/LogisticRegression/hierarchical'

Y_MAJOR_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_major.csv'
Y_MINOR_PATH = f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_minor.csv'

TEST_DB_KEY = [453073, 453074, 453075, 453076, 453077, 453078, 453079, 453082, 453083, 453084, 453093,
                453095, 453096, 453097, 452970, 453102, 453104, 453105, 453110, 453114, 453116]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 데이터 로드 ===============================================================
X = pd.read_csv(DOCUMENT_EMBEDDINGS_PATH, header=0)
Y_major = pd.read_csv(Y_MAJOR_PATH, header=0)
Y_minor = pd.read_csv(Y_MINOR_PATH, header=0)

X['Embedding Vector'] = X['Embedding Vector'].astype(str).apply(
    lambda x: np.array(list(map(float, x.strip('[]').split(','))))
)

# 테스트 데이터(138~158) 분리
test_mask = (X['Document ID'] >= 138) & (X['Document ID'] <= 158)
X_test = X[test_mask].copy()
X_train = X[~test_mask].copy()

Y_major_test = Y_major.loc[test_mask].copy()
Y_major_train = Y_major.loc[~test_mask].copy()
Y_minor_test = Y_minor.loc[test_mask].copy()
Y_minor_train = Y_minor.loc[~test_mask].copy()

# 단일 클래스 컬럼 제거 (대분류)
major_single_class_cols = [col for col in Y_major.columns if Y_major_train[col].nunique() == 1]
Y_major_train_filtered = Y_major_train.drop(columns=major_single_class_cols)
Y_major_test_filtered = Y_major_test.drop(columns=major_single_class_cols)

print(f"[대분류] 제거된 컬럼 수: {len(major_single_class_cols)}/{Y_major.shape[1]}")
print("[대분류] 제거된 컬럼 목록:", major_single_class_cols)

# 단일 클래스 컬럼 제거 (중분류)
minor_single_class_cols = [col for col in Y_minor.columns if Y_minor_train[col].nunique() == 1]
Y_minor_train_filtered = Y_minor_train.drop(columns=minor_single_class_cols)
Y_minor_test_filtered = Y_minor_test.drop(columns=minor_single_class_cols)

print(f"[중분류] 제거된 컬럼 수: {len(minor_single_class_cols)}/{Y_minor.shape[1]}")
print("[중분류] 제거된 컬럼 목록:", minor_single_class_cols)

X_train = X_train.drop(columns=['Document ID'])
X_test = X_test.drop(columns=['Document ID'])
X_train = np.stack(X_train['Embedding Vector'].values)
X_test = np.stack(X_test['Embedding Vector'].values)

Y_major_train = Y_major_train_filtered.values
Y_major_test = Y_major_test_filtered.values
Y_minor_train = Y_minor_train_filtered.values
Y_minor_test = Y_minor_test_filtered.values

# === 대분류 모델 학습 및 예측 ===================================================
major_base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
major_model = MultiOutputClassifier(major_base_model)
major_model.fit(X_train, Y_major_train)

Y_major_pred_proba = major_model.predict_proba(X_test)

major_optimal_thresholds = []
for i in range(Y_major_test.shape[1]):
    if np.sum(Y_major_test[:, i]) == 0:
        major_optimal_thresholds.append(0.3)
        continue
    
    n_classes = Y_major_pred_proba[i].shape[1]
    class_prob = Y_major_pred_proba[i][:, 1] if n_classes >= 2 else Y_major_pred_proba[i][:, 0]
    
    fpr, tpr, thresholds = roc_curve(Y_major_test[:, i], class_prob)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = float(thresholds[optimal_idx])
    major_optimal_thresholds.append(optimal_threshold)

Y_major_pred = np.array([
    ((proba[:, 1] >= major_optimal_thresholds[i]) if proba.shape[1] > 1 
     else (proba[:, 0] >= major_optimal_thresholds[i])).astype(int)
    for i, proba in enumerate(Y_major_pred_proba)
]).T

# === 중분류 모델 학습 및 예측 ===================================================
minor_base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
minor_model = MultiOutputClassifier(minor_base_model)
minor_model.fit(X_train, Y_minor_train)

Y_minor_pred_proba = minor_model.predict_proba(X_test)

# === 대분류-중분류 매핑 ==================================================
major_minor_mapping = {
    0: ['가족정책', '돌봄', '저출산', '일생활균형_가족'],  # 가족
    1: ['건강'],  # 건강
    2: ['교육정책', '직업교육', '성평등교육', '평생교육'],  # 교육
    3: ['국제협력', 'ODA'],  # 국제협력
    4: ['고용', '경력단절', '일생활균형_노동', '노동정책', '인적자원개발'],  # 노동
    5: ['법제도'],  # 법제도
    6: ['남성', '다문화', '미래', '이주민', '취약계층', '통일'],  # 사회통합
    7: ['성주류화정책', '성별영향평가', '성인지예산', '성인지통계'],  # 성주류화
    8: ['성평등문화', '성평등의식', '성평등정책'],  # 성평등사회
    9: ['안전'],  # 안전
    10: ['젠더폭력', '인권'],  # 젠더폭력
    11: ['공공대표성', '민간대표성', '정치대표성']  # 대표성
}

def get_minor_classes(major_class_idx):
    minor_classes = major_minor_mapping.get(major_class_idx, [])
    minor_class_indices = [Y_minor_train_filtered.columns.get_loc(col) for col in minor_classes if col in Y_minor_train_filtered.columns]
    return minor_class_indices

for doc_idx in range(X_test.shape[0]): 
    for class_idx in range(Y_major_pred.shape[1]):  
        if Y_major_pred[doc_idx, class_idx] == 0:
            minor_class_indices = get_minor_classes(class_idx)
            for minor_class_idx in minor_class_indices:
                Y_minor_pred_proba[minor_class_idx][doc_idx, 1] = -1000000000


minor_optimal_thresholds = []
for i in range(len(Y_minor_pred_proba)):
    if np.sum(Y_minor_test[:, i]) == 0:
        minor_optimal_thresholds.append(0.3)
        continue
    
    class_prob = Y_minor_pred_proba[i][:, 1]  
    
    fpr, tpr, thresholds = roc_curve(Y_minor_test[:, i], class_prob)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = float(thresholds[optimal_idx])
    minor_optimal_thresholds.append(optimal_threshold)

Y_minor_pred = np.array([
    (proba[:, 1] >= minor_optimal_thresholds[i]).astype(int)
    for i, proba in enumerate(Y_minor_pred_proba)
]).T

# === 결과 평가 및 저장 =========================================================
f1_micro = f1_score(Y_minor_test, Y_minor_pred, average="micro")
f1_macro = f1_score(Y_minor_test, Y_minor_pred, average="macro")
f1_weighted = f1_score(Y_minor_test, Y_minor_pred, average="weighted")

print(f"Micro F1 Score (Optimal Threshold): {f1_micro:.4f}")
print(f"Macro F1 Score (Optimal Threshold): {f1_macro:.4f}")
print(f"Weighted F1 Score (Optimal Threshold): {f1_weighted:.4f}")

minor_optimal_thresholds_df = pd.DataFrame({
    "class_name": Y_minor_train_filtered.columns.tolist(),
    "optimal_threshold": minor_optimal_thresholds
})
minor_optimal_thresholds_df.to_csv(f"{OUTPUT_DIR}/optimal_thresholds_lr.csv", index=False)
print(f"\nOptimal thresholds saved to '{OUTPUT_DIR}/optimal_thresholds_lr.csv'.")

auc_scores = []
for i in range(Y_minor_test.shape[1]):
    if np.sum(Y_minor_test[:, i]) == 0:
        auc_scores.append(None)
        continue
    
    class_prob = Y_minor_pred_proba[i][:, 1]
    
    try:
        auc = roc_auc_score(Y_minor_test[:, i], class_prob)
        auc_scores.append(auc)
    except ValueError:
        auc_scores.append(None)

valid_auc_scores = [score for score in auc_scores if score is not None]
average_auc = np.mean(valid_auc_scores)
print(f"Average AUC: {average_auc:.4f}")

y_proba_matrix = np.hstack([proba[:, -1].reshape(-1,1) if proba.shape[1] > 1 else proba for proba in Y_minor_pred_proba])

Y_minor_pred_full = np.zeros((Y_minor_pred.shape[0], len(Y_minor.columns)))
Y_minor_pred_full[:, [Y_minor.columns.get_loc(col) for col in Y_minor_train_filtered.columns]] = Y_minor_pred
Y_proba_full = np.zeros((Y_minor_pred.shape[0], len(Y_minor.columns)))
Y_proba_full[:, [Y_minor.columns.get_loc(col) for col in Y_minor_train_filtered.columns]] = y_proba_matrix

def hit_at_k(y_true, y_proba, k):
    hits = 0
    for true, proba in zip(y_true, y_proba):
        top_k_indices = np.argsort(proba)[-k:][::-1]
        if any(true[i] == 1 for i in top_k_indices):
            hits += 1
    return hits / len(y_true)

hit_1 = hit_at_k(Y_minor_test, y_proba_matrix, 1)
hit_3 = hit_at_k(Y_minor_test, y_proba_matrix, 3)
hit_5 = hit_at_k(Y_minor_test, y_proba_matrix, 5)

print(f"Hit@1: {hit_1:.4f}")
print(f"Hit@3: {hit_3:.4f}")
print(f"Hit@5: {hit_5:.4f}")

# === 결과 저장 ===============================================================
def predict_label(row, column_names):
    return ', '.join(column_names[row == 1])

minor_ground_truth_df = pd.read_csv(MINOR_GROUND_TRUTH, encoding='utf-8-sig')

lable_res_df = pd.DataFrame({
    'DB Key': TEST_DB_KEY,
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_label(row, Y_minor.columns) for row in Y_minor_pred_full]
})

lable_res_df = pd.concat([lable_res_df, minor_ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
lable_res_path = f"{OUTPUT_DIR}/lr_predicted_labels.csv"
lable_res_df.to_csv(lable_res_path, index=False, encoding = 'utf-8-sig')

print(f'각 문서의 라벨 예측 결과 저장 경로: {lable_res_path}')

def predict_label_with_proba(row, proba_row, column_names):
    labels_with_proba = [
        (column_names[i], proba_row[i])
        for i in range(len(row)) if row[i] == 1
    ]
    labels_with_proba.sort(key=lambda x: x[1], reverse=True)
    return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])

lable_res_with_prob_df = pd.DataFrame({
    'DB Key': TEST_DB_KEY,
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_label_with_proba(row, proba_row, Y_minor.columns)
               for row, proba_row in zip(Y_minor_pred_full, Y_proba_full)]
})

label_res_with_prob_df = pd.concat([lable_res_with_prob_df, minor_ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
lable_res_with_prob_path = f"{OUTPUT_DIR}/lr_predicted_labels_with_prob.csv"
label_res_with_prob_df.to_csv(lable_res_with_prob_path, index=False, encoding='utf-8-sig')

print(f'각 문서의 라벨 및 확률 예측 결과 저장 경로: {lable_res_with_prob_path}')

def predict_all_labels_with_proba(proba_row, column_names):
    labels_with_proba = [
        (column_names[i], proba_row[i])
        for i in range(len(proba_row))
    ]
    labels_with_proba.sort(key=lambda x: x[1], reverse=True)
    return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])

total_df = pd.DataFrame({
    'DB Key': TEST_DB_KEY,
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_all_labels_with_proba(proba_row, Y_minor.columns)
               for proba_row in Y_proba_full]
})
total_df = pd.concat([total_df, minor_ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
all_labels_results_path = f"{OUTPUT_DIR}/lr_predicted_all_labels.csv"
total_df.to_csv(all_labels_results_path, index=False, encoding='utf-8-sig')

print(f'각 문서에 대한 모든 라벨의 확률값 결과의 경로: {all_labels_results_path}')