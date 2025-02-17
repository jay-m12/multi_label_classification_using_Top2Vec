import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

TOPIC_SIZE = 'major'
X = pd.read_csv('/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_163.csv', header=0)
Y = pd.read_csv(f'/home/women/doyoung/Top2Vec/preprocessing/output/Y_{TOPIC_SIZE}.csv', header=0)
TEST900_PATH = '/home/women/doyoung/Top2Vec/embedding/output/document_embeddings_900.csv'
GROUND_TRUTH = f'/home/women/doyoung/Top2Vec/preprocessing/output/{TOPIC_SIZE}_GT.csv'
OUTPUT_DIR = f'/home/women/doyoung/Top2Vec/classification/output/LogisticRegression/{TOPIC_SIZE}'
TITLE900_PATH = '/home/women/doyoung/Top2Vec/preprocessing/input/title_900.txt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GT
ground_truth_df = pd.read_csv(GROUND_TRUTH, encoding='utf-8-sig')

X['Embedding Vector'] = X['Embedding Vector'].astype(str).apply(
    lambda x: np.array(list(map(float, x.strip('[]').split(','))))
)

test_db_key = [453073, 453074, 453075, 453076, 453077, 453078, 453079, 453082, 453083, 453084, 453093,
                453095, 453096, 453097, 452970, 453102, 453104, 453105, 453110, 453114, 453116]

test_mask = (X['Document ID'] >= 138) & (X['Document ID'] <= 158)
X_test = X[test_mask].copy()
Y_test_df = Y.loc[test_mask].copy()
X_train = X[~test_mask].copy()
Y_train_df = Y.loc[~test_mask].copy()

# 단일 클래스 컬럼 식별 및 제거
single_class_cols = [col for col in Y.columns if Y_train_df[col].nunique() == 1]
Y_train_filtered = Y_train_df.drop(columns=single_class_cols)
Y_test_filtered = Y_test_df.drop(columns=single_class_cols)

print(f"제거된 컬럼 수: {len(single_class_cols)}/{Y.shape[1]}")
print("제거된 컬럼 목록:", single_class_cols)

test_document_ids = X_test['Document ID'].values

X_train = X_train.drop(columns=['Document ID'])
X_test = X_test.drop(columns=['Document ID'])

X_train = np.stack(X_train['Embedding Vector'].values)
X_test = np.stack(X_test['Embedding Vector'].values)

Y_train = Y_train_filtered.values
Y_test = Y_test_filtered.values

# 모델 학습
base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
model = MultiOutputClassifier(base_model)
model.fit(X_train, Y_train)

Y_pred_proba = model.predict_proba(X_test)

optimal_thresholds = []

for i in range(Y_test.shape[1]):
    if np.sum(Y_test[:, i]) == 0:
        optimal_thresholds.append(0.3)
        continue
    
    n_classes = Y_pred_proba[i].shape[1]
    
    if n_classes >= 2:
        class_prob = Y_pred_proba[i][:, 1]
    else:
        class_prob = Y_pred_proba[i][:, 0]
    
    fpr, tpr, thresholds = roc_curve(Y_test[:, i], class_prob)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_thresholds.append(optimal_threshold)

# 최적 임계값을 적용하여 예측
Y_pred = np.array([
    ((proba[:, 1] >= optimal_thresholds[i]) if proba.shape[1] > 1 
     else (proba[:, 0] >= optimal_thresholds[i])).astype(int)
    for i, proba in enumerate(Y_pred_proba)
]).T

# F1 점수 계산 (최적 임계값 기반)
f1_micro = f1_score(Y_test, Y_pred, average="micro")
f1_macro = f1_score(Y_test, Y_pred, average="macro")
f1_weighted = f1_score(Y_test, Y_pred, average="weighted")

print(f"Micro F1 Score (Optimal Threshold): {f1_micro:.4f}")
print(f"Macro F1 Score (Optimal Threshold): {f1_macro:.4f}")
print(f"Weighted F1 Score (Optimal Threshold): {f1_weighted:.4f}")

optimal_thresholds_df = pd.DataFrame({
    "class_name": Y_train_filtered.columns.tolist(), 
    "optimal_threshold": optimal_thresholds
})
optimal_thresholds_df.to_csv(f"{OUTPUT_DIR}/optimal_thresholds_lr.csv", index=False)
print("\nOptimal thresholds saved to '{OUTPUT_DIR}/optimal_thresholds_lr.csv'.")

# AUC 계산
auc_scores = []
for i in range(Y_test.shape[1]):
    if np.sum(Y_test[:, i]) == 0:
        auc_scores.append(None)
        continue
    
    n_classes = Y_pred_proba[i].shape[1]
    class_prob = Y_pred_proba[i][:, 1] if n_classes >= 2 else Y_pred_proba[i][:, 0]
    
    try:
        auc = roc_auc_score(Y_test[:, i], class_prob)
        auc_scores.append(auc)
    except ValueError:
        auc_scores.append(None)

valid_auc_scores = [score for score in auc_scores if score is not None]
average_auc = np.mean(valid_auc_scores)
print(f"Average AUC: {average_auc:.4f}")

y_proba_matrix = np.hstack([proba[:, -1].reshape(-1,1) if proba.shape[1] > 1 else proba for proba in Y_pred_proba])

Y_pred_full = np.zeros((Y_pred.shape[0], len(Y.columns)))
Y_pred_full[:, [Y.columns.get_loc(col) for col in Y_train_filtered.columns]] = Y_pred
Y_proba_full = np.zeros((Y_pred.shape[0], len(Y.columns)))
Y_proba_full[:, [Y.columns.get_loc(col) for col in Y_train_filtered.columns]] = y_proba_matrix

def hit_at_k(y_true, y_proba, k):
    hits = 0
    for true, proba in zip(y_true, y_proba):
        top_k_indices = np.argsort(proba)[-k:][::-1]
        if any(true[i] == 1 for i in top_k_indices):
            hits += 1
    return hits / len(y_true)

# hit@k 계산
hit_1 = hit_at_k(Y_test, y_proba_matrix, 1)
hit_3 = hit_at_k(Y_test, y_proba_matrix, 3)
hit_5 = hit_at_k(Y_test, y_proba_matrix, 5)

print(f"Hit@1: {hit_1:.4f}")
print(f"Hit@3: {hit_3:.4f}")
print(f"Hit@5: {hit_5:.4f}")




# 예측 라벨 출력(확률값 제외) [1]==============================
def predict_label(row, column_names):
    return ', '.join(column_names[row == 1])
lable_res_df = pd.DataFrame({
    'DB Key': test_db_key,
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_label(row, Y.columns) for row in Y_pred_full]
})
lable_res_df = pd.concat([lable_res_df, ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
lable_res_path = f"{OUTPUT_DIR}/lr_predicted_labels.csv"
lable_res_df.to_csv(lable_res_path, index=False, encoding = 'utf-8-sig')

print(f'각 문서의 라벨 예측 결과 저장 경로: {lable_res_path}')




# 예측된 라벨과 확률을 함께 출력 (확률 내림차순 정렬 포함) [2] ==============================
def predict_label_with_proba(row, proba_row, column_names):
    labels_with_proba = [
        (column_names[i], proba_row[i])
        for i in range(len(row)) if row[i] == 1
    ]
    labels_with_proba.sort(key=lambda x: x[1], reverse=True)
    return ','.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])
lable_res_with_prob_df = pd.DataFrame({
    'DB Key': test_db_key,  
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_label_with_proba(row, proba_row, Y.columns) 
                   for row, proba_row in zip(Y_pred_full, Y_proba_full)]
})
label_res_with_prob_df = pd.concat([lable_res_with_prob_df, ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
lable_res_with_prob_path = f"{OUTPUT_DIR}/lr_predicted_labels_with_prob.csv"
label_res_with_prob_df.to_csv(lable_res_with_prob_path, index=False, encoding='utf-8-sig')

print(f'각 문서의 라벨 및 확률 예측 결과 저장 경로: {lable_res_with_prob_path}')



# 전체 라벨 확률값 출력 [3]==============================
def predict_all_labels_with_proba(proba_row, column_names):
    labels_with_proba = [
        (column_names[i], proba_row[i])
        for i in range(len(proba_row))
    ]
    labels_with_proba.sort(key=lambda x: x[1], reverse=True)
    return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])
test_document_ids = X.loc[test_mask, 'Document ID'].values  
total_df = pd.DataFrame({
    'DB Key': test_db_key,
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_all_labels_with_proba(proba_row, Y.columns)
               for proba_row in Y_proba_full]
})
total_df = pd.concat([total_df, ground_truth_df], axis=0).sort_values(by=['DB Key', 'Model'], ascending=[True, True]).reset_index(drop=True)
all_labels_results_path = f"{OUTPUT_DIR}/lr_predicted_all_labels.csv"
total_df.to_csv(all_labels_results_path, index=False, encoding='utf-8-sig')

print(f'각 문서에 대한 모든 라벨의 확률값 결과의 경로: {all_labels_results_path}')


# 900개 전체 문서에 대한 예측된 라벨 및 확률값 추출하기 =========================================
test900_df = pd.read_csv(TEST900_PATH, header = 0)

test900_df['Embedding Vector'] = test900_df['Embedding Vector'].astype(str).apply(
    lambda x: np.array(list(map(float, x.strip('[]').split(','))))
)
test900_ids = test900_df['Document ID'].values
X_test_900 = np.stack(test900_df.drop(columns = ['Document ID'])['Embedding Vector'].values)

Y_pred_proba_900 = model.predict_proba(X_test_900)
Y_pred_900 = np.array([
    ((proba[:,1] >= optimal_thresholds[i]) if proba.shape[1] > 1
     else(proba[:,0] >= optimal_thresholds[i])).astype(int)
     for i, proba in enumerate(Y_pred_proba_900)
]).T

y_proba_matrix_900 = np.hstack([proba[:, -1].reshape(-1,1) if proba.shape[1] > 1 else proba for proba in Y_pred_proba_900])
Y_pred_full_900 = np.zeros((Y_pred_900.shape[0], len(Y.columns)))
Y_pred_full_900[:, [Y.columns.get_loc(col) for col in Y_train_filtered.columns]] = Y_pred_900
Y_proba_full_900 = np.zeros((Y_pred_900.shape[0], len(Y.columns)))
Y_proba_full_900[:, [Y.columns.get_loc(col) for col in Y_train_filtered.columns]] = y_proba_matrix_900

def predict_label_with_proba(row, proba_row, column_names):
    labels_with_proba = [
        (column_names[i], proba_row[i])
        for i in range(len(row)) if row[i] == 1
    ]
    labels_with_proba.sort(key=lambda x: x[1], reverse=True)
    return ', '.join([f"{label}-{proba:.3f}" for label, proba in labels_with_proba])

def load_titles(title_file):
    title_dict = {}
    with open(title_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '-' in line:
                db_key, title = line.split(' - ', 1) 
                title_dict[db_key] = title
    return title_dict

label_res_with_prob_df_900 = pd.DataFrame({
    'DB Key': test900_ids,  
    'Model': 'Top2Vec-LogisticRegression',
    'Labels': [predict_label_with_proba(row, proba_row, Y.columns) 
               for row, proba_row in zip(Y_pred_full_900, Y_proba_full_900)]
})
label_res_with_prob_df_900_path = f"{OUTPUT_DIR}/test_900_predictions_with_prob.csv"
label_res_with_prob_df_900.to_csv(label_res_with_prob_df_900_path, index=False, encoding='utf-8-sig')

print(f'900개 테스트 데이터 예측 결과 저장 경로: {label_res_with_prob_df_900_path}')