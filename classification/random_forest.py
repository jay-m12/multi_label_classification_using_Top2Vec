import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

X = pd.read_csv('doyoung/TOP2VEC/data/full/test_1000_new/ori_160/document_embeddings.csv', header=0)
Y = pd.read_csv('/home/women/doyoung/TOP2VEC/data/full/Y_df.csv', header=0)

X['Embedding Vector'] = X['Embedding Vector'].astype(str).apply(
    lambda x: np.array(list(map(float, x.strip('[]').split(','))))
)

test_mask = (X['Document ID'] >= 138) & (X['Document ID'] <= 162)
X_test = X[test_mask].copy()
Y_test = Y[test_mask].copy()

X_test = X_test.drop(columns=['Document ID'])

valid_columns = [col for col in Y_test.columns if len(np.unique(Y_test[col].values)) > 1]

common_columns = list(set(valid_columns) & set(X_test.columns))
missing_columns = list(set(valid_columns) - set(X_test.columns))  # 누락된 컬럼 확인

Y_test = Y_test[common_columns]
X_test = X_test[common_columns]

if missing_columns:
    print(f"Warning: The following columns exist in `Y_test` but are missing in `X_test`: {missing_columns}")

optimal_thresholds = []
y_pred = []

for class_name in common_columns:
    y_true = Y_test[class_name].values
    y_score = X_test[class_name].values

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]

    optimal_thresholds.append(optimal_threshold)
    y_pred.append((y_score >= optimal_threshold).astype(int))

y_pred = np.array(y_pred).T
y_true = Y_test.values

f1_micro = f1_score(Y_test, y_pred, average="micro")
f1_macro = f1_score(Y_test, y_pred, average="macro")
f1_weighted = f1_score(Y_test, y_pred, average="weighted")

print(f"Micro F1 Score: {f1_micro:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

auc_scores = []
for i in range(Y_test.shape[1]):
    if np.sum(Y_test[:, i]) == 0:
        auc_scores.append(None)
        continue
    
    class_prob = y_pred_proba[i][:, 1] if y_pred_proba[i].shape[1] > 1 else y_pred_proba[i][:, 0]
    
    try:
        auc = roc_auc_score(Y_test[:, i], class_prob)
        auc_scores.append(auc)
    except ValueError:
        auc_scores.append(None)

valid_auc_scores = [score for score in auc_scores if score is not None]
average_auc = np.mean(valid_auc_scores)
print(f"Average AUC: {average_auc:.4f}")

def hit_at_k(y_true, y_proba, k):
    hits = 0
    for true, proba in zip(y_true, y_proba):
        top_k_indices = np.argsort(proba)[-k:][::-1]
        if any(true[i] == 1 for i in top_k_indices):
            hits += 1
    return hits / len(y_true)

y_proba_matrix = np.hstack([proba[:, -1].reshape(-1,1) if proba.shape[1] > 1 else proba for proba in y_pred_proba])

hit_1 = hit_at_k(Y_test, y_proba_matrix, 1)
hit_3 = hit_at_k(Y_test, y_proba_matrix, 3)
hit_5 = hit_at_k(Y_test, y_proba_matrix, 5)

print(f"Hit@1: {hit_1:.4f}")
print(f"Hit@3: {hit_3:.4f}")
print(f"Hit@5: {hit_5:.4f}")
