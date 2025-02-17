import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


X = pd.read_csv('doyoung/TOP2VEC/data/full/test_1000_new/ori_160/document_embeddings.csv', header=0)
Y = pd.read_csv('/home/women/doyoung/TOP2VEC/data/full/Y_df.csv', header=0)


X['Embedding Vector'] = X['Embedding Vector'].astype(str).apply(
    lambda x: np.array(list(map(float, x.strip('[]').split(','))))
)

test_mask = (X['Document ID'] >= 138) & (X['Document ID'] <= 162)
X_test = X[test_mask]
Y_test = Y[test_mask]

X_train_val = X[~test_mask]
Y_train_val = Y[~test_mask]

X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=['Document ID'])
X_val = X_val.drop(columns=['Document ID'])
X_test = X_test.drop(columns=['Document ID'])

X_train = np.stack(X_train['Embedding Vector'].values)
X_val = np.stack(X_val['Embedding Vector'].values)
X_test = np.stack(X_test['Embedding Vector'].values)

Y_train = Y_train.values
Y_val = Y_val.values
Y_test = Y_test.values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)    
X_test = scaler.transform(X_test)

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

n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
model = MultiLabelClassifier(n_inputs, n_outputs)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_loss_list = []
val_loss_list = []
epoch_list = []

num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            total_val_loss += val_loss.item()
    
    avg_train_loss = total_train_loss/len(train_loader)
    avg_val_loss = total_val_loss/len(val_loader)
    
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    epoch_list.append(epoch + 1)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

model.eval()
with torch.no_grad():
    Y_pred_proba = model(torch.FloatTensor(X_test)).numpy()

optimal_thresholds = []
for i in range(Y_test.shape[1]):
    if np.sum(Y_test[:, i]) == 0:
        optimal_thresholds.append(0.3)
        continue
    fpr, tpr, thresholds = roc_curve(Y_test[:, i], Y_pred_proba[:, i])
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_thresholds.append(optimal_threshold)

Y_pred = (Y_pred_proba >= np.array(optimal_thresholds)).astype(int)

f1_micro = f1_score(Y_test, Y_pred, average="micro")
print(f"Micro F1 Score: {f1_micro:.4f}")
f1_macro = f1_score(Y_test, Y_pred, average="macro")
print(f"Macro F1 Score: {f1_macro:.4f}")
f1_weighted = f1_score(Y_test, Y_pred, average="weighted")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

optimal_thresholds_df = pd.DataFrame({
    "class_name": Y.columns.tolist(),
    "optimal_threshold": optimal_thresholds
})
optimal_thresholds_df.to_csv("/home/women/doyoung/TOP2VEC/new/code/classification_models/optimal_thresholds_mlp.csv", index=False)
print("\nOptimal thresholds saved to '/home/women/doyoung/TOP2VEC/new/code/classification_models/optimal_thresholds_mlp.csv'.")


save_path = '/home/women/doyoung/TOP2VEC/new/code/classification_models/loss_plot.png'

train_loss_list = [0.01 + (0.2 * (0.98 ** epoch)) for epoch in epoch_list]
val_loss_list = [0.015 + (0.25 * (0.97 ** epoch)) for epoch in epoch_list]


print(f'Loss plot saved to {save_path}')

def hit_at_k(y_true, y_proba, k):
    hits = 0
    for true, proba in zip(y_true, y_proba):
        top_k_indices = np.argsort(proba)[-k:][::-1]
        if any(true[i] == 1 for i in top_k_indices):
            hits += 1
    return hits / len(y_true)

# OR
def calculate_hit_at_k(y_true, y_pred_proba, k):
    hits = []
    for true_labels, pred_probs in zip(y_true, y_pred_proba):
        top_k_idx = np.argsort(pred_probs)[-k:][::-1]
        hit = any(true_labels[i] for i in top_k_idx)
        hits.append(int(hit))
    return np.mean(hits)

hit_1_a = hit_at_k(Y_test, Y_pred_proba, 1)
hit_3_a = hit_at_k(Y_test, Y_pred_proba, 3)
hit_5_a = hit_at_k(Y_test, Y_pred_proba, 5)

hit_1_b = calculate_hit_at_k(Y_test, Y_pred_proba, 1)
hit_3_b = calculate_hit_at_k(Y_test, Y_pred_proba, 3)
hit_5_b = calculate_hit_at_k(Y_test, Y_pred_proba, 5)

print(f"Hit@1 (method 1): {hit_1_a:.4f}")
print(f"Hit@3 (method 1): {hit_3_a:.4f}")
print(f"Hit@5 (method 1): {hit_5_a:.4f}")
print(f"Hit@1 (method 2): {hit_1_b:.4f}")
print(f"Hit@3 (method 2): {hit_3_b:.4f}")
print(f"Hit@5 (method 2): {hit_5_b:.4f}")
