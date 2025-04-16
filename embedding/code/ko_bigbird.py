from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")  # BigBirdModel
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer

model.to(device)

with open("/home/women/doyoung/Top2Vec/preprocessing/output/text_gt.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
len(lines)

len_list = []
for line in lines:
    len_list.append(len(line.split(" ")))

def chunk_tokens(token_list, max_length=4096, overlap=512, pad_token=0):
    chunks = []
    start = 0
    while start < len(token_list):
        end = start + max_length
        chunk = token_list[start:end]

        # 마지막 청크가 4096보다 짧다면 패딩 추가
        if len(chunk) < max_length:
            padding_needed = max_length - len(chunk)
            chunk += [pad_token] * padding_needed

        chunks.append(chunk)

        start += (max_length - overlap)  
    return chunks

embeddings = torch.empty(len(lines), 768)

for i, line in enumerate(lines):
    if i%50==0: print(i)
    
    tokenized_ids = tokenizer.convert_tokens_to_ids(line.split(" "))
    chunked_tokens = chunk_tokens(tokenized_ids)
    
    input_ids = torch.tensor(chunked_tokens, device=device)
    attention_mask = (input_ids != 0).long()

    cls_embeddings = []
    for j in range(len(input_ids)):
        with torch.no_grad():
            outputs = model(input_ids=input_ids[j].unsqueeze(0), attention_mask=attention_mask[j].unsqueeze(0))
        cls_embeddings.append(outputs.last_hidden_state[:, 0, :])

    mean_embedding = torch.mean(torch.stack(cls_embeddings), dim=0)

    embeddings[i] = mean_embedding

data = []

for i in range(len(embeddings)):
    document_id = i  
    embedding_vector = embeddings[i].cpu().numpy().tolist() 
    data.append([document_id, embedding_vector])

df = pd.DataFrame(data, columns=["Document ID", "Embedding Vector"])

df.to_csv("/home/women/doyoung/Top2Vec/embedding/ko_bigbird/output/gpt_document_embeddings_bb.csv", index=False, quoting=1)
 
print("CSV 저장 완료: document_embeddings.csv")
