import os
import numpy as np
import pandas as pd
from top2vec import Top2Vec

class Top2VecTrainer:
    def __init__(self, output_path, input_path):
        self.OUTPUT_PATH = output_path
        self.INPUT_PATH = input_path
        
        os.makedirs(os.path.dirname(self.OUTPUT_PATH), exist_ok=True)
        self.preprocessed_text = []
    
    def load_data(self):
        with open(self.INPUT_PATH, "r", encoding="utf-8") as f:
            self.preprocessed_text = [line.strip() for line in f if line.strip()]
        print("입력 문서 개수:", len(self.preprocessed_text))
    
    def train_model(self):
        self.top2vec_model = Top2Vec(
            documents=self.preprocessed_text, 
            ngram_vocab=True,                 
            embedding_model="doc2vec",        
            min_count=3,
            topic_merge_delta=0.001,  
            speed="learn",                    
            hdbscan_args={
                "min_cluster_size": 2,          
                "cluster_selection_epsilon": 0         
            },
            umap_args={
                "n_neighbors": 2,   
                "n_components": 5,   
                "min_dist": 0.1      
            }
        )
    
    def save_model(self):
        self.top2vec_model.save(self.OUTPUT_PATH)
        print(f"모델 저장 완료")
    
    def process(self):
        self.load_data()
        self.train_model()
        self.save_model()


class Top2VecProcessor:
    def __init__(self, output_dir, input_dir, model_path, title_path):
        self.OUTPUT_DIR = output_dir
        self.INPUT_DIR = input_dir
        self.MODEL_PATH = model_path
        self.TITLE_PATH = title_path
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.preprocessed_text = []
        self.title_ids = []
        self.document_vectors = None
        self.top2vec_model = None
        self.word_vectors_updated = None
        self.vocab_updated = None
    
    def load_data(self):
        with open(self.INPUT_DIR, 'r', encoding='utf-8') as f:
            self.preprocessed_text = [line.strip() for line in f if line.strip()]
        print("입력 문서 개수:", len(self.preprocessed_text))

        with open(self.TITLE_PATH, 'r', encoding='utf-8') as f:
            self.title_ids = [line.split('-')[0].strip() for line in f if line.strip()]
        print("추출된 Title ID 개수:", len(self.title_ids))
    
    def load_model(self):
        self.top2vec_model = Top2Vec.load(self.MODEL_PATH)
        self.document_vectors = self.top2vec_model.document_vectors
        print("모델 로드 완료")
    
    def add_custom_word_embeddings(self):
        word_vectors = self.top2vec_model.word_vectors
        vocab = self.top2vec_model.vocab
        required_words = ["민간", "공공", "대표성"]
        
        for word in required_words:
            if word not in vocab:
                raise ValueError(f"'{word}' 단어가 어휘에 없습니다.")

        민간_vector = word_vectors[vocab.index("민간")]
        공공_vector = word_vectors[vocab.index("공공")]
        대표성_vector = word_vectors[vocab.index("대표성")]
        
        민간_대표성_vector = (민간_vector + 대표성_vector) / 2
        공공_대표성_vector = (공공_vector + 대표성_vector) / 2
        
        new_words = ["민간대표성", "공공대표성"]
        new_vectors = np.vstack([민간_대표성_vector, 공공_대표성_vector])
        
        self.word_vectors_updated = np.vstack([word_vectors, new_vectors])
        self.vocab_updated = vocab + new_words
    
    def save_embeddings(self):

        document_df_900 = pd.DataFrame({
            'Document ID': self.title_ids,
            'Embedding Vector': [list(vec) for vec in self.document_vectors[163:]]  
        })
        document_df_900.to_csv(f"{self.OUTPUT_DIR}/document_embeddings_900.csv", index=False)
        
        document_df_163 = pd.DataFrame({
            'Document ID': list(range(163)),
            'Embedding Vector': [list(vec) for vec in self.document_vectors[:163]]
        })
        document_df_163.to_csv(f"{self.OUTPUT_DIR}/document_embeddings_163.csv", index=False)
        
        document_df_all = pd.DataFrame({
            'Document ID': list(range(len(self.document_vectors))),
            'Embedding Vector': [list(vec) for vec in self.document_vectors]
        })
        document_df_all.to_csv(f"{self.OUTPUT_DIR}/document_embeddings_all.csv", index=False)
        
        word_df = pd.DataFrame(self.word_vectors_updated, index=self.vocab_updated)
        word_df.to_csv(f"{self.OUTPUT_DIR}/word_embeddings.csv", index_label="Word")
        
        topic_df = pd.DataFrame(self.top2vec_model.topic_vectors)
        topic_df.to_csv(f"{self.OUTPUT_DIR}/topic_embeddings.csv", index_label="Topic ID")
        
        print("CSV 파일 저장 완료: document_embeddings_900.csv, document_embeddings_163.csv, document_embeddings_all.csv, word_embeddings.csv, topic_embeddings.csv")
    
    def save_document_topics(self):
        doc_ids = list(range(len(self.document_vectors)))
        doc_topics = self.top2vec_model.get_documents_topics(doc_ids)[0]
        
        doc_topic_df = pd.DataFrame({
            "Document Number": doc_ids,
            "Assigned Topic ID": doc_topics
        })
        
        output_file = os.path.join(self.OUTPUT_DIR, "document_topic_assignment.csv")
        doc_topic_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Document-topic assignment saved to: {output_file}")
    
    def process(self):
        self.load_data()
        self.load_model()
        self.add_custom_word_embeddings()
        self.save_embeddings()
        self.save_document_topics()


if __name__ == "__main__":
    # trainer = Top2VecTrainer(
    #     output_path='/home/women/doyoung/TOP2VEC/new/code/top2vec/trained_model/top2vec_trained900_3_1102',
    #     input_path='/home/women/doyoung/TOP2VEC/new/code/top2vec/모듈화_0211/text_1102.txt'
    # )
    # trainer.process()
    
    processor = Top2VecProcessor(
        output_dir='/home/women/doyoung/TOP2VEC/top2vec_output',
        input_dir='/home/women/doyoung/TOP2VEC/new/code/top2vec/모듈화_0211/text.txt',
        model_path='/home/women/doyoung/TOP2VEC/new/code/top2vec/모듈화_0211/top2vec_trained900_1102',
        title_path='/home/women/doyoung/TOP2VEC/new/code/top2vec/모듈화_0211/title.txt'
    )
    processor.process()

