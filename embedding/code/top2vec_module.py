import os
import numpy as np
import pandas as pd
from top2vec import Top2Vec
from gensim.models import Doc2Vec
from joblib import dump, load

class Top2VecTrainer:  
    def __init__(self, top2vec_output_path, input_path, doc_count):
        self.DOC_COUNT = doc_count
        self.TOP2VEC_OUTPUT_PATH = f'{top2vec_output_path}_{self.DOC_COUNT}'
        self.INPUT_PATH = input_path

        
        os.makedirs(os.path.dirname(self.TOP2VEC_OUTPUT_PATH), exist_ok=True)
        self.preprocessed_text = []
    
    def load_data(self):
        with open(self.INPUT_PATH, "r", encoding="utf-8") as f:
            self.preprocessed_text = [line.strip() for line in f if line.strip()]
            self.preprocessed_text = self.preprocessed_text[0:self.DOC_COUNT]
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
        # Top2Vec 모델 저장
        self.top2vec_model.save(self.TOP2VEC_OUTPUT_PATH)
        print(f"Top2Vec 모델 저장 완료: {self.TOP2VEC_OUTPUT_PATH}")
    
    def process(self):
        self.load_data()
        self.train_model()
        self.save_model()

class Top2VecProcessor:  # 모델이 학습한 문서 및 단어에 대한 임베딩값 추출 
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
    
    def load_data_ori(self):
        with open(self.INPUT_DIR, 'r', encoding='utf-8') as f:
            self.preprocessed_text = [line.strip() for line in f if line.strip()]
        print("입력 문서 개수:", len(self.preprocessed_text))

        with open(self.TITLE_PATH, 'r', encoding='utf-8') as f:
            self.title_ids = [line.split('-')[0].strip() for line in f if line.strip()]
        print("추출된 Title ID 개수:", len(self.title_ids))

        print(f"Title IDs 개수: {len(self.title_ids)}")  # 문서 개수
        print(f"Document Vectors 개수: {self.document_vectors.shape[0]}")  # 벡터 개수 확인



    def load_data(self):
        with open(self.INPUT_DIR, 'r', encoding='utf-8') as f:
            self.preprocessed_text = [line.strip() for line in f if line.strip()]
        print("입력 문서 개수:", len(self.preprocessed_text))

        with open(self.TITLE_PATH, 'r', encoding='utf-8') as f:
            self.title_ids = [line.split('-')[0].strip() for line in f if line.strip()]
        print("추출된 Title ID 개수:", len(self.title_ids))

        print(f"Title IDs 개수: {len(self.title_ids)}")
        
        if self.document_vectors is not None:
            print(f"학습된 모델 속 Document Vectors 개수: {self.document_vectors.shape[0]}")
        else:
            print("Warning: Document Vectors가 로드되지 않았습니다.")


    
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

        private_vector = word_vectors[vocab.index("민간")]
        public_vector = word_vectors[vocab.index("공공")]
        represent_vector = word_vectors[vocab.index("대표성")]
        
        private_represent_vector = (private_vector + represent_vector) / 2
        public_represent_vector = (public_vector + represent_vector) / 2
        
        new_words = ["민간대표성", "공공대표성"]
        new_vectors = np.vstack([private_represent_vector, public_represent_vector])
        
        self.word_vectors_updated = np.vstack([word_vectors, new_vectors])
        self.vocab_updated = vocab + new_words
    
    def save_embeddings(self):

        document_df_900 = pd.DataFrame({
            'Document ID': self.title_ids,
            'Embedding Vector': [list(vec) for vec in self.document_vectors[:]]  
        })
        document_df_900.to_csv(f"{self.OUTPUT_DIR}/document_embeddings_900.csv", index=False)
        
        
        word_df = pd.DataFrame(self.word_vectors_updated, index=self.vocab_updated)
        word_df.to_csv(f"{self.OUTPUT_DIR}/word_embeddings.csv", index_label="Word")
        
        print("CSV 파일 저장 완료: document_embeddings_900.csv, word_embeddings.csv")
    
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
        self.load_model()
        self.load_data()
        
        self.add_custom_word_embeddings()
        self.save_embeddings()
        self.save_document_topics()

class Top2VecInference:
    def __init__(self, top2vec_model_path, output_dir):
        self.top2vec_model_path = top2vec_model_path
        self.output_dir = output_dir
        self.doc2vec_model = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        top2vec_model = load(self.top2vec_model_path)
        self.doc2vec_model = top2vec_model.model
        print("추론을 위한 Doc2Vec 모델 로드 완료")

    def infer_document_vectors(self, documents):
        doc_vectors = np.array([self.doc2vec_model.infer_vector(doc) for doc in documents])
        return doc_vectors

    def infer_word_vectors(self, words):
        word_vectors = np.array([self.doc2vec_model.wv[word] for word in words if word in self.doc2vec_model.wv])
        return word_vectors

    def save_vectors_to_csv(self, document_vectors, word_vectors, words):
        # 문서 벡터 저장
        doc_df = pd.DataFrame({
            "Document ID": list(range(1, len(document_vectors) + 1)), 
            "Embedding Vector": [list(vec) for vec in document_vectors]
        })
        doc_csv_path = os.path.join(self.output_dir, "inferred_document_vectors.csv")
        doc_df.to_csv(doc_csv_path, index=False, encoding="utf-8-sig")
        print(f"추론한 문서 벡터 저장 완료: {doc_csv_path}")

        # 단어 벡터 저장
        word_df = pd.DataFrame(word_vectors, index=words)
        word_csv_path = os.path.join(self.output_dir, "inferred_word_vectors.csv")
        word_df.to_csv(word_csv_path, index_label="Word", encoding="utf-8-sig")
        print(f"추론한 단어 벡터 저장 완료: {word_csv_path}")

    def process(self, documents):
        self.load_model()
        document_vectors = self.infer_document_vectors(documents)

        unique_words = sorted(set(" ".join([" ".join(doc) for doc in documents]).split()))
        word_vectors = self.infer_word_vectors(unique_words)

        self.save_vectors_to_csv(document_vectors, word_vectors, unique_words)



if __name__ == "__main__":
    # 모델 학습시킨 후, 학습된 모델 저장
    trainer = Top2VecTrainer(
    top2vec_output_path='/home/women/doyoung/Top2Vec/embedding/trained_models',
    input_path='/home/women/doyoung/Top2Vec/preprocessing/input/text_900.txt',
    doc_count=900
    )      
    trainer.process()     
    
    # # 모델이 학습한 문서 및 단어에 대한 임베딩값 추출 
    # processor = Top2VecProcessor(
    #     output_dir='./embedding/output0305/',
    #     input_dir='./data/input/text_900.txt',
    #     model_path='./model/top2vec_model',
    #     title_path='./data/input/title_900.txt'
    # )
    # processor.process()

    # # 학습된 doc2vec을 활용한 문서/단어 임베딩 추론
    # inference = Top2VecInference(
    #     top2vec_model_path='./model/top2vec_model', 
    #     output_dir='./embedding/output/inferred_doc_word/',
    # )
   
    # # 새로운 문서들로 구성된 2차원 리스트. txt/csv 입력 데이터를 2차원 리스트로 변환해야 함.
    # documents = [
    #     ["고양이 집사 아파트 주인"],
    #     ["포도 사과 오렌지 가게 위치"],
    #     ["책상 의자 소파 침대"]
    # ]
    # inference.process(documents)

