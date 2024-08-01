import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class QAChatbot:
    def __init__(self, model_name='all-MiniLM-L6-v2', data_path='./data-collection/qa/medquad.csv'):
        self.data_path = data_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.answer_embeddings = None
        self.load_data()
        self.generate_embeddings()

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df.dropna(inplace=True)  # Remove rows with any NaN values
        self.df.reset_index(drop=True, inplace=True)  # Reset the index to ensure continuous indices

    def generate_embeddings(self):
        if os.path.exists('answer_embeddings.npy'):
            self.answer_embeddings = np.load('answer_embeddings.npy')
        else:
            self.answer_embeddings = self.model.encode(self.df['answer'].tolist(), convert_to_tensor=False)
            np.save('answer_embeddings.npy', self.answer_embeddings)

    def find_closest_answers(self, query, top_k=1):
        # Encode the query to the same space as your answers
        query_embedding = self.model.encode([query], convert_to_tensor=False)  # Ensure output is not a tensor

        # Ensure the embedding is 2D
        query_embedding = query_embedding.reshape(1, -1)  # Reshape to (1, embedding_size)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.answer_embeddings)[0]  # Ensure answer_embeddings are also 2D

        # Get the top K answers with highest cosine similarity scores
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.df['answer'].iloc[i], similarities[i]) for i in top_indices]

    def display_answers(self, query):
        closest_answers = self.find_closest_answers(query)
        for answer, _ in closest_answers:
            print(f"* {answer}\n")

"""if __name__ == "__main__":
    chatbot = QAChatbot()
    query = input("Enter your question: ")
    chatbot.display_answers(query)"""