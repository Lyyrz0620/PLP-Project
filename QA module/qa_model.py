import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from collections import Counter
from typing import List, Tuple, Dict
import os
import pickle

class QAModel:
    def __init__(self, 
                 encoder_model: str = "BAAI/bge-small-en-v1.5",
                 reranker_model: str = "BAAI/bge-reranker-base",
                 top_k: int = 100,
                 top_n: int = 20,
                 top_games: int = 3,
                 reviews_per_game: int = 3):
        """
        Initialize the QA model with specified components.
        
        Args:
            encoder_model: Name of the encoder model
            reranker_model: Name of the reranker model
            top_k: Number of initial retrieved reviews
            top_n: Number of reviews after reranking
            top_games: Number of top games to recommend
            reviews_per_game: Number of reviews to show per game
        """
        self.encoder = SentenceTransformer(encoder_model)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
        self.top_k = top_k
        self.top_n = top_n
        self.top_games = top_games
        self.reviews_per_game = reviews_per_game
        self.reviews = None
        self.app_names = None
        self.faiss_index = None

    def save_model(self, save_dir: str):
        """
        Save the model components to disk.
        
        Args:
            save_dir: Directory to save the model components
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(save_dir, "faiss_index.bin"))
        
        # Save reviews and app_names
        with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
            pickle.dump({
                'reviews': self.reviews,
                'app_names': self.app_names
            }, f)
        
        # Save model parameters
        with open(os.path.join(save_dir, "params.pkl"), "wb") as f:
            pickle.dump({
                'top_k': self.top_k,
                'top_n': self.top_n,
                'top_games': self.top_games,
                'reviews_per_game': self.reviews_per_game
            }, f)
        
        print(f"Model saved to {save_dir}")

    def load_model(self, save_dir: str):
        """
        Load the model components from disk.
        
        Args:
            save_dir: Directory containing the saved model components
        """
        # Load FAISS index
        self.faiss_index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
        
        # Load reviews and app_names
        with open(os.path.join(save_dir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.reviews = data['reviews']
            self.app_names = data['app_names']
        
        # Load model parameters
        with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
            params = pickle.load(f)
            self.top_k = params.get('top_k', 100)
            self.top_n = params.get('top_n', 20)
            self.top_games = params.get('top_games', 3)
            self.reviews_per_game = params.get('reviews_per_game', 3)
        
        print(f"Model loaded from {save_dir}")
        print(f"Current parameters: top_k={self.top_k}, top_n={self.top_n}, "
              f"top_games={self.top_games}, reviews_per_game={self.reviews_per_game}")

    def update_parameters(self, 
                        top_k: int = None,
                        top_n: int = None,
                        top_games: int = None,
                        reviews_per_game: int = None):
        """
        Update model parameters after loading.
        
        Args:
            top_k: New value for top_k
            top_n: New value for top_n
            top_games: New value for top_games
            reviews_per_game: New value for reviews_per_game
        """
        if top_k is not None:
            self.top_k = top_k
        if top_n is not None:
            self.top_n = top_n
        if top_games is not None:
            self.top_games = top_games
        if reviews_per_game is not None:
            self.reviews_per_game = reviews_per_game
            
        print(f"Updated parameters: top_k={self.top_k}, top_n={self.top_n}, "
              f"top_games={self.top_games}, reviews_per_game={self.reviews_per_game}")

    def load_data(self, data_path: str, sample_percentage: float = 1.0, random_state: int = 42):
        """
        Load and prepare the dataset.
        
        Args:
            data_path: Path to the CSV file containing game reviews
            sample_percentage: Percentage of data to load (0.0 to 1.0)
            random_state: Random seed for reproducibility
        """

        df = pd.read_csv(data_path)
        

        if sample_percentage < 1.0:

            df = df.sample(frac=sample_percentage, random_state=random_state)
            print(f"Loaded {len(df)} samples ({sample_percentage*100:.1f}% of total data)")
        

        if 'review_text_clean' not in df.columns or 'app_name' not in df.columns:
            raise ValueError("CSV file must contain 'review_text_clean' and 'app_name' columns")
        

        self.reviews = df['review_text_clean'].astype(str).tolist()  
        self.app_names = df['app_name'].astype(str).tolist()  

        print("Creating FAISS index...")
        embeddings = self.encoder.encode(self.reviews, show_progress_bar=True)  
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        print("FAISS index created successfully")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode the user query into an embedding vector.
        
        Args:
            query: User's query string
            
        Returns:
            Query embedding as numpy array
        """
        return self.encoder.encode([query])[0]

    def retrieve_reviews(self, query_embedding: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Retrieve top-k reviews using FAISS.
        
        Args:
            query_embedding: Encoded query vector
            
        Returns:
            Tuple of (retrieved reviews, corresponding app names)
        """
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            self.top_k
        )
        
        retrieved_reviews = [self.reviews[i] for i in indices[0]]
        retrieved_app_names = [self.app_names[i] for i in indices[0]]
        

        app_counts = Counter(retrieved_app_names)
        print(f"Retrieved reviews distribution (top 5): {app_counts.most_common(5)}")
        
        return retrieved_reviews, retrieved_app_names

    def rerank_reviews(self, query: str, reviews: List[str]) -> List[int]:
        """
        Rerank reviews using the reranker model.
        
        Args:
            query: Original user query
            reviews: List of reviews to rerank
            
        Returns:
            Indices of top-n reviews after reranking
        """
        pairs = [[query, review] for review in reviews]
        inputs = self.reranker_tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits
        

        all_scores = scores.squeeze().tolist()

        top_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:self.top_n]
        

        print(f"Reranking scores range: min={min(all_scores):.2f}, max={max(all_scores):.2f}")
        
        return top_indices

    def vote_top_games(self, app_names: List[str]) -> List[Tuple[str, int]]:
        """
        Vote for top games based on frequency.
        
        Args:
            app_names: List of app names
            
        Returns:
            List of tuples (app_name, count) for top games
        """
        counter = Counter(app_names)
        return counter.most_common(self.top_games)

    def generate_recommendations(self, query: str) -> Dict:
        """
        Generate game recommendations based on user query.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing top game recommendations and their reasons
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Retrieve reviews
        retrieved_reviews, retrieved_app_names = self.retrieve_reviews(query_embedding)
        print(f"Retrieved {len(retrieved_reviews)} reviews in initial search")
        
        # Rerank reviews
        top_indices = self.rerank_reviews(query, retrieved_reviews)
        top_reviews = [retrieved_reviews[i] for i in top_indices]
        top_app_names = [retrieved_app_names[i] for i in top_indices]
        print(f"After reranking, kept {len(top_reviews)} reviews")
        
        # Vote for top games
        top_games = self.vote_top_games(top_app_names)
        print(f"Top games after voting: {top_games}")
        
        # Generate recommendations with multiple reviews per game
        recommendations = {
            'top_games': [],
            'reasons': [] 
        }
        
        for app_name, count in top_games:

            game_reviews = [(r, i) for i, (r, a) in enumerate(zip(top_reviews, top_app_names)) 
                          if a == app_name]
            print(f"Found {len(game_reviews)} reviews for game: {app_name}")
            

            game_reviews.sort(key=lambda x: x[1])

            selected_reviews = [r for r, _ in game_reviews[:self.reviews_per_game]]
            
            recommendations['top_games'].append(app_name)
            recommendations['reasons'].append(selected_reviews)
        
        return recommendations

def main():
    # Example usage
    model = QAModel()
    
    # Check if saved model exists
    save_dir = "saved_model"
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "faiss_index.bin")):
        print("Loading saved model...")
        model.load_model(save_dir)
        

        model.update_parameters(
            top_k=200,  
            top_n=50,   
            reviews_per_game=5  
        )
    else:
        print("Training new model...")
        # Load 10% of the data for testing
        model.load_data('final_data_v3.csv', sample_percentage=0.1)
        # Save the model
        model.save_model(save_dir)
    
    # Example query
    query = "What are the best action games with good graphics?"
    recommendations = model.generate_recommendations(query)
    
    print("Top 3 Recommended Games:")
    for i, (app_name, reviews) in enumerate(zip(recommendations['top_games'], recommendations['reasons']), 1):
        print(f"{i}. App Name: {app_name}")
        print("Reasons:")
        for j, review in enumerate(reviews, 1):
            print(f"   {j}. {review}")
        print()

if __name__ == "__main__":
    main() 