import torch

class Config:
    """General configuration settings applicable across different models."""

    def __init__(self, train_size=50, test_size=20, max_sequence_length=100, vocab_size=10_000, embedding_dim=50, batch_size=32,seed=22):
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size         
        self.embedding_dim = embedding_dim        
        self.batch_size = batch_size  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")
