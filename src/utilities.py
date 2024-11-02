import dotenv

class Config:
    """General configuration settings applicable across different models."""

    def __init__(self, train_size=50, test_size=20, seed=22):
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        dotenv.load_dotenv()
