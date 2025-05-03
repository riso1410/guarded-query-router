import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

## original code - https://github.com/lgalke/text-clf-baselines/blob/main/models.py


def inverse_document_frequency(
    encoded_docs: list[int], vocab_size: int
) -> torch.FloatTensor:
    """
    Calculate inverse document frequency (IDF) scores for a corpus.
    
    Args:
        encoded_docs (list[int]): List of documents where each document is represented as a list of token IDs
        vocab_size (int): Size of the vocabulary
    
    Returns:
        torch.FloatTensor: IDF scores for each token in the vocabulary
    """
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i, j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)

    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)


# def collate_for_mlp(list_of_samples):
#     """ Collate function that creates batches of flat docs tensor and offsets """
#     offset = 0
#     flat_docs, offsets, labels = [], [], []
#     for example in list_of_samples:
#         print(example)
#         doc = example['summary']
#         label = example['correct_f1']
#         if isinstance(doc, tokenizers.Encoding):
#             doc = doc.ids
#         offsets.append(offset)
#         flat_docs.extend(doc)
#         labels.append(label)
#         offset += len(doc)
#     return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)


def prepare_inputs(
    input_ids: list[int], device: str = None
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Prepare inputs for the MLP model by flattening documents and calculating offsets.
    
    Args:
        input_ids (list[int]): List of documents where each document is represented as a list of token IDs
        device (str, optional): Device to place tensors on. Defaults to None.
    
    Returns:
        tuple[torch.LongTensor, torch.LongTensor]: A tuple containing the flattened input tensor and offsets tensor
    """
    lens = [len(doc) for doc in input_ids]
    offsets = []
    current_offset = 0
    for _l in lens:
        offsets.append(current_offset)
        current_offset += _l

    offsets = torch.LongTensor(offsets)
    flat_inputs = torch.LongTensor([tok for doc in input_ids for tok in doc])
    if device is not None:
        offsets = offsets.to(device)
        flat_inputs = flat_inputs.to(device)
    return flat_inputs, offsets


def prepare_inputs_optimized(
    input_ids: list[list[int]], device: str = None
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Optimized version of prepare_inputs using torch operations for better performance.
    
    Args:
        input_ids (list[list[int]]): List of documents where each document is represented as a list of token IDs
        device (str, optional): Device to place tensors on. Defaults to None.
    
    Returns:
        tuple[torch.LongTensor, torch.LongTensor]: A tuple containing the flattened input tensor and offsets tensor
    """
    lens = torch.LongTensor([len(doc) for doc in input_ids]).to(device)
    offsets = torch.cumsum(lens, dim=0) - lens
    flat_inputs = torch.cat([torch.tensor(doc) for doc in input_ids])
    # flat_inputs = torch.cat([doc for doc in input_ids])

    if device is not None:
        offsets = offsets.to(device)
        flat_inputs = flat_inputs.to(device)
    return flat_inputs, offsets


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for text classification.
    
    This model uses an embedding bag to efficiently process documents of varying lengths,
    followed by one or more hidden layers with optional dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_hidden_layers: int = 1,
        hidden_size: int = 1024,
        hidden_act: int = "relu",
        dropout: float = 0.5,
        idf: torch.Tensor = None,
        mode: str = "mean",
        pretrained_embedding: torch.Tensor = None,
        freeze: bool = True,
        embedding_dropout: float = 0.5,
        problem_type: str = "classification",
    ) -> None:
        """
        Initialize the MLP model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            num_classes (int): Number of output classes
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 1024.
            hidden_act (int, optional): Activation function. Defaults to "relu".
            dropout (float, optional): Dropout probability for hidden layers. Defaults to 0.5.
            idf (torch.Tensor, optional): IDF weights for tokens. Defaults to None.
            mode (str, optional): Pooling mode for EmbeddingBag ('sum', 'mean'). Defaults to "mean".
            pretrained_embedding (torch.Tensor, optional): Pretrained embeddings. Defaults to None.
            freeze (bool, optional): Whether to freeze embeddings. Defaults to True.
            embedding_dropout (float, optional): Dropout probability for embeddings. Defaults to 0.5.
            problem_type (str, optional): Type of problem ('classification', 'regression', 'multi_label_classification'). Defaults to "classification".
        
        Returns:
            None
        """
        nn.Module.__init__(self)
        # Treat TF-IDF mode appropriately
        mode = "sum" if idf is not None else mode
        self.idf = idf

        # Input-to-hidden (efficient via embedding bag)
        if pretrained_embedding is not None:
            # vocabsize is defined by embedding in this case
            self.embed = nn.EmbeddingBag.from_pretrained(
                pretrained_embedding, freeze=freeze, mode=mode
            )
            embedding_size = pretrained_embedding.size(1)
            self.embedding_is_pretrained = True
        else:
            assert vocab_size is not None
            self.embed = nn.EmbeddingBag(vocab_size, hidden_size, mode=mode)
            embedding_size = hidden_size
            self.embedding_is_pretrained = False

        self.activation = getattr(F, hidden_act)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Hidden-to-hidden
        for i in range(num_hidden_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(embedding_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Hidden-to-output
        self.layers.append(
            nn.Linear(hidden_size if self.layers else embedding_size, num_classes)
        )

        # Loss function
        if problem_type == "classification":
            self.loss_function = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.loss_function = nn.MSELoss()
        elif problem_type == "multi_label_classification":
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def forward(
        self, input: torch.tensor, offsets: torch.tensor, labels: torch.tensor = None
    ) -> torch.tensor:
        """
        Forward pass through the MLP model.
        
        Args:
            input (torch.tensor): Flattened token IDs for all documents in batch
            offsets (torch.tensor): Offsets indicating the start of each document
            labels (torch.tensor, optional): Ground truth labels. Defaults to None.
        
        Returns:
            torch.tensor: If labels are provided, returns (loss, logits); otherwise just logits
        """
        # Use idf weights if present
        idf_weights = self.idf[input] if self.idf is not None else None

        h = self.embed(input, offsets, per_sample_weights=idf_weights)

        if self.idf is not None:
            # In the TF-IDF case: renormalize according to l2 norm
            h = h / torch.linalg.norm(h, dim=1, keepdim=True)

        if not self.embedding_is_pretrained:
            # No nonlinearity when embedding is pretrained
            h = self.activation(h)

        h = self.embedding_dropout(h)

        for i, layer in enumerate(self.layers):
            # at least one
            h = layer(h)
            if i != len(self.layers) - 1:
                # No activation/dropout for final layer
                h = self.activation(h)
                h = self.dropout(h)

        if labels is not None:
            loss = self.loss_function(h, labels)
            return loss, h
        return h
