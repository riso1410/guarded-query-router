import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
import tokenizers


def inverse_document_frequency(encoded_docs, vocab_size):
    """ Returns IDF scores in shape [vocab_size] """
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc="Computing IDF"):
        for j in doc:
            counts[i,j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)

    tfidf.fit(counts)

    return torch.FloatTensor(tfidf.idf_)

def collate_for_mlp(list_of_samples):
    """ Collate function that creates batches of flat docs tensor and offsets """
    offset = 0
    flat_docs, offsets, labels = [], [], []
    for example in list_of_samples:
        print(example)
        doc = example['summary']
        label = example['correct_f1']
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        labels.append(label)
        offset += len(doc)
    return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)


def prepare_inputs(input_ids, device=None):
    lens = [len(doc) for doc in input_ids]
    offsets = []
    current_offset = 0
    for l in lens:
        offsets.append(current_offset)
        current_offset += l

    offsets = torch.LongTensor(offsets)
    flat_inputs = torch.LongTensor([tok for doc in input_ids for tok in doc])
    if device is not None:
        offsets = offsets.to(device)
        flat_inputs = flat_inputs.to(device)
    return flat_inputs, offsets

class MLP(nn.Module):
    """Simple MLP"""
    def __init__(self, vocab_size, num_classes,
                 num_hidden_layers=1,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, idf=None, mode='mean',
                 pretrained_embedding=None, freeze=True,
                 embedding_dropout=0.5, problem_type="classification"):
        nn.Module.__init__(self)
        # Treat TF-IDF mode appropriately
        mode = 'sum' if idf is not None else mode
        self.idf = idf

        # Input-to-hidden (efficient via embedding bag)
        if pretrained_embedding is not None:
            # vocabsize is defined by embedding in this case
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze, mode=mode)
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
        self.layers.append(nn.Linear(hidden_size if self.layers else embedding_size, num_classes))

        # Loss function
        if problem_type == "classification":
            self.loss_function = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.loss_function = nn.MSELoss()
        elif problem_type == "multi_label_classification":
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def forward(self, input, offsets, labels=None):
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


