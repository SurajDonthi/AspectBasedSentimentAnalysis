import torch.nn as nn


class SequenceClassifierModel(nn.Module):

    def __init__(self, embedding_size: int = 768,
                 #  hidden_size: int = 128,
                 dropout: float = 0.3,
                 num_classes: int = 3):
        """
        Any Text Classification task like Sentiment Analysis, Topic Classification, \
            Toxic Comment Classification etc.

        Args:
            embedding_size (int, optional): Embedding Size of the BERT model. Defaults to 768.
            hidden_size (int, optional): hidden_size of the . Defaults to 128.
            dropout (float, optional): Dropout is applited before the linear layers. \
                Set dropout to 0 if not required. Defaults to 0.3.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        super().__init__()
        self.embedding_size = embedding_size
        # self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self._layers()

    def _layers(self):

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            # nn.Linear(self.embedding_size, self.hidden_size),
            # nn.ReLU(True),
            nn.Linear(self.embedding_size, self.num_classes),
            nn.Softmax(dim=-1)
        )
        nn.init.xavier_uniform_(self.classifier[1].weight)
        self.classifier[1].bias.data.fill_(0.01)

    def forward(self, embeddings):
        hidden_state_cls = embeddings[0][:, 0, :]
        logits = self.classifier(hidden_state_cls)

        return logits


class TokenClassifierModel(nn.Module):

    def __init__(self, embedding_size: int = 768,
                 #  hidden_size: int = 128,
                 dropout: float = 0.3,
                 max_len: int = 512):
        """
        Any Token Classification task like Named Entity Recognition (NER), \
            POS Tagging, etc.

        Args:
            embedding_size (int, optional): Embedding Size of the BERT model. Defaults to 768.
            hidden_size (int, optional): hidden_size of the . Defaults to 128.
            dropout (float, optional): Dropout is applited before the linear layers. \
                Set dropout to 0 if not required. Defaults to 0.3.
            max_len (int, optional): Length of tokens input to the BERT model. \
                max_len <= embedding_size. Defaults to 2.
        """
        super().__init__()
        self.embedding_size = embedding_size
        # self.hidden_size = hidden_size
        self.max_len = max_len
        self.dropout = dropout
        self._layers()

    def _layers(self):

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            # nn.Linear(self.embedding_size, self.hidden_size),
            # nn.ReLU(True),
            nn.Linear(self.embedding_size, self.max_len),
        )

    def forward(self, embeddings):
        hidden_state_cls = embeddings[0][:, 0, :]
        logits = self.classifier(hidden_state_cls)

        return logits


class DummyClassifier(nn.Module):
    """
    Can be used for tasks which require only embeddings as output like Semantic Similarity.
    """

    def forward(self, embeddings):
        """
        Returns the input embeddings

        Args:
            embeddings (dict(torch.Tensor)): BERT Embeddings

        Returns:
            embeddings (dict(torch.Tensor)): BERT Embeddings
        """
        return embeddings
