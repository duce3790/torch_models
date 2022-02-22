import torch.nn as nn
import torch.nn.functional as F


class FNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, token_num, seq_len, hidden_dim=None, use_direct_connection=True):
        super().__init__()
        embedding_dim = 300
        self.token_num = token_num
        self.embedding = nn.Embedding(token_num, embedding_dim=embedding_dim)
        self.use_direct_connection = use_direct_connection
        if self.use_direct_connection:
            self.direct_connection_fc = nn.Linear(seq_len * embedding_dim, token_num)
        if hidden_dim is None:
            hidden_dim = 100
        self.fc2 = nn.Linear(seq_len * embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, token_num)

    def forward(self, input):
        embedding = self.embedding(input).view(input.shape[0], -1)
        output = self.fc3(F.tanh(self.fc2(embedding)))
        if self.use_direct_connection:
            output += self.direct_connection_fc(embedding)
        return F.log_softmax(output)