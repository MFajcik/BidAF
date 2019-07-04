import torch
from torch import nn

from models.modelparts import BiLSTM, Embedder, SpanPredictionModule


class Baseline(SpanPredictionModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedder = Embedder(vocab, config)
        self.encoder = BiLSTM(config)
        self.lin_S = nn.Linear(config["RNN_nhidden"] * 2, config["RNN_nhidden"] * 2)
        self.lin_E = nn.Linear(config["RNN_nhidden"] * 2, config["RNN_nhidden"] * 2)
        self.dropout = nn.Dropout(p=config["dropout_rate"])

    def forward(self, batch):
        q_emb, d_emb = self.embedder(batch.question), self.embedder(batch.document)
        q_enc, d_enc = self.encoder(q_emb), self.encoder(d_emb)
        q = q_enc.max(dim=-2)[0]
        q_s, q_e = self.dropout(self.lin_S(q)), self.dropout(self.lin_E(q))
        q_s.unsqueeze_(-1), q_e.unsqueeze_(-1)
        s, e = torch.bmm(d_enc, q_s), torch.bmm(d_enc, q_e)
        s.squeeze_(-1), e.squeeze_(-1)
        return s, e