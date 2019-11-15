import torch
import torch.nn.functional as F
from torch import nn

from models.modelparts import BiLSTM, Embedder, SpanPredictionModule


class BidafSimplified(SpanPredictionModule):
    """
    No character embeddings
    No highway network
    """

    def __init__(self, config, vocab):
        super().__init__()
        self.embedder = Embedder(vocab, config)
        self.encoder = BiLSTM(config)
        self.lin_S = nn.Linear(config["RNN_nhidden"] * 10, 1)
        self.lin_E = nn.Linear(config["RNN_nhidden"] * 10, 1)
        self.dropout = nn.Dropout(p=config["dropout_rate"])

        self.att_weight_c = nn.Linear(config["RNN_nhidden"] * 2, 1)
        self.att_weight_q = nn.Linear(config["RNN_nhidden"] * 2, 1)
        self.att_weight_cq = nn.Linear(config["RNN_nhidden"] * 2, 1)

        config_modeling = config.copy()
        config_modeling["RNN_layers"] = 2
        config_modeling["RNN_input_dim"] = config["RNN_nhidden"] * 8
        self.modeling_layer = BiLSTM(config_modeling)

        config_output = config.copy()
        config_output["RNN_input_dim"] = config["RNN_nhidden"] * 2
        self.output_layer = BiLSTM(config_output)

    def forward(self, batch):
        # 2. Word Embedding Layer
        q_emb, d_emb = self.embedder(batch.question), self.embedder(batch.document)

        # 3. Contextual embedding
        q_enc, d_enc = self.dropout(self.encoder(q_emb)), self.dropout(self.encoder(d_emb))

        # 4. Attention flow
        # (batch, c_len, q_len)
        S = self.compute_similarity_matrix(d_enc, q_enc)
        C2Q = self.co_attention(S, q_enc)
        Q2C = self.max_attention(S, d_enc)

        G = torch.cat((d_enc, C2Q, d_enc * C2Q, d_enc * Q2C), dim=-1)

        M = self.dropout(self.modeling_layer(G))
        M_2 = self.dropout(self.output_layer(M))

        # (batch, c_len)
        # Original paper also puts dropout here?
        start_logits = self.lin_S(torch.cat((G, M), dim=-1)).squeeze(-1)
        end_logits = self.lin_E(torch.cat((G, M_2), dim=-1)).squeeze(-1)

        return start_logits, end_logits

    def compute_similarity_matrix(self, c, q):
        """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
        """

        c_len, q_len = c.shape[1], q.shape[1]

        # "a" component similarity
        # (batch x c_len x 1) -> (batch x c_len x q_len)
        a_similarity = self.att_weight_c(c).expand(-1, -1, q_len)

        # "b" component similarity
        # (batch x q_len x 1) -> (batch x 1 x q_len) -> (batch x c_len x q_len)
        b_similarity = self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1)

        # Element wise similarity
        # A(B ⊙ w)
        # B = Q query_matrix (usually smaller), A = context_matrix
        element_wise_similarity = torch.bmm(c, (q * self.att_weight_cq.weight.unsqueeze(0))
                                            .transpose(1, 2)) + self.att_weight_cq.bias

        # Now add together to get total similarity
        total_similarity = a_similarity + b_similarity + element_wise_similarity

        return total_similarity

    def max_attention(self, similarity_matrix, context_vectors):
        """
        Compute max-attention w.r.t. context
        :param similarity_matrix: (batch, c_len, q_len)
        :param context_vectors  (batch, c_len, hidden_size * 2)
        :return (batch, c_len, hidden_size * 2)
        """

        c_len = context_vectors.shape[1]

        # (batch, 1, c_len)
        max_similarities_across_queries = torch.max(similarity_matrix, dim=-1)[0].unsqueeze(1)

        # compare with each context word as dot product

        # (batch, 1, hidden_size * 2)
        s_max = torch.bmm(max_similarities_across_queries, context_vectors)

        S_max = s_max.expand(-1, c_len, -1)

        return S_max

    def co_attention(self, similarity_matrix, query_vectors):
        """
        Compute co-attention w.r.t. query
        :param similarity_matrix: (batch, c_len, q_len)
        :param query_vectors  (batch, q_len, hidden_size * 2)
        :return (batch, c_len, hidden_size * 2)
        """
        s_soft = F.softmax(similarity_matrix, dim=-1)
        return torch.bmm(s_soft, query_vectors)
