import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedder(torch.nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.scale_grad = config['scale_emb_grad_by_freq']
        self.init_vocab(vocab, config['optimize_embeddings'])
        logging.info(f"Optimize embeddings = {config['optimize_embeddings']}")
        logging.info(f"Scale grad by freq: {self.scale_grad}")
        logging.info(f"Vocabulary size = {len(vocab.vectors)}")

    def init_vocab(self, vocab, optimize_embeddings=False, device=None):
        self.embedding_dim = vocab.vectors.shape[1]
        self.embeddings = torch.nn.Embedding(len(vocab), self.embedding_dim, scale_grad_by_freq=self.scale_grad)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.embeddings.weight.requires_grad = optimize_embeddings
        self.vocab = vocab
        if device is not None:
            self.embeddings = self.embeddings.to(device)

    def forward(self, input):
        return self.embeddings(input)


class CharEmbedder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embeddings = nn.Embedding(len(vocab), config["char_embedding_size"], padding_idx=1)
        self.embeddings.weight.data.uniform_(-0.001, 0.001)
        self.dropout = nn.Dropout(p=config["dropout_rate"])
        self.char_conv = nn.Conv2d(1,  # input channels
                                   config["char_channel_size"],  # output channels
                                   (config["char_embedding_size"], config["char_channel_width"])  # kernel size
                                   )

    def forward(self, input):
        """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
        """
        batch_size = input.size(0)
        word_len = input.shape[-1]
        # (batch, seq_len, word_len, char_dim)
        # TODO:check input on embeddings
        x = self.dropout(self.embeddings(input))
        char_dim = x.shape[-1]
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, char_dim, word_len).unsqueeze(1)
        # (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze()
        # (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, x.shape[-1])
        return x


class HighwayNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = config["highway_layers"]
        for i in range(self.layers):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(nn.Linear(config["highway_dim"] * 2, config["highway_dim"] * 2),
                                  nn.ReLU()))
            gate = nn.Linear(config["highway_dim"] * 2, config["highway_dim"] * 2)

            # We should bias the highway layer to just carry its input forward when training starts.
            # We do that by setting the bias on gate affine transformation to be positive, because
            # that means `g` will be biased to be high, so we will carry the input forward.
            # The bias on `B(x)` is the second half of the bias vector in each Linear layer.
            gate.bias.data.fill_(1)

            setattr(self, f'highway_gate{i}',
                    nn.Sequential(gate,
                                  nn.Sigmoid()))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        for i in range(self.layers):
            h = getattr(self, f'highway_linear{i}')(x)
            g = getattr(self, f'highway_gate{i}')(x)
            x = (1 - g) * h + g * x
        return x


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

    def get_output_dim(self):
        raise NotImplementedError("Objects need to implement this method!")


class RNN(Encoder):
    def __init__(self, config, padding_value=None):
        super(RNN, self).__init__(config)
        self.rnn = None
        self.padding_value = 0. if padding_value is None else padding_value

    def init_hidden(self, directions, initfun=torch.zeros, requires_grad=True):
        """
        Init RNN hidden state
        :param directions: number of directions
        :param initfun: function to initialize hidden state from,
                default: torch.randn, which provides samples from normal gaussian distribution (0 mean, 1 variance)
        :param requires_grad: if the hidden states should be learnable, default = True

        Initializes variable self.hidden
        """
        self.hidden_params = torch.nn.Parameter(
            initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad)
        )
        self.cell_params = torch.nn.Parameter(
            initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad))

    def forward(self, inp, lengths=None, batch_first=True):
        """
        :param inp: Shape BATCH_SIZE x LEN x H_DIM
        """
        assert self.rnn
        bsz = inp.shape[0]
        if self.init_hidden is not None:
            hidden_params = self.hidden_params.repeat(1, bsz, 1)
            cell_params = self.cell_params.repeat(1, bsz, 1)
            outp = self.rnn(inp, (hidden_params, cell_params))[0]
        elif lengths is None:
            outp = self.rnn(inp)[0]
        else:
            inp_packed = pack_padded_sequence(inp, lengths, batch_first=batch_first, enforce_sorted=False)
            outp_packed = self.rnn(inp_packed)[0]
            outp, output_lengths = pad_packed_sequence(outp_packed, batch_first=batch_first,
                                                       padding_value=self.padding_value)
        return outp

    def get_output_dim(self):
        return self.output_dim


class LSTM(RNN):
    def __init__(self, config, init_hidden=None, **kwargs):
        super().__init__(config, **kwargs)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=config['dropout_rate'],
            batch_first=True,
            bidirectional=False)
        if init_hidden:
            self.init_hidden(directions=1)
        else:
            self.init_hidden = init_hidden
        self.output_dim = config['RNN_nhidden']


class BiLSTM(RNN):
    def __init__(self, config, init_hidden=None, **kwargs):
        super().__init__(config, **kwargs)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=float(config['dropout_rate']),
            batch_first=True,
            bidirectional=True)
        if init_hidden:
            self.init_hidden(directions=2)
        else:
            self.init_hidden = init_hidden
        self.output_dim = config['RNN_nhidden'] * 2


class SpanPredictionModule(nn.Module):
    def predict(self, batch):
        start_pred_logits, end_pred_logits = self.forward(batch)
        start_pred, end_pred = torch.nn.functional.softmax(start_pred_logits, dim=1), torch.nn.functional.softmax(
            end_pred_logits, dim=1)
        return self.decode(start_pred, end_pred, naive=True)

    @staticmethod
    def decode(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> \
            (torch.Tensor, torch.Tensor):
        """
        This method has been borrowed from AllenNLP
        :param span_start_logits:
        :param span_end_logits:
        :return:
        """
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        valid_span_log_probs = span_log_probs + span_log_mask

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        valid_span_probs = F.softmax(valid_span_log_probs, dim=-1)

        best_span_probs, best_spans = valid_span_probs.max(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        return best_span_probs, (span_start_indices, span_end_indices)
