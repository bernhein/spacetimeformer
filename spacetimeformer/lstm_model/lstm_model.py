import random

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf
from torch.utils.tensorboard import SummaryWriter
from spacetimeformer.data.decker_format.tensorboard_writer import tensorboardWriter
from spacetimeformer.spacetimeformer_model.nn.embed import SpacetimeformerEmbedding

class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
        input_dim: int = 1,
        hidden_dim: int = 1,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
    def forward(self, x_context: torch.Tensor):
        outputs, (hidden, cell) = self.lstm(x_context)
        return hidden, cell


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 7,
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_t, hidden, cell):
        output, (hidden, cell) = self.lstm(x_t, (hidden, cell))
        y_t1 = self.fc(output)
        return y_t1, hidden, cell


class LSTM_Seq2Seq(nn.Module):
    def __init__(self, stfEmbedding: SpacetimeformerEmbedding, encoder: LSTM_Encoder, decoder: LSTM_Decoder):
        super().__init__()
        self.stfEmbedding = stfEmbedding
        self.encoder = encoder
        self.decoder = decoder

    def _merge(self, x, y):
        return torch.cat((x, y), dim=-1)

    def forward(
        self,
        x_context,
        y_context,
        x_target,
        y_target,
        teacher_forcing_prob,
    ):

        # Encoder Embedding
        context_emb, _, _ = self.stfEmbedding(
            y_context, x_context, is_encoder=True  # 
        )
        # Decoder Embedding
        target_emb, _, _ = self.stfEmbedding(
            y_target, x_target, is_encoder=True  # 
        )

        pred_len = target_emb.shape[1]
        batch_size = target_emb.shape[0]
        y_dim = 7
        outputs = -torch.ones(batch_size, pred_len, y_dim).to(y_target.device)        

        batch_size_ctxt = context_emb.shape[0]
        for i in range(batch_size_ctxt):
           hidden, cell = self.encoder(context_emb[i].unsqueeze(0))

        decoder_input = target_emb[0].unsqueeze(0)

        for t in range(1, batch_size):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[t] = output #.squeeze(1)

            decoder_input = target_emb[t].unsqueeze(0) # if random.random() < teacher_forcing_prob else output

        return outputs


class LSTM_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int = 6,
        d_y: int = 1,
        time_emb_dim: int = 0,
        n_layers: int = 2,
        hidden_dim: int = 32,
        dropout_p: float = 0.2,
        # training
        learning_rate: float = 1e-3,
        teacher_forcing_prob: float = 0.5,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        # embed_method: str = "spatio-temporal-event",  # embedding method
        # initial_downsample_convs: int = 0,      # initial downsampling convs
        # start_token_len: int = 64,              # length of the start token
        null_value: float = None,               # null value
        d_model: int = 256,
        comment: str = "lstm",
    ):
        super().__init__(
            l2_coeff=l2_coeff,
            learning_rate=learning_rate,
            loss=loss,
            linear_window=linear_window,
            comment=comment, 
            d_model=d_model
        )


        # Embedding
        self.embedding = SpacetimeformerEmbedding(
            d_y=d_y,
            d_x=d_x,
            d_model=d_model,
            null_value=null_value,
        )
        input_dim = d_model

        self.encoder = LSTM_Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.decoder = LSTM_Decoder(
            output_dim=7,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.model = LSTM_Seq2Seq(self.embedding, self.encoder, self.decoder).to(self.device)

        self.teacher_forcing_prob = teacher_forcing_prob

    @property
    def train_step_forward_kwargs(self):
        return {"force": self.teacher_forcing_prob}

    @property
    def eval_step_forward_kwargs(self):
        return {"force": 0.0}

    def forward_model_pass(self, x_c, y_c, x_t, y_t, force=None):
        assert force is not None
        preds = self.model.forward(x_c, y_c, x_t, y_t, teacher_forcing_prob=force)
        return (preds,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="Hidden dimension for LSTM network.",
        )
        parser.add_argument(
            "--n_layers",
            type=int,
            default=2,
            help="Number of stacked LSTM layers",
        )
        parser.add_argument(
            "--dropout_p",
            type=float,
            default=0.3,
            help="Dropout fraction for LSTM.",
        )
        parser.add_argument(
            "--d_model", type=int, default=256, help="Transformer embedding dimension."
        )