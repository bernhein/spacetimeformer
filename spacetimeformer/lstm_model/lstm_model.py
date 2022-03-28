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
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
    def forward(self, x_context: torch.Tensor):
        outputs, (hidden, cell) = self.lstm(x_context)
        return hidden, cell


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
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
        # x_context, _, _ = self.stfEmbedding(
        #     y_context, x_context, is_encoder=True  # 
        # )
        context_emb, _, _ = self.stfEmbedding(
            y_context, x_context, is_encoder=True  # 
        )
        # Decoder Embedding
        # x_target, _, _ = self.stfEmbedding(
        #     y_target, x_target, is_encoder=True  # 
        # )
        target_emb, _, _ = self.stfEmbedding(
            y_target, x_target, is_encoder=True  # 
        )
        # if self.stfEmbedding is not None:
        #     x_context = self.stfEmbedding(x_context)
        #     x_target = self.stfEmbedding(x_target)

        pred_len = target_emb.shape[1]
        batch_size = target_emb.shape[0]
        y_dim = target_emb.shape[2]
        outputs = -torch.ones(batch_size, pred_len, y_dim).to(y_target.device)
        # merged_context = self._merge(x_context, y_context)
        hidden, cell = self.encoder(context_emb)

        # decoder_input = self._merge(x_context_emb[:, -1], y_context[:, -1]).unsqueeze(1)
        decoder_input = context_emb

        x = target_emb[0] # Trigger token <SOS>

        for t in range(1, pred_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            # outputs[:, t] = output.squeeze(1)
            outputs[t] = output
            
            best_guess = output.argmax(1) 
            x = target_emb[t] if random.random() < teacher_forcing_prob else best_guess 

            # decoder_y = (
            #     target_emb[:, t].unsqueeze(1)

            #     if random.random() < teacher_forcing_prob
            #     else output
            # )
            # argmax??
            # decoder_input = self._merge(x_target[:, t].unsqueeze(1), decoder_y)
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
        embed_method: str = "spatio-temporal-event",  # embedding method
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
        )

        self.writer = {
            'typeEvent':    SummaryWriter(comment=comment + "-typeEvent"),
            'id':           SummaryWriter(comment=comment + "-id"),
            'typeVal_0':    SummaryWriter(comment=comment + "-typeVal_0"),
            'typeVal_1':    SummaryWriter(comment=comment + "-typeVal_1"),
            'typeVal_2':    SummaryWriter(comment=comment + "-typeVal_2"),
            'typeVal_3':    SummaryWriter(comment=comment + "-typeVal_3"),
            'typeEventID':  SummaryWriter(comment=comment + "-typeEventID"),
        }
        self.motors_data, self.valves_data, self.embedding_events, self.embeddingObservData, self.cols = tensorboardWriter(d_model=d_model)


        self.embed_method = embed_method

        # Embedding
        self.embedding = SpacetimeformerEmbedding(
            d_y=d_y,
            d_x=d_x,
            d_model=d_model,
            # time_emb_dim=time_emb_dim,
            # downsample_convs=initial_downsample_convs,
            method=embed_method,
            # start_token_len=start_token_len,
            null_value=null_value,
        )
        input_dim = d_model
        # input_dim = (time_emb_dim if time_emb_dim > 0 else d_x) + d_y

        self.encoder = LSTM_Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.decoder = LSTM_Decoder(
            output_dim=d_model,
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
        parser.add_argument(
            "--embed_method",
            type=str,
            choices=["spatio-temporal", "temporal", "spatio-temporal-event"],
            default="spatio-temporal-event",
            help="Embedding method. spatio-temporal enables long-sequence spatio-temporal transformer mode while temporal recovers default architecture.",
        )
