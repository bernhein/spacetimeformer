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
  def __init__(self, embedding_size, hidden_size, num_layers, p, stfEmbedding: SpacetimeformerEmbedding):
    super(LSTM_Encoder, self).__init__()

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = stfEmbedding
    # self.embedding = nn.Embedding(input_size, embedding_size)
    
    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

  # Shape of x (26, 32) [Sequence_length, batch_size]
  def forward(self, x_context, y_context):

    # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
    embedding = self.dropout(self.embedding(x_context, y_context, True))
    
    # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
    outputs, (hidden_state, cell_state) = self.LSTM(embedding)

    return hidden_state, cell_state


class LSTM_Decoder(nn.Module):
  def __init__(self, embedding_size, hidden_size, num_layers, p, output_size, stfEmbedding: SpacetimeformerEmbedding):
    super(LSTM_Decoder, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    #self.input_size = input_size

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
    self.output_size = output_size

    # Regularization parameter
    self.dropout = nn.Dropout(p)

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = stfEmbedding

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
    self.fc = nn.Linear(hidden_size, output_size)

  # Shape of x (32) [batch_size]
  def forward(self, x_target, y_target, hidden_state, cell_state):

    # Shape of x (1, 32) [1, batch_size]
    x = x_target.unsqueeze(0)
    y = y_target.unsqueeze(0)

    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
    embed = self.dropout(self.embedding(x, y, False))

    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
    outputs, (hidden_state, cell_state) = self.LSTM(embed, (hidden_state, cell_state))

    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
    predictions = self.fc(outputs)

    # Shape --> predictions (32, 4556) [batch_size , output_size]
    predictions = predictions.squeeze(0)

    return predictions, hidden_state, cell_state



class LSTM_Seq2Seq(nn.Module):
    def __init__(self, encoder: LSTM_Encoder, decoder: LSTM_Decoder, d_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model

    def _merge(self, x, y):
        return torch.cat((x, y), dim=-1)

    def forward(self,
        x_context,
        y_context,
        x_target,
        y_target, 
        tfr=0.5):
        # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
        batch_size = x_context.shape[1]

        # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
        pred_len = x_target.shape[0]

        # Shape --> outputs (14, 32, 5766) 
        outputs = -torch.ones(batch_size, pred_len, self.d_model).to(y_target.device)

        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
        hidden_state, cell_state = self.encoder(x_context, y_context)

        # Shape of x (32 elements)
        x = x_target[0] 
        y = y_target[0]

        for i in range(1, pred_len):
            # Shape --> output (32, 5766) 
            output, hidden_state, cell_state = self.decoder(x, y, hidden_state, cell_state)
            outputs[i] = output
            best_guess = output.argmax(1) # 0th dimension is batch size, 1st dimension is word embedding
            x = x_target[i] if random.random() < tfr else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word
            y = y_target[i]
        # Shape --> outputs (14, 32, 5766) 
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
            d_model,
            hidden_dim, 
            n_layers, 
            dropout_p,
            4,
            self.embedding
        ).to(self.device)
        self.decoder = LSTM_Decoder(
            d_model,
            hidden_dim, 
            n_layers, 
            dropout_p,
            4,
            self.embedding
        ).to(self.device)
        
        # self.decoder = LSTM_Decoder(
        #     output_dim=d_model,
        #     input_dim=input_dim,
        #     hidden_dim=hidden_dim,
        #     n_layers=n_layers,
        #     dropout=dropout_p,
        #     stfEmbedding=self.embedding
        # )
        self.model = LSTM_Seq2Seq(self.encoder, self.decoder, d_model).to(self.device)

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
