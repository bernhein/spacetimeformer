import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import spacetimeformer as stf

from .encoder import VariableDownsample


class SpacetimeformerEmbedding(nn.Module):
    def __init__(
        self,
        d_y,      # dimension of the output
        d_x,        # dimension of the input
        d_model=256,    # dimension of the model
        method="spatio-temporal-event",   # method to use for embedding
        downsample_convs=1,     # number of downsampling convolutions
        # start_token_len=0,      # length of the start token
        null_value=None,        # value to use for null values
    ):
        super().__init__()
        
        assert method in ["spatio-temporal-event", "spatio-temporal", "temporal"]
        self.method = method

        self.x_embedder = stf.Time2Vec(d_x, embed_dim=d_model)


        if self.method == "spatio-temporal":
            self.var_embedder = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)

        if self.method == "spatio-temporal-event":
            #self.var_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
            self.typeEvnt_embedder   = nn.Embedding(num_embeddings=54, embedding_dim=d_model) # sourceType & eventName
            self.id_embedder         = nn.Embedding(num_embeddings=75, embedding_dim=d_model) # id
            # self.typeVal_emb    = nn.Linear(5, d_model)  # sourceType & value & value idx (0..3)
            self.typeVal_0_embedder    = nn.Linear(2, d_model)  # sourceType & value & value idx (0..3)
            self.typeVal_1_embedder    = nn.Linear(2, d_model)  # sourceType & value & value idx (0..3)
            self.typeVal_2_embedder    = nn.Linear(2, d_model)  # sourceType & value & value idx (0..3)
            self.typeVal_3_embedder    = nn.Linear(2, d_model)  # sourceType & value & value idx (0..3)

        # self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [VariableDownsample(d_y, d_model) for _ in range(downsample_convs)]
        )

        self._benchmark_embed_enc = None
        self._benchmark_embed_dec = None
        self.d_model = d_model
        self.null_value = null_value

    def __call__(self, y, x, is_encoder=True):
        self.device = x.device
        if self.method == "spatio-temporal":
            val_time_emb, space_emb, var_idxs = self.spatio_temporal_embed(
                y, x, is_encoder
            )
        elif self.method == "spatio-temporal-event":
            val_time_emb, space_emb = self.spatio_temporal_event_embed(
                y, x, is_encoder
            )
            var_idxs = None
        else:
            val_time_emb, space_emb = self.temporal_embed(y, x, is_encoder)
            var_idxs = None

        return val_time_emb, space_emb, var_idxs

    def temporal_embed(self, y, x, is_encoder=True):
        bs, length, d_y = y.shape # batch size, length, dimension of y

        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        if not self.TIME:
            x = torch.zeros_like(x)
        x = torch.cat((x, local_pos), dim=-1)
        t2v_emb = self.x_emb(x)

        # 
        emb_inp = torch.cat((y, t2v_emb), dim=-1)
        emb = self.y_emb(emb_inp)

        # "given" embedding
        # given = torch.ones((bs, length)).long().to(x.device)
        # if not is_encoder and self.GIVEN:
        #     given[:, self.start_token_len :] = 0
        # given_emb = self.given_emb(given)
        # emb += given_emb

        if is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        return emb, torch.zeros_like(emb)

    SPACE = True
    TIME = True
    VAL = True
    GIVEN = True
    EVENT = True

    def spatio_temporal_embed(self, y, x, is_encoder=True):
        bs, length, d_y = y.shape # batch size, length, dimension of y

        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1)
        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        x = torch.cat((x, local_pos), dim=-1)
        if not self.TIME:
            x = torch.zeros_like(x)
        if not self.VAL:
            y = torch.zeros_like(y)
        t2v_emb = self.x_emb(x).repeat(1, d_y, 1)
        val_time_inp = torch.cat((y, t2v_emb), dim=-1)
        val_time_emb = self.y_emb(val_time_inp)

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T
            # if not is_encoder:
            #     # mask missing values that need prediction...
            #     given[:, self.start_token_len :, :] = 0
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1)
            if self.null_value is not None:
                # mask null values
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask
            given_emb = self.given_emb(given)
            val_time_emb += given_emb

        if is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # var embedding
        var_idx = torch.Tensor([[i for j in range(length)] for i in range(d_y)])
        var_idx = var_idx.long().to(x.device).view(-1).unsqueeze(0).repeat(bs, 1)
        var_idx_true = var_idx.clone()
        if not self.SPACE:
            var_idx = torch.zeros_like(var_idx)
        var_emb = self.var_emb(var_idx)

        return val_time_emb, var_emb, var_idx_true

    def getEmbeddingVal(self,  key:str, val):
        val = val.to(self.device)

        if key == 'typeEvent':
            res = torch.sum(self.typeEvnt_embedder(val), -2)
            return res.view(res.shape[1],res.shape[2])
        elif key == 'id':
            _id     = val.to(torch.int64)
            res = torch.sum(self.id_embedder(_id), -2)
            r = res.view(res.shape[1],res.shape[2])
            return r
        elif key == 'typeVal_0':
            res = self.typeVal_0_embedder(val)
            r = res.view(res.shape[1],res.shape[2])
            return r
        elif key == 'typeVal_1':
            res = self.typeVal_0_embedder(val)
            return res.view(res.shape[1],res.shape[2])
        elif key == 'typeVal_2':
            res = self.typeVal_0_embedder(val)
            return res.view(res.shape[1],res.shape[2])
        elif key == 'typeVal_3':
            res = self.typeVal_0_embedder(val)
            return res.view(res.shape[1],res.shape[2])

    def spatio_temporal_event_embed(self, y, x, is_encoder=True):
        bs, length, d_y = y.shape # batch size, length, dimension of y

        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        x = torch.cat((x, local_pos), dim=-1)
        if not self.TIME:
            x = torch.zeros_like(x)
            
        t2v_emb = self.x_embedder(x)
        # value embeddings
        # splitted = torch.split(y, 1, dim=-1)
        val_0, val_1, val_2, val_3, sourceType, id, event = torch.split(y, 1, dim=-1) 

        embedding = None
        _id     = torch.from_numpy(id.to(torch.int64).cpu().numpy()).to(y.device)
        id_emb  = torch.sum(self.id_embedder(_id), -2)
        if is_encoder:
            typeEvnt = torch.cat((sourceType, event), -1).int()
            #torch.cat((sourceType, val_0), -1).to(y.device)
            # typeVal = torch.cat((sourceType, val_0, val_1, val_2, val_3), -1).to(y.device)
            typeVal_0 = torch.cat((sourceType, val_0), -1).to(y.device)
            typeVal_1 = torch.cat((sourceType, val_1), -1).to(y.device)
            typeVal_2 = torch.cat((sourceType, val_2), -1).to(y.device)
            typeVal_3 = torch.cat((sourceType, val_3), -1).to(y.device)

            typeEvnt_emb    = torch.sum(self.typeEvnt_embedder(typeEvnt), -2)
            
            # typeVal_emb   = self.typeVal_emb(typeVal)
            typeVal_0_emb   = self.typeVal_0_embedder(typeVal_0)
            typeVal_1_emb   = self.typeVal_1_embedder(typeVal_1)
            typeVal_2_emb   = self.typeVal_2_embedder(typeVal_2)
            typeVal_3_emb   = self.typeVal_3_embedder(typeVal_3)

            # sum up all embeddings
            # embedding = typeEvnt_emb + id_emb + typeVal_emb + t2v_emb
            embedding = t2v_emb + typeEvnt_emb + id_emb + typeVal_0_emb + typeVal_1_emb + typeVal_2_emb + typeVal_3_emb
        
        else:
            # sum up all embeddings
            embedding = t2v_emb + typeEvnt_emb + id_emb

        # if is_encoder:
        #     for conv in self.downsize_convs:
        #         emb = conv(emb)
        #         length //= 2

        return embedding, torch.zeros_like(embedding)
