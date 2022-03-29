import torch
import torch.nn as nn

import spacetimeformer as stf

from .ffn import FeedForwardNetwork
from math import ceil

class SpacetimeformerEmbedding(nn.Module):
    def __init__(
        self,
        d_y,      # dimension of the output
        d_x,        # dimension of the input
        d_model=256,    # dimension of the model
        downsample_convs=1,     # number of downsampling convolutions
        null_value=None,        # value to use for null values
    ):
        super().__init__()
        
        self.x_embedder = stf.Time2Vec(d_x, embed_dim=d_model)
        self.typeEvnt_embedder   = nn.Embedding(num_embeddings=54, embedding_dim=d_model) # sourceType & eventName
        self.id_embedder         = nn.Embedding(num_embeddings=75, embedding_dim=d_model) # id
        self.typeVal_0_embedder    = FeedForwardNetwork(input=2,hidden=ceil(d_model/2),output=d_model) # sourceType & value & value idx (0..3)
        self.typeVal_1_embedder    = FeedForwardNetwork(input=2,hidden=ceil(d_model/2),output=d_model)
        self.typeVal_2_embedder    = FeedForwardNetwork(input=2,hidden=ceil(d_model/2),output=d_model)
        self.typeVal_3_embedder    = FeedForwardNetwork(input=2,hidden=ceil(d_model/2),output=d_model)

        self._benchmark_embed_dec = None
        self.d_model = d_model
        self.null_value = null_value

    def __call__(self, y, x, is_encoder=True):
        self.device = x.device

        val_time_emb, space_emb = self.spatio_temporal_event_embed(
            y, x, is_encoder
        )
        var_idxs = None

        return val_time_emb, space_emb, var_idxs

    SPACE = True
    TIME = True
    VAL = True
    GIVEN = True
    EVENT = True

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

        # value splitted up
        val_0, val_1, val_2, val_3, sourceType, id, event = torch.split(y, 1, dim=-1) 

        # embeddings
        embedding = None
        _id     = torch.from_numpy(id.to(torch.int64).cpu().numpy()).to(y.device)
        id_emb  = torch.sum(self.id_embedder(_id), -2)
        typeEvnt = torch.cat((sourceType, event), -1).int()
        typeEvnt_emb    = torch.sum(self.typeEvnt_embedder(typeEvnt), -2)
        
        if is_encoder:
            typeVal_0 = torch.cat((sourceType, val_0), -1).to(y.device)
            typeVal_1 = torch.cat((sourceType, val_1), -1).to(y.device)
            typeVal_2 = torch.cat((sourceType, val_2), -1).to(y.device)
            typeVal_3 = torch.cat((sourceType, val_3), -1).to(y.device)
            
            typeVal_0_emb   = self.typeVal_0_embedder(typeVal_0)
            typeVal_1_emb   = self.typeVal_1_embedder(typeVal_1)
            typeVal_2_emb   = self.typeVal_2_embedder(typeVal_2)
            typeVal_3_emb   = self.typeVal_3_embedder(typeVal_3)

            # sum up all embeddings
            embedding = t2v_emb + typeEvnt_emb + id_emb + typeVal_0_emb + typeVal_1_emb + typeVal_2_emb + typeVal_3_emb
        
        else:
            # sum up all embeddings
            embedding = t2v_emb + typeEvnt_emb + id_emb

        return embedding, torch.zeros_like(embedding)
