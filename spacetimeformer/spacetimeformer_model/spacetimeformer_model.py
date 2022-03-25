from typing import Tuple
from spacetimeformer.data.decker_format.tensorboard_writer import tensorboardWriter

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
import spacetimeformer as stf



class Spacetimeformer_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_y: int = 1,
        d_x: int = 4,
        start_token_len: int = 64,
        attn_factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 2048,
        dropout_emb: float = 0.05,
        dropout_token: float = 0.05,
        dropout_qkv: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn_out: float = 0.05,
        global_self_attn: str = "performer",
        local_self_attn: str = "none",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "none",
        performer_kernel: str = "relu",
        embed_method: str = "spatio-temporal-event",
        performer_relu: bool = True,
        performer_redraw_interval: int = 1000,
        activation: str = "gelu",
        post_norm: bool = False,
        norm: str = "layer",
        init_lr: float = 1e-10,
        base_lr: float = 3e-4,
        warmup_steps: float = 0,
        decay_factor: float = 0.25,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        l2_coeff: float = 0,
        loss: str = "nll",
        linear_window: int = 0,
        class_loss_imp: float = 0.1,
        time_emb_dim: int = 6,
        null_value: float = None,
        verbose=True,
        comment="stf"
    ):
        super().__init__(l2_coeff=l2_coeff, loss=loss, linear_window=linear_window)
        self.spacetimeformer = stf.spacetimeformer_model.nn.Spacetimeformer(
            d_y=d_y,
            d_x=d_x,
            start_token_len=start_token_len,
            attn_factor=attn_factor,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            initial_downsample_convs=initial_downsample_convs,
            intermediate_downsample_convs=intermediate_downsample_convs,
            dropout_emb=dropout_emb,
            dropout_attn_out=dropout_attn_out,
            dropout_qkv=dropout_qkv,
            dropout_ff=dropout_ff,
            dropout_token=dropout_token,
            global_self_attn=global_self_attn,
            local_self_attn=local_self_attn,
            global_cross_attn=global_cross_attn,
            local_cross_attn=local_cross_attn,
            activation=activation,
            post_norm=post_norm,
            device=self.device,
            norm=norm,
            embed_method=embed_method,
            performer_attn_kernel=performer_kernel,
            performer_redraw_interval=performer_redraw_interval,
            time_emb_dim=time_emb_dim,
            verbose=True,
            null_value=null_value,    
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

        self.start_token_len = start_token_len
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.embed_method = embed_method
        self.class_loss_imp = class_loss_imp
        self.set_null_value(null_value)

        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint(f" *** Spacetimeformer Summary: *** ")
        qprint(f"\tModel Dim: {d_model}")
        qprint(f"\tFF Dim: {d_ff}")
        qprint(f"\tEnc Layers: {e_layers}")
        qprint(f"\tDec Layers: {d_layers}")
        qprint(f"\tEmbed Dropout: {dropout_emb}")
        qprint(f"\tToken Dropout: {dropout_token}")
        qprint(f"\tFF Dropout: {dropout_ff}")
        qprint(f"\tAttn Out Dropout: {dropout_attn_out}")
        qprint(f"\tQKV Dropout: {dropout_qkv}")
        qprint(f"\tL2 Coeff: {l2_coeff}")
        qprint(f"\tWarmup Steps: {warmup_steps}")
        qprint(f"\tNormalization Scheme: {norm}")
        qprint(f" ***                         *** ")        

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        time_mask = self.time_masked_idx if train else None

        forecast_loss, class_loss, acc, output, mask = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        x_c, y_c, x_t, y_t = batch
        stats = self._compute_stats(mask * output, mask * y_t)
        stats["forecast_loss"] = forecast_loss
        # stats["class_loss"] = class_loss
        stats["loss"] = forecast_loss # + self.class_loss_imp * class_loss
        # stats["acc"] = acc

        """
        # temporary traffic stats:
        preds = self._inv_scaler(output.detach().cpu().numpy())
        true = self._inv_scaler(y_t.detach().cpu().numpy())
        mask = mask.detach().cpu().numpy()
        time_based_mae = abs((mask * preds) - (mask * true)).mean((0, -1))
        for time_idx in range(len(time_based_mae)):
            stats[f"mae_traffic_time_{time_idx}"] = time_based_mae[time_idx]
        """
        return stats

    # def classification_loss(
    #     self, logits: torch.Tensor, labels: torch.Tensor
    # ) -> Tuple[torch.Tensor]:

    #     labels = labels.view(-1).to(logits.device)
    #     d_y = labels.max() + 1

    #     logits = logits.view(
    #         -1, d_y
    #     )  #  = torch.cat(logits.chunk(bs, dim=0), dim=1).squeeze(0)

    #     class_loss = F.cross_entropy(logits, labels)
    #     acc = torchmetrics.functional.accuracy(
    #         torch.softmax(logits, dim=1),
    #         labels,
    #     )
    #     return class_loss, acc


    # def classification_loss_event(
    #     self, logits: torch.Tensor, labels: torch.Tensor
    # ) -> Tuple[torch.Tensor]:

    #     labels = labels.view(-1).to(logits.device)
    #     d_y = labels.max() + 1

    #     logits = logits.view(
    #         -1, d_y
    #     )  #  = torch.cat(logits.chunk(bs, dim=0), dim=1).squeeze(0)

    #     class_loss = F.cross_entropy(logits, labels)
    #     acc = torchmetrics.functional.accuracy(
    #         torch.softmax(logits, dim=1),
    #         labels,
    #     )
    #     return class_loss, acc

    def compute_loss(self, batch, time_mask=None, forward_kwargs={}):
        x_c, y_c, x_t, y_t = batch
        outputs, (logits, labels) = self(x_c, y_c, x_t, y_t, **forward_kwargs)

        forecast_loss, mask = self.forecasting_loss(
            outputs=outputs, y_t=y_t, time_mask=time_mask
        )

        # if self.embed_method == "spatio-temporal" and self.class_loss_imp > 0:
        #     class_loss, acc = self.classification_loss(logits=logits, labels=labels)
        # elif self.embed_method == "spatio-temporal-event" and self.class_loss_imp > 0:
        #     class_loss, acc = self.classification_loss_event(logits=logits, labels=labels)
        # else:
        #     class_loss, acc = 0.0, -1.0
        class_loss = None
        acc = None

        return forecast_loss, class_loss, acc, outputs, mask

    def forward_model_pass(self, x_c, y_c, x_t, y_t, output_attn=False):
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
            y_t = y_t.unsqueeze(-1)
        batch_x = y_c
        batch_x_mark = x_c
        
        batch_y = y_t
        batch_y_mark = x_t

        dec_inp = torch.cat(
            [
                # batch_y[:, : self.start_token_len, :],
                torch.zeros((batch_y.shape[0], y_t.shape[1], batch_y.shape[-1])).to(
                    self.device
                ),
            ],
            dim=1,
        ).float()

        output, (logits, labels), attn = self.spacetimeformer(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
            output_attention=output_attn,
        )

        if output_attn:
            return output, (logits, labels), attn
        return output, (logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.base_lr, weight_decay=self.l2_coeff,
        )
        scheduler = stf.lr_scheduler.WarmupReduceLROnPlateau(
            optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=2,
            factor=self.decay_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/forecast_loss",
                "reduce_on_plateau": True,
            },
        }

    # def on_train_start(self) -> None:
    def on_train_epoch_end(self) -> None:
        
        typeVals = [f'typeVal_{x}' for x in range(4)]

        # get unique file names
        DrillingMotor_names = self.motors_data.query("Type == 'DrillingMotor'")['LogfileName'].unique()
        Axis_names = self.motors_data.query("Type == 'Axis'")['LogfileName'].unique()
        FreqConverter_names = self.motors_data.query("Type == 'FreqConverter'")['LogfileName'].unique()
        Sensor_names = self.motors_data['SensorBMK'].unique()
        ImpulseValve_names = self.valves_data.query("Type == 'ImpulseValve'")['LogfileName'].unique()
        MagneticValve = self.valves_data.query("Type == 'MagneticValve'")['LogfileName'].unique()

        def add_type(x):
            if x in DrillingMotor_names:
                return 'DrillingMotor'
            elif x in Axis_names:
                return 'Axis'
            elif x in FreqConverter_names:
                return 'FreqConverter'
            elif x in Sensor_names:
                return 'Sensor'
            elif x in ImpulseValve_names:
                return 'ImpulseValve'
            elif x in MagneticValve:
                return 'MagneticValve'
            else:
                return ''


        builded_embeddings = {}
        for embedding_type in ['typeEvent', 'id'] + typeVals:

            vals = self.embeddingObservData[embedding_type]['vals'] # .to(self.device)

            embed_space = self.spacetimeformer.embedding.getEmbeddingVal(key=embedding_type, val=vals),
            df = pd.DataFrame(
                embed_space[0].detach().cpu().numpy(),
                columns=self.cols
            )
            df['label'] = self.embeddingObservData[embedding_type]['labels']
            df['type'] = df["label"].apply(lambda x: add_type(x))
            lbls = df[['label', 'type']].values

            # Make data available for embedding combinations
            builded_embeddings[embedding_type] = df

            # Log data to WandB and TensorBoard
            self.logger.log_table(key=f'{embedding_type}', dataframe=df, step=self.trainer.global_step)
            self.writer[embedding_type].add_embedding(embed_space[0], metadata=lbls, global_step=self.trainer.global_step)

        ################################
        # build combinations of event and type and id embeddings 

        self.cols

        df_axis = pd.DataFrame(columns=self.cols + ['label', 'type'])
        empty_row = {'label': [], 'type': []}
        empty_row.update({x: [] for x in self.cols})

        df_empty_row = pd.DataFrame(empty_row)
        df_all_embeddings = df_empty_row.copy()
        for idx, row in builded_embeddings['id'].iterrows():
            if row['type'] == 'Axis':
                # iterate through axis events
                for event in self.embedding_events['axis']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'axis_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['type']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            elif row['type'] == 'DrillingMotor':
                # iterate through axis events
                for event in self.embedding_events['drillingMotor']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'motor_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['type']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            
            elif row['type'] == 'FreqConverter':
                # iterate through axis events
                for event in self.embedding_events['freqConv']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'motor_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['type']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            elif row['type'] == 'MagneticValve':
                # iterate through axis events
                for event in self.embedding_events['valve']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'mValve_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['type']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            elif row['type'] == 'ImpulseValve':
                # iterate through axis events
                for event in self.embedding_events['valve']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'iValve_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['type']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            elif row['type'] == 'Sensor':

                sum_row = row[self.cols]
                sum_row['label'] = f'{row["label"]}_Sensor'
                sum_row['type'] = row['type']

                # append it
                df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)
            elif row['label'] == 'Mode':
                # iterate through axis events
                for event in self.embedding_events['mode']:
                    event_row = builded_embeddings['typeEvent'].query(f"label == 'mode_{event}'")
                    sum_row = event_row[self.cols] + row[self.cols]
                    sum_row['label'] = f'{row["label"]}_{event}'
                    sum_row['type'] = row['label']

                    # append it
                    df_all_embeddings = pd.concat([df_all_embeddings, sum_row], ignore_index=True)


        df_all_embeddings[self.cols] = df_all_embeddings[self.cols].apply(pd.to_numeric)
        
        lbls = df_all_embeddings[['label', 'type']].values
        embed_space = torch.from_numpy(df_all_embeddings[self.cols].values)

        # self.embedding_events['']
        # builded_embeddings['typeEvent']
        self.writer['typeEventID'].add_embedding(embed_space, metadata=lbls, global_step=self.trainer.global_step)


            
        return super().on_train_epoch_end()


    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument(
            "--start_token_len",
            type=int,
            required=True,
            help="Length of decoder start token. Adds this many of the final context points to the start of the target sequence.",
        )
        parser.add_argument(
            "--d_model", type=int, default=256, help="Transformer embedding dimension."
        )
        parser.add_argument(
            "--n_heads", type=int, default=8, help="Number of self-attention heads."
        )
        parser.add_argument(
            "--enc_layers", type=int, default=4, help="Transformer encoder layers."
        )
        parser.add_argument(
            "--dec_layers", type=int, default=3, help="Transformer decoder layers."
        )
        parser.add_argument(
            "--d_ff",
            type=int,
            default=1024,
            help="Dimension of Transformer up-scaling MLP layer. (often 4 * d_model)",
        )
        parser.add_argument(
            "--attn_factor",
            type=int,
            default=5,
            help="ProbSparse attention factor. N/A to other attn mechanisms.",
        )
        parser.add_argument(
            "--dropout_emb",
            type=float,
            default=0.2,
            help="Embedding dropout rate. Drop out elements of the embedding vectors during training.",
        )
        parser.add_argument(
            "--dropout_token",
            type=float,
            default=0.0,
            help="Token dropout rate. Drop out entire input tokens during training.",
        )
        parser.add_argument(
            "--dropout_attn_out",
            type=float,
            default=0.0,
            help="Attention dropout rate. Dropout elements of the attention matrix. Only applicable to attn mechanisms that explicitly compute the attn matrix (e.g. Full).",
        )
        parser.add_argument(
            "--dropout_qkv",
            type=float,
            default=0.0,
            help="Query, Key and Value dropout rate. Dropout elements of these attention vectors during training.",
        )
        parser.add_argument(
            "--dropout_ff",
            type=float,
            default=0.3,
            help="Standard dropout applied to activations of FF networks in the Transformer.",
        )
        parser.add_argument(
            "--global_self_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "prob",
                "performer",
                "nystromformer",
                "benchmark",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--global_cross_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "performer",
                "benchmark",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--local_self_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "prob",
                "performer",
                "benchmark",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--local_cross_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "performer",
                "benchmark",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="gelu",
            choices=["relu", "gelu"],
            help="Activation function for Transformer encoder and decoder layers.",
        )
        parser.add_argument(
            "--post_norm",
            action="store_true",
            help="Enable post-norm architecture for Transformers. See https://arxiv.org/abs/2002.04745.",
        )
        parser.add_argument(
            "--norm",
            type=str,
            choices=["layer", "batch", "scale", "power", "none"],
            default="batch",
        )
        parser.add_argument(
            "--init_lr", type=float, default=1e-10, help="Initial learning rate."
        )
        parser.add_argument(
            "--base_lr",
            type=float,
            default=5e-4,
            help="Base/peak LR. The LR is annealed to this value from --init_lr over --warmup_steps training steps.",
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=0, help="LR anneal steps."
        )
        parser.add_argument(
            "--decay_factor",
            type=float,
            default=0.25,
            help="Factor to reduce LR on plateau (after warmup period is over).",
        )
        parser.add_argument(
            "--initial_downsample_convs",
            type=int,
            default=0,
            help="Add downsampling Conv1Ds to the encoder embedding layer to reduce context sequence length.",
        )
        parser.add_argument(
            "--class_loss_imp",
            type=float,
            default=0.1,
            help="Coefficient for node classification loss function. Set to 0 to disable this feature. Does not significantly impact forecasting results due to detached gradient.",
        )
        parser.add_argument(
            "--intermediate_downsample_convs",
            type=int,
            default=0,
            help="Add downsampling Conv1Ds between encoder layers.",
        )
        parser.add_argument(
            "--time_emb_dim",
            type=int,
            default=12,
            help="Time embedding dimension. Embed *each dimension of x* with this many learned periodic values.",
        )
        parser.add_argument(
            "--performer_kernel",
            type=str,
            default="relu",
            choices=["softmax", "relu"],
            help="Performer attention kernel. See Performer paper for details.",
        )
        parser.add_argument(
            "--performer_redraw_interval",
            type=int,
            default=125,
            help="Training steps between resampling orthogonal random features for FAVOR+ attention",
        )
        parser.add_argument(
            "--embed_method",
            type=str,
            choices=["spatio-temporal", "temporal", "spatio-temporal-event"],
            default="spatio-temporal",
            help="Embedding method. spatio-temporal enables long-sequence spatio-temporal transformer mode while temporal recovers default architecture.",
        )
