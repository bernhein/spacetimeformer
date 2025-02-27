from argparse import ArgumentParser
import random
import sys
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import spacetimeformer as stf
import wandb

torch.backends.cudnn.benchmark = False

_MODELS = ["spacetimeformer", "lstm"]

_DSETS = [
    "decker"
]


def create_parser(p):
    model = p[0]
    dset = p[1]

    # Throw error now before we get confusing parser issues
    assert (
        model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    stf.data.CSVTimeSeries.add_cli(parser)
    stf.data.CSVTorchDset.add_cli(parser)
    stf.data.DataModule.add_cli(parser)

    if model == "lstm":
        stf.lstm_model.LSTM_Predictor.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Predictor.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--null_value", type=float, default=None)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser


def create_model(config):
    x_dim, y_dim = None, None
    # @todo configure dataset decker
    if config.dset == "decker":
        x_dim = 4 # hour, minute, microsecond, timestamp: 
        y_dim = 7 # val_0, val_1, val_2, val_3, sourceType, id, event

    assert x_dim is not None
    assert y_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Predictor(
            # encoder
            d_x=x_dim,
            d_y=y_dim,
            # time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            # add
            d_model= config.d_model,
            comment=args.run_name,
        )
    elif config.model == "spacetimeformer":
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Predictor(
            d_y=y_dim,
            d_x=x_dim,
            # start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_token=config.dropout_token,
            dropout_attn_out=config.dropout_attn_out,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            post_norm=config.post_norm,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            # embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            class_loss_imp=config.class_loss_imp,
            null_value=config.null_value,
            comment=args.run_name
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None

    
    #data_path = config.data_path
    data_path = "/home/hein/MasterThesis/MasterThesis/Preprocessing/TrainData/"
    target_cols = [
            "val_0",
            "val_1",
            "val_2",
            "val_3",
            "sourceType",
            "ID",
            "Event",
        ]
    dset = stf.data.CSVTimeSeries(
        data_path=data_path,
        target_cols=target_cols,
    )
    DATA_MODULE = stf.data.DataModule(
        datasetCls=stf.data.CSVTorchDset,
        dataset_kwargs={
            "csv_time_series": dset,
            "context_points": config.context_points,
            "target_points": config.target_points,
            "time_resolution": config.time_resolution,
        },
        batch_size=config.batch_size,
        workers=config.workers,
    )
    INV_SCALER = dset.reverse_scaling
    SCALER = dset.apply_scaling
    NULL_VAL = None

    return DATA_MODULE, INV_SCALER, SCALER, NULL_VAL


def create_callbacks(config):
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=f"./spacetimeformer/spacetimeformer/data/stf_model_checkpoints/{config.run_name}_{''.join([str(random.randint(0,9)) for _ in range(9)])}",
        monitor="val/mse",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}-{val/mse:.2f}",
        save_top_k=1,
    )
    callbacks = [saving]

    if config.early_stopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=5,
            )
        )
    callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                epochs=config.teacher_forcing_anneal_epochs,
            )
        )
    # @todo check if callback is needed
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.target_points,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(args):

    project = os.getenv("STF_WANDB_PROJ")
    entity = os.getenv("STF_WANDB_ACCT")
    log_dir = os.getenv("STF_LOG_DIR")

    if project is None:
        project = "MasterThesis"
    if entity is None:
        entity = "bern-hein"

    if log_dir is None:
        log_dir = "./data/STF_LOG_DIR"
        print(
            "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
        )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    assert (
        project is not None and entity is not None
    ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
        your wandb user/organization name and project title, respectively."
    
    
    
    experiment = wandb.init(
        project=project,
        entity=entity,
        config=args,
        dir=log_dir,
        reinit=True,
        #sync_tensorboard=True
    )
    config = wandb.config
    wandb.run.name = args.run_name
    wandb.run.save()
    logger = WandbLogger(
        experiment=experiment, 
        save_dir="./spacetimeformer/spacetimeformer/data/stf_LOG_DIR"
    )
    logger.log_hyperparams(config)

    # Dset
    data_module, inv_scaler, scaler, null_val = create_dset(args)

    # Model
    args.null_value = null_val
    forecaster = create_model(args)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    # Callbacks
    callbacks = create_callbacks(args)
    test_samples = next(iter(data_module.test_dataloader()))

    if args.wandb and args.plot:
        callbacks.append(
            stf.plot.PredictionPlotterCallback(
                test_samples, total_samples=min(8, args.batch_size)
            )
        )
    if args.wandb and args.model == "spacetimeformer" and args.attn_plot:

        callbacks.append(
            stf.plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(16, args.batch_size),
                raw_data_dir=wandb.run.dir,
            )
        )    

    logger.watch(forecaster, log="all")
    

    trainer = pl.Trainer(
        gpus=args.gpus,
        callbacks=callbacks,
        logger=logger, 
        strategy="dp",
        log_gpu_memory=True,
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        # track_grad_norm=2,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module, ckpt_path="best")

    experiment.finish()

###############################################################################
## -                     Params to train the models                        - ##
###############################################################################


hyper_parameters = [
    ["spacetimeformer", "decker", "--run_name", "stf_d96_c1512-t945-bs6_dl4-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "32", "--d_model", "96", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "4", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["spacetimeformer", "decker", "--run_name", "stf_d312_c1512-t1260-bs8-mae", "--context_points", "1512", "--target_points", "1260", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "312", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "3", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["spacetimeformer", "decker", "--run_name", "stf_d312_c1512-t945-bs8-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "312", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "3", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["spacetimeformer", "decker", "--run_name", "stf_d156_c1512-t945-bs8-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "156", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "3", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["spacetimeformer", "decker", "--run_name", "stf_d96_c1512-t1260-bs8-mae", "--context_points", "1512", "--target_points", "1260", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "96", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "3", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["spacetimeformer", "decker", "--run_name", "stf_d96_c1512-t945-bs32-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "32", "--d_model", "96", "--d_ff", "400", "--enc_layers", "3", "--dec_layers", "3", "--local_cross_attn", "performer", "--local_self_attn", "performer", "--l2_coeff", ".01", "--dropout_emb", ".1", "--loss", "mae", "--time_resolution", "1", "--dropout_ff", ".2", "--n_heads", "8", "--debug", "--early_stopping", "--wandb", "--attn_plot"],
    ["lstm", "decker", "--run_name", "lstm_d96_c500-t100-mae", "--context_points", "500", "--target_points", "100", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "256", "--d_model", "96", "--l2_coeff", ".01", "--loss", "mae", "--time_resolution", "1", "--debug", "--early_stopping", "--wandb"],
    ["lstm", "decker", "--run_name", "lstm_d96_c1512-t1260-mae", "--context_points", "1512", "--target_points", "1260", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "96", "--l2_coeff", ".01","--loss", "mae", "--time_resolution", "1", "--debug", "--early_stopping", "--wandb"],
    ["lstm", "decker", "--run_name", "lstm_d96_c1512-t945-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "96", "--l2_coeff", ".01",  "--loss", "mae", "--time_resolution", "1", "--debug", "--early_stopping", "--wandb"],
    ["lstm", "decker", "--run_name", "lstm_d312_c1512-t945-mae", "--context_points", "1512", "--target_points", "945", "--grad_clip_norm", "1", "--gpus", "0", "--batch_size", "8", "--d_model", "312", "--l2_coeff", ".01","--loss", "mae", "--time_resolution", "1", "--debug", "--early_stopping", "--wandb"],
]


wandb.tensorboard.patch(save=False, tensorboardX=True, root_logdir="./spacetimeformer/spacetimeformer/data/stf_LOG_DIR")

if __name__ == "__main__":
    # CLI

    for params in hyper_parameters:
        parser = create_parser(params)

        args = parser.parse_args(params)

        # for trial in range(args.trials):
        # try:
        main(args)
        # except:
        #     print("failed for following args:")
        #     print(args)
