from argparse import ArgumentParser
import random
import sys
import warnings
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# from torch.utils.tensorboard import SummaryWriter
# from pytorch_lightning.loggers import TensorBoardLogger

import spacetimeformer as stf
import pandas as pd
import wandb

torch.backends.cudnn.benchmark = False


_MODELS = ["spacetimeformer", "lstm"]

_DSETS = [
    "decker"
]


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

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
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)

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
        x_dim = 4
        y_dim = 7

    assert x_dim is not None
    assert y_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_y=y_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "spacetimeformer":
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_y=y_dim,
            d_x=x_dim,
            start_token_len=config.start_token_len,
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
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            class_loss_imp=config.class_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None

    
    #data_path = config.data_path
    data_path = "/home/hein/MasterThesis/MasterThesis/Preprocessing/TrainData/2022.02.03"
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
    # if config.time_mask_loss:
    #     callbacks.append(
    #         stf.callbacks.TimeMaskedLossCallback(
    #             start=config.time_mask_start,
    #             end=config.target_points,
    #             steps=config.time_mask_anneal_steps,
    #         )
    #     )
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

    # if args.wandb and args.plot:
    #     callbacks.append(
    #         stf.plot.PredictionPlotterCallback(
    #             test_samples, total_samples=min(8, args.batch_size)
    #         )
    #     )
    # if args.wandb and args.model == "spacetimeformer" and args.attn_plot:

        # callbacks.append(
        #     stf.plot.AttentionMatrixCallback(
        #         test_samples,
        #         layer=0,
        #         total_samples=min(16, args.batch_size),
        #         raw_data_dir=wandb.run.dir,
        #     )
        # )

    callbacks.append(
        stf.callbacks.EmbeddingCallback(
            d_model=args.d_model
        )
    )
    # writer = SummaryWriter(wandb.run.dir)
    # x_weight = forecaster.spacetimeformer.embedding.x_embedder.weight
    id_weight = forecaster.spacetimeformer.embedding.id_embedder.weight
    # type_weight = forecaster.spacetimeformer.embedding.type_emb.weight
    # event_weight = forecaster.spacetimeformer.embedding.event_emb.weight
    typeEvent_weight = forecaster.spacetimeformer.embedding.typeEvnt_embedder.weight
    typeVal_0_weight = forecaster.spacetimeformer.embedding.typeVal_0_embedder.weight
    typeVal_1_weight = forecaster.spacetimeformer.embedding.typeVal_1_embedder.weight
    typeVal_2_weight = forecaster.spacetimeformer.embedding.typeVal_2_embedder.weight
    typeVal_3_weight = forecaster.spacetimeformer.embedding.typeVal_3_embedder.weight

    # cols = [f"out_{i}" for i in range(features.shape[1])]
    # 
    # # create pandas dataframe from feature outputs of shape (1478, 512) and add labels
    # df   = pd.DataFrame(features, columns=cols)
    # df['LABEL'] = labels
    # 
    # # log pandas DataFrame to W&B easily
    # table = wandb.Table(columns=df.columns.to_list(), data=df.values)
    # wandb.init(project="embedding_projector")
    # wandb.log({"Pet Breeds": table})
    # wandb.finish()

    # words
    motors_src = "/home/hein/MasterThesis/MasterThesis/motors.json"
    valves_src ="/home/hein/MasterThesis/MasterThesis/valves.json"

    motors_data = pd.read_json(motors_src)
    valves_data = pd.read_json(valves_src)
    # get unique file names
    motor_names = motors_data['LogfileName'].unique()
    valve_names = valves_data['LogfileName'].unique()
    sensor_names = motors_data['SensorBMK'].unique()
    
    # unique events
    _motor_events = ['Start_RT','Start_FT', 'Fault_RT', 'Fault_FT']
    _axis_events = ['TargetChange_RT', 'TargetChange_FT', 'VeloChange_RT', 'VeloChange_FT', 'ERR_RT', 'ERR_FT', 'Start_RT', 'Start_FT', 'Halt_RT', 'Halt_FT', 'Reset_RT', 'Reset_FT']
    _freqConv_events = ['Start_RT', 'Start_FT', 'TargetVeloReached_RT', 'TargetVeloReached_FT', 'RelBrake_RT', 'RelBrake_FT', 'CW_RT', 'CW_FT', 'Error_RT', 'Error_FT']
    _valve_events = ['Input_1_RT', 'Input_1_FT', 'Input_2_RT', 'Input_2_FT', 'Sensor_1_RT', 'Sensor_1_FT', 'Sensor_2_RT', 'Sensor_2_FT']
    _mode_events = ['ControlVoltage', 'ControlVoltage_FT', 'Auto_RT', 'Auto_FT', 'Manual_RT', 'Manual_FT', 'BasePosBusy_RT', 'BasePosBusy_FT', 'BasePosErr_RT', 'BasePosErr_FT', 'ToolChgByOperatorEnable_RT', 'ToolChgByOperatorEnable_FT', 'ToolChgSemiAuto_RT', 'ToolChgSemiAuto_FT', 'Fault_RT', 'Fault_FT', 'FaultRst_RT', 'FaultRst_FT', 'GenerRst_RT', 'GenerRst_FT', 'AirPressureOK_RT', 'AirPressureOK_FT', 'ReleaseAx_RT', 'ReleaseAx_FT', 'ESTOP_RT', 'ESTOP_FT']
    
    _events = list(set(_motor_events + _axis_events + _freqConv_events + _valve_events + _mode_events))

    # create one hot encoding for events
    event_lookup_table = {}
    e_idx = 0
    for x in _events:
        event_lookup_table[x] = e_idx
        e_idx += 1
    numbers={
        'typeEvent': 54,
        'id':75,
        'typeVal_0':400,
        'typeVal_1':400,
        'typeVal_2':400,
        'typeVal_3':400,
    }

    typeVals = [f'typeVal_{x}' for x in range(4)]
    for embeddint_type in ['typeEvent', 'id'] + typeVals:

        cols = [f'D_{i}' for i in range(config.d_model)]
        df = None
        labels = None
        vals = None

        if embeddint_type in typeVals: # typeVal_x
            vals = (torch.arange(numbers[embeddint_type])/10).view(1, numbers[embeddint_type], 1)
            labels = [f'{i/10}' for i in range(numbers[embeddint_type])]

            motor_event  = [f'motor_{x/10}'    for x in range(numbers[embeddint_type])]
            a_event      = [f'axis_{x/10}'     for x in range(numbers[embeddint_type])]
            FC_event     = [f'FC_{x/10}'       for x in range(numbers[embeddint_type])]
            iValve_event = [f'iValve_{x/10}'   for x in range(numbers[embeddint_type])]
            mValve_event = [f'mValve_{x/10}'   for x in range(numbers[embeddint_type])]
            mode_event   = [f'mode_{x/10}'     for x in range(numbers[embeddint_type])]

            type_axis       = [1  for x in range(numbers[embeddint_type])] # np.empty(len(_axis_events))
            type_FC         = [2  for x in range(numbers[embeddint_type])] # np.empty(len(_freqConv_events))
            type_motor      = [3  for x in range(numbers[embeddint_type])] # np.empty(len(_motor_events))
            type_iValve     = [20 for x in range(numbers[embeddint_type])] # np.empty(len(_valve_events))
            type_mValve     = [21 for x in range(numbers[embeddint_type])] # np.empty(len(_valve_events))
            type_mode       = [30 for x in range(numbers[embeddint_type])] # np.empty(len(_mode_events))

            e_axis      = [x/10 for x in range(numbers[embeddint_type])]
            e_fc        = [x/10 for x in range(numbers[embeddint_type])]
            e_motor     = [x/10 for x in range(numbers[embeddint_type])]
            e_iValve    = [x/10 for x in range(numbers[embeddint_type])]
            e_mValve    = [x/10 for x in range(numbers[embeddint_type])]
            e_mode      = [x/10 for x in range(numbers[embeddint_type])]
            
            np.concatenate((type_axis, type_FC, type_motor, type_iValve, type_mValve, type_mode))
            first = type_axis + type_FC + type_motor + type_iValve + type_mValve + type_mode
            second = e_axis + e_fc + e_motor + e_iValve + e_mValve + e_mode
            all = np.array([
                first,
                second
            ])

            t = torch.FloatTensor([[first, second]])
            vals = t.view(1,len(first),2).type(torch.float)

            labels = motor_event + a_event + FC_event + iValve_event + mValve_event + mode_event

        elif embeddint_type == 'typeEvent':
            # list(set(_motor_events + _axis_events + _freqConv_events + _valve_events + _mode_events))
            motor_event = [f'motor_{x}' for x in _motor_events]
            a_event = [f'axis_{x}' for x in _axis_events]
            FC_event = [f'FC_{x}' for x in _freqConv_events]
            iValve_event = [f'iValve_{x}' for x in _valve_events]
            mValve_event = [f'mValve_{x}' for x in _valve_events]
            mode_event = [f'mode_{x}' for x in _mode_events]

            type_axis       = [1  for x in range(len(_axis_events))]# np.empty(len(_axis_events))
            type_FC         = [2  for x in range(len(_freqConv_events))] # np.empty(len(_freqConv_events))
            type_motor      = [3  for x in range(len(_motor_events))] #np.empty(len(_motor_events))
            type_iValve     = [20 for x in range(len(_valve_events))] # np.empty(len(_valve_events))
            type_mValve     = [21 for x in range(len(_valve_events))] # np.empty(len(_valve_events))
            type_mode       = [30 for x in range(len(_mode_events))] # np.empty(len(_mode_events))

            e_axis  = [event_lookup_table[x] for x in _axis_events]
            e_fc    = [event_lookup_table[x] for x in _freqConv_events]
            e_motor = [event_lookup_table[x] for x in _motor_events]
            e_iValve    = [event_lookup_table[x] for x in _valve_events]
            e_mValve    = [event_lookup_table[x] for x in _valve_events]
            e_mode      = [event_lookup_table[x] for x in _mode_events]
            
            np.concatenate((type_axis, type_FC, type_motor, type_iValve, type_mValve, type_mode))
            first = type_axis + type_FC + type_motor + type_iValve + type_mValve + type_mode
            second = e_axis + e_fc + e_motor + e_iValve + e_mValve + e_mode
            all = np.array([
                first,
                second
                # np.concatenate([type_axis, type_FC, type_motor, type_iValve, type_mValve, type_mode]),
                # np.concatenate([e_axis, e_fc, e_motor, e_iValve, e_mValve, e_mode])
            ])

            t = torch.FloatTensor([[first, second]])
            vals = t.view(1,len(first),2).type(torch.int64)

            labels = motor_event + a_event + FC_event + iValve_event + mValve_event + mode_event

            print('TypeEvent')

        elif embeddint_type == 'id':
            vals = torch.arange(numbers[embeddint_type]).view(1,numbers[embeddint_type],1)
            labels = list(motor_names) + list(valve_names) + list(sensor_names) + ['Mode']

        else:
            vals = torch.arange(numbers[embeddint_type]).view(1, numbers[embeddint_type], 1)
    
        df = pd.DataFrame(
            forecaster.spacetimeformer.embedding.getEmbeddingVal(key=embeddint_type, val=vals).detach().numpy(), # features
            columns=cols
        )
        df['LABELS'] = labels

        table = wandb.Table(columns=df.columns.to_list(), data=df.values)
        wandb.log({f'{numbers[embeddint_type]}': table})



    
    # words = tokenizer.vocab.keys()
    # word_embedding = model.embeddings.word_embeddings.weight
    # writer.add_embedding(x_weight,
    #     metadata = words,
    #     tag = f'x-timestamp embedding'
    # )
    # writer.add_embedding(id_weight,
    #     metadata= words,
    #     tag = 'id embedding'
    # )
    # writer.add_embedding(typeEvent_weight,
    # #    metadata= words,
    #     tag = 'typeEvent embedding'
    # )
    # # writer.add_embedding(event_weight,
    # #     metadata= words,
    # #     tag = 'typeEvnt embedding'
    # # )
    # # writer.add_embedding(type_weight,
    # #     metadata= words,
    # #     tag = 'typeEvnt embedding'
    # # )
    # writer.add_embedding(typeVal_0_weight,
    # #    metadata= words,
    #     tag = 'typeVal_0 embedding'
    # )
    # writer.add_embedding(typeVal_1_weight,
    # #    metadata= words,
    #     tag = 'typeVal_1 embedding'
    # )
    # writer.add_embedding(typeVal_2_weight,
    # #    metadata= words,
    #     tag = 'typeVal_2 embedding'
    # )
    # writer.add_embedding(typeVal_3_weight,
    # #    metadata= words,
    #     tag = 'typeVal_3 embedding'
    # )
    # writer.close()
    # tbl_embeddings = TensorBoardLogger("tb_logs", name="spaceTimeFormer")
    # tbl_embeddings #.log_embedding(writer, id_weight, words, "id embedding")

    # callbacks.append(writer)


    

    tb_logger = TensorBoardLogger('tb_logs', 'test')
    logger.watch(forecaster, log="all")

    trainer = pl.Trainer(
        gpus=args.gpus,
        callbacks=callbacks,
        logger=[logger, tb_logger],
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


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
