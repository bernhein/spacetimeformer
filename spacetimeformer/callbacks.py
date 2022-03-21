import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
import wandb


class TeacherForcingAnnealCallback(pl.Callback):
    def __init__(self, start, end, epochs):
        assert start >= end
        self.start = start
        self.end = end
        self.epochs = epochs
        self.slope = float((start - end)) / epochs

    def on_validation_epoch_end(self, trainer, model):
        current = model.teacher_forcing_prob
        new_teacher_forcing_prob = max(self.end, current - self.slope)
        model.teacher_forcing_prob = new_teacher_forcing_prob

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--teacher_forcing_start", type=float, default=0.8)
        parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
        parser.add_argument("--teacher_forcing_anneal_epochs", type=int, default=8)


class TimeMaskedLossCallback(pl.Callback):
    def __init__(self, start, end, steps):
        assert start <= end
        self.start = start
        self.end = end
        self.steps = steps
        self.slope = float((end - start)) / steps
        self._time_mask = self.start

    @property
    def time_mask(self):
        return round(self._time_mask)

    def on_train_start(self, trainer, model):
        if model.time_masked_idx is None:
            model.time_masked_idx = self.time_mask

    def on_train_batch_end(self, trainer, model, *args):
        self._time_mask = min(self.end, self._time_mask + self.slope)
        model.time_masked_idx = self.time_mask
        model.log("time_masked_idx", self.time_mask)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--time_mask_start", type=int, default=1)
        parser.add_argument("--time_mask_end", type=int, default=12)
        parser.add_argument("--time_mask_anneal_steps", type=int, default=1000)
        parser.add_argument("--time_mask_loss", action="store_true")


class EmbeddingCallback(pl.Callback):
    def __init__(self, d_model:int=16) -> None:
        
        self.d_model = d_model
        
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

        self.testData = {}

        typeVals = [f'typeVal_{x}' for x in range(4)]
        for embedding_type in ['typeEvent', 'id'] + typeVals:

            cols = [f'D_{i}' for i in range(self.d_model)]
            self.cols = cols
            df = None
            labels = None
            vals = None

            if embedding_type in typeVals: # typeVal_x
                vals = (torch.arange(numbers[embedding_type])/10).view(1, numbers[embedding_type], 1)
                labels = [f'{i/10}' for i in range(numbers[embedding_type])]

                motor_event  = [f'motor_{x/10}'    for x in range(numbers[embedding_type])]
                a_event      = [f'axis_{x/10}'     for x in range(numbers[embedding_type])]
                FC_event     = [f'FC_{x/10}'       for x in range(numbers[embedding_type])]
                iValve_event = [f'iValve_{x/10}'   for x in range(numbers[embedding_type])]
                mValve_event = [f'mValve_{x/10}'   for x in range(numbers[embedding_type])]
                mode_event   = [f'mode_{x/10}'     for x in range(numbers[embedding_type])]

                type_axis       = [1  for x in range(numbers[embedding_type])] # np.empty(len(_axis_events))
                type_FC         = [2  for x in range(numbers[embedding_type])] # np.empty(len(_freqConv_events))
                type_motor      = [3  for x in range(numbers[embedding_type])] # np.empty(len(_motor_events))
                type_iValve     = [20 for x in range(numbers[embedding_type])] # np.empty(len(_valve_events))
                type_mValve     = [21 for x in range(numbers[embedding_type])] # np.empty(len(_valve_events))
                type_mode       = [30 for x in range(numbers[embedding_type])] # np.empty(len(_mode_events))

                e_axis      = [x/10 for x in range(numbers[embedding_type])]
                e_fc        = [x/10 for x in range(numbers[embedding_type])]
                e_motor     = [x/10 for x in range(numbers[embedding_type])]
                e_iValve    = [x/10 for x in range(numbers[embedding_type])]
                e_mValve    = [x/10 for x in range(numbers[embedding_type])]
                e_mode      = [x/10 for x in range(numbers[embedding_type])]
                
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

            elif embedding_type == 'typeEvent':
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

            elif embedding_type == 'id':
                vals = torch.arange(numbers[embedding_type]).view(1,numbers[embedding_type],1)
                labels = list(motor_names) + list(valve_names) + list(sensor_names) + ['Mode']

            else:
                vals = torch.arange(numbers[embedding_type]).view(1, numbers[embedding_type], 1)
        

            self.testData[embedding_type] = {
                'val': vals,
                'labels': labels
            }


    def on_train_batch_end(self, trainer, model, *args):
        
        typeVals = [f'typeVal_{x}' for x in range(4)]
        for embedding_type in ['typeEvent', 'id'] + typeVals:


            df = pd.DataFrame(
                model.spacetimeformer.embedding.getEmbeddingVal(key=embedding_type, val=self[embedding_type].vals).detach().numpy(), # features
                columns=self.cols
            )
            df['LABELS'] = self[embedding_type].labels

            table = wandb.Table(columns=df.columns.to_list(), data=df.values)
            model.logger.log({f'{embedding_type}': table})


    def on_epoch_end(self, trainer, pl_module):
        print('Test')
        