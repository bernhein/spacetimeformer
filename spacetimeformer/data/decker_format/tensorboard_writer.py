import pandas as pd
import torch



def tensorboardWriter(d_model):
    # words
    motors_src = "/home/hein/MasterThesis/MasterThesis/motors.json"
    valves_src ="/home/hein/MasterThesis/MasterThesis/valves.json"
    motors_data = pd.read_json(motors_src)
    valves_data = pd.read_json(valves_src)
    # get unique file names
    motor_names     = motors_data['LogfileName'].unique()
    valve_names     = valves_data['LogfileName'].unique()
    sensor_names    = motors_data['SensorBMK'].unique()
    
    # unique events
    _motor_events       = ['Start_RT','Start_FT', 'Fault_RT', 'Fault_FT']
    _axis_events        = ['TargetChange_RT', 'TargetChange_FT', 'VeloChange_RT', 'VeloChange_FT', 'ERR_RT', 'ERR_FT', 'Start_RT', 'Start_FT', 'Halt_RT', 'Halt_FT', 'Reset_RT', 'Reset_FT']
    _freqConv_events    = ['Start_RT', 'Start_FT', 'TargetVeloReached_RT', 'TargetVeloReached_FT', 'RelBrake_RT', 'RelBrake_FT', 'CW_RT', 'CW_FT', 'Error_RT', 'Error_FT']
    _valve_events       = ['Input_1_RT', 'Input_1_FT', 'Input_2_RT', 'Input_2_FT', 'Sensor_1_RT', 'Sensor_1_FT', 'Sensor_2_RT', 'Sensor_2_FT']
    _mode_events        = ['ControlVoltage', 'ControlVoltage_FT', 'Auto_RT', 'Auto_FT', 'Manual_RT', 'Manual_FT', 'BasePosBusy_RT', 'BasePosBusy_FT', 'BasePosErr_RT', 'BasePosErr_FT', 'ToolChgByOperatorEnable_RT', 'ToolChgByOperatorEnable_FT', 'ToolChgSemiAuto_RT', 'ToolChgSemiAuto_FT', 'Fault_RT', 'Fault_FT', 'FaultRst_RT', 'FaultRst_FT', 'GenerRst_RT', 'GenerRst_FT', 'AirPressureOK_RT', 'AirPressureOK_FT', 'ReleaseAx_RT', 'ReleaseAx_FT', 'ESTOP_RT', 'ESTOP_FT']
    
    embedding_events = {
        'drillingMotor': _motor_events,
        'axis':          _axis_events,
        'freqConv':      _freqConv_events,
        'valve':         _valve_events,
        'mode':          _mode_events
    }
    _events = list(set(_motor_events + _axis_events + _freqConv_events + _valve_events + _mode_events))
    # create one hot encoding for events
    event_lookup_table = {}
    e_idx = 0
    for x in _events:
        event_lookup_table[x] = e_idx
        e_idx += 1
    emb_counts={
        'typeEvent': 54,
        'id':75,
        'typeVal_0':400,
        'typeVal_1':400,
        'typeVal_2':400,
        'typeVal_3':400,
    }
    # Where to save all the data
    embeddingObservData = {}
    typeVals = [f'typeVal_{x}' for x in range(4)]
    for embedding_type in ['typeEvent', 'id'] + typeVals:
        cols = [f'D_{i}' for i in range(d_model)]
        labels = None
        vals = None
        if embedding_type in typeVals: # 
            vals = (torch.arange(emb_counts[embedding_type])/10).view(1, emb_counts[embedding_type], 1)
            labels = [f'{i/10}' for i in range(emb_counts[embedding_type])]
            motor_event  = [f'motor_{x/10}'    for x in range(emb_counts[embedding_type])]
            a_event      = [f'axis_{x/10}'     for x in range(emb_counts[embedding_type])]
            FC_event     = [f'FC_{x/10}'       for x in range(emb_counts[embedding_type])]
            iValve_event = [f'iValve_{x/10}'   for x in range(emb_counts[embedding_type])]
            mValve_event = [f'mValve_{x/10}'   for x in range(emb_counts[embedding_type])]
            mode_event   = [f'mode_{x/10}'     for x in range(emb_counts[embedding_type])]
            type_axis       = [1  for x in range(emb_counts[embedding_type])] # np.empty(len(_axis_events))
            type_FC         = [2  for x in range(emb_counts[embedding_type])] # np.empty(len(_freqConv_events))
            type_motor      = [3  for x in range(emb_counts[embedding_type])] # np.empty(len(_motor_events))
            type_iValve     = [20 for x in range(emb_counts[embedding_type])] # np.empty(len(_valve_events))
            type_mValve     = [21 for x in range(emb_counts[embedding_type])] # np.empty(len(_valve_events))
            type_mode       = [30 for x in range(emb_counts[embedding_type])] # np.empty(len(_mode_events))
            e_axis      = [x/10 for x in range(emb_counts[embedding_type])]
            e_fc        = [x/10 for x in range(emb_counts[embedding_type])]
            e_motor     = [x/10 for x in range(emb_counts[embedding_type])]
            e_iValve    = [x/10 for x in range(emb_counts[embedding_type])]
            e_mValve    = [x/10 for x in range(emb_counts[embedding_type])]
            e_mode      = [x/10 for x in range(emb_counts[embedding_type])]
            
            first = type_axis + type_FC + type_motor + type_iValve + type_mValve + type_mode
            second = e_axis + e_fc + e_motor + e_iValve + e_mValve + e_mode
            # create tensor
            t = torch.FloatTensor([[first, second]])
            #create embedding observation data
            vals = t.view(1,len(first),2).type(torch.float)
            labels = motor_event + a_event + FC_event + iValve_event + mValve_event + mode_event
        elif embedding_type == 'typeEvent':
            # list(set(_motor_events + _axis_events + _freqConv_events + _valve_events + _mode_events))
            motor_event  = [f'motor_{x}' for x in _motor_events]
            a_event      = [f'axis_{x}' for x in _axis_events]
            FC_event     = [f'FC_{x}' for x in _freqConv_events]
            iValve_event = [f'iValve_{x}' for x in _valve_events]
            mValve_event = [f'mValve_{x}' for x in _valve_events]
            mode_event   = [f'mode_{x}' for x in _mode_events]
            type_axis       = [1  for x in range(len(_axis_events))]# np.empty(len(_axis_events))
            type_FC         = [2  for x in range(len(_freqConv_events))] # np.empty(len(_freqConv_events))
            type_motor      = [3  for x in range(len(_motor_events))] #np.empty(len(_motor_events))
            type_iValve     = [20 for x in range(len(_valve_events))] # np.empty(len(_valve_events))
            type_mValve     = [21 for x in range(len(_valve_events))] # np.empty(len(_valve_events))
            type_mode       = [30 for x in range(len(_mode_events))] # np.empty(len(_mode_events))
            e_axis      = [event_lookup_table[x] for x in _axis_events]
            e_fc        = [event_lookup_table[x] for x in _freqConv_events]
            e_motor     = [event_lookup_table[x] for x in _motor_events]
            e_iValve    = [event_lookup_table[x] for x in _valve_events]
            e_mValve    = [event_lookup_table[x] for x in _valve_events]
            e_mode      = [event_lookup_table[x] for x in _mode_events]
            
            first  = type_axis + type_FC + type_motor + type_iValve + type_mValve + type_mode
            second = e_axis + e_fc + e_motor + e_iValve + e_mValve + e_mode
            # create tensor
            t = torch.FloatTensor([[first, second]])
            # create embedding observation data
            vals    = t.view(1,len(first),2).type(torch.int64)
            labels  = motor_event + a_event + FC_event + iValve_event + mValve_event + mode_event
        elif embedding_type == 'id':
            vals    = torch.arange(emb_counts[embedding_type]).view(1,emb_counts[embedding_type],1)
            labels  = list(motor_names) + list(valve_names) + list(sensor_names) + ['Mode']
        else:
            vals = torch.arange(emb_counts[embedding_type]).view(1, emb_counts[embedding_type], 1)
        # add data
        embeddingObservData[f'{embedding_type}'] = {
            'vals': vals,
            'labels': labels
        }
    
    cols = [f'D_{i}' for i in range(d_model)] 
    return motors_data, valves_data, embedding_events, embeddingObservData, cols