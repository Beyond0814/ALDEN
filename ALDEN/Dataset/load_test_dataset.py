import sys
import csv
import os
import librosa
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, ConcatDataset

sys.path.append('./')
from Script.data_padding import tensor_padding_1d, tensor_padding_2d
from Script.load_dataset_path import Dataset_Info, Label_Mapping
# from Script.load_logger import log
import logging

def Get_Eval_Dataset(data_config, dataset_name, mode_type=None):
    # if dataset_name in ['ASVspoof19LA', 'ASVspoof19LA-Trim', 'ASVspoof21LA-all', 'ASVspoof21LA-eval', 'ASVspoof21LA-hidden','ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden', 'InTheWild', 'LibriSeVoc', 'FakeAVCeleb', 'ASVspoof5', 'CVoiceFake-en', 'TIMIT-TTS-clean', 'Codecfake', 'CodecFake', 'ODSS-en', 'Diffusion', 'DFADD']:
    #     protocol = Dataset_Info[dataset_name][f'{mode_type}_protocol']
    #     label_list, name_list = Get_Data_List(DirMeta=protocol, DatasetName=dataset_name, ModeType=mode_type)
    #     data_set = Get_Data_dict(Config=data_config, NameList=name_list, LabelList=label_list, DatasetName=dataset_name, ModeType=mode_type)
    if dataset_name in ['Fake-or-Real-ori', 'Fake-or-Real-norm']:
        label_list, name_list = Get_Data_No_Protocol_List(DirMeta=Dataset_Info[dataset_name][f'{mode_type}_path'])
        data_set = Get_Data_dict(Config=data_config, NameList=name_list, LabelList=label_list, DatasetName=dataset_name, ModeType=mode_type)
    elif dataset_name == 'DECRO-en':
        mode_type_en = 'en_' + mode_type
        en_protocol = Dataset_Info[dataset_name][f'{mode_type_en}_protocol']
        en_label_list, en_name_list = Get_Data_List(DirMeta=en_protocol, DatasetName=dataset_name, ModeType=mode_type)
        en_data_set = Get_Data_dict(Config=data_config, NameList=en_name_list, LabelList=en_label_list, DatasetName=dataset_name, ModeType=mode_type_en)
        data_set = en_data_set
    elif dataset_name == 'DECRO-ch':
        mode_type_ch = 'ch_' + mode_type
        ch_protocol = Dataset_Info[dataset_name][f'{mode_type_ch}_protocol']
        ch_label_list, ch_name_list = Get_Data_List(DirMeta=ch_protocol, DatasetName=dataset_name, ModeType=mode_type)
        ch_data_set = Get_Data_dict(Config=data_config, NameList=ch_name_list, LabelList=ch_label_list, DatasetName=dataset_name, ModeType=mode_type_ch)
        data_set = ch_data_set
    elif dataset_name == 'DECRO-all':
        mode_type_en = 'en_' + mode_type
        en_protocol = Dataset_Info[dataset_name][f'{mode_type_en}_protocol']
        en_label_list, en_name_list = Get_Data_List(DirMeta=en_protocol, DatasetName=dataset_name, ModeType=mode_type)
        en_data_set = Get_Data_dict(Config=data_config, NameList=en_name_list, LabelList=en_label_list, DatasetName=dataset_name, ModeType=mode_type_en)
        mode_type_ch = 'ch_' + mode_type
        ch_protocol = Dataset_Info[dataset_name][f'{mode_type_ch}_protocol']
        ch_label_list, ch_name_list = Get_Data_List(DirMeta=ch_protocol, DatasetName=dataset_name, ModeType=mode_type)
        ch_data_set = Get_Data_dict(Config=data_config, NameList=ch_name_list, LabelList=ch_label_list, DatasetName=dataset_name, ModeType=mode_type_ch)
        data_set = ConcatDataset([en_data_set, ch_data_set])
    elif dataset_name == 'WaveFake-all':
        # WaveFake_all = WaveFake_en + WaveFake_ch
        protocol = Dataset_Info[dataset_name][f'all_label_protocol']
        label_list, name_list = Get_Data_List(DirMeta=protocol, DatasetName=dataset_name, ModeType=mode_type)
        data_set = Get_Data_dict(Config=data_config, NameList=name_list, LabelList=label_list, DatasetName=dataset_name, ModeType=mode_type)
    elif dataset_name == 'WaveFake-en':
        protocol = Dataset_Info[dataset_name][f'en_label_protocol']
        label_list, name_list = Get_Data_List(DirMeta=protocol, DatasetName=dataset_name, ModeType=mode_type)
        data_set = Get_Data_dict(Config=data_config, NameList=name_list, LabelList=label_list, DatasetName=dataset_name, ModeType=mode_type)
    else:
        protocol = Dataset_Info[dataset_name][f'{mode_type}_protocol']
        label_list, name_list = Get_Data_List(DirMeta=protocol, DatasetName=dataset_name, ModeType=mode_type)
        data_set = Get_Data_dict(Config=data_config, NameList=name_list, LabelList=label_list, DatasetName=dataset_name, ModeType=mode_type)

    data_loader = DataLoader(data_set, 
                             batch_size=data_config[f'{mode_type}_bs'], 
                             num_workers=data_config['num_workers'], 
                             shuffle=data_config['shuffle'], 
                             drop_last=data_config['drop_last'])
    logging.info(f'Load {dataset_name} {mode_type} dataset, total {len(data_set)} samples.')
    return data_loader

def Get_Data_No_Protocol_List(DirMeta):
    label_list, name_list = {}, []

    for label in ['Real', 'Fake']:
        root_dir = os.path.join(DirMeta, label)
        # 遍历根目录下的所有文件和子目录
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # 检查文件是否为 wav 格式
                if filename.endswith(".wav"):
                    # 构建文件路径
                    file_path = os.path.join(dirpath, filename)
                    name_list.append(file_path)
                    label_list[file_path] = Label_Mapping[label]    
    return label_list, name_list

def Get_Data_List(DirMeta, DatasetName=None, ModeType=None):
    label_list, name_list = {}, []

    with open(DirMeta, 'r') as f:
        # if DatasetName in ['ASVspoof19LA', 'ASVspoof19LA-Trim', 'ASVspoof21LA-all', 'ASVspoof21LA-eval', 'ASVspoof21LA-hidden','ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden', 'ASVspoof5', 'LibriSeVoc', 'DECRO-en', 'DECRO-ch', 'DECRO-all', 'WaveFake-all', 'WaveFake-en', 'FakeAVCeleb', 'CVoiceFake-en', 'TIMIT-TTS-clean', 'Codecfake', 'CodecFake', 'Diffusion', 'DFADD']:
        #     l_meta = f.readlines()
        # elif DatasetName == 'InTheWild':
        #     l_meta = csv.reader(f)
        #     next(l_meta, None)  # skip header

        if DatasetName == 'InTheWild':
            l_meta = csv.reader(f)
            next(l_meta, None)  # skip header
        else:
            l_meta = f.readlines()

        for line in l_meta:
            if DatasetName in ['ASVspoof19LA', 'ASVspoof19LA-Trim']:
                _, utt, _, _, label = line.strip().split(' ')
            elif DatasetName in ['ASVspoof21LA-all', 'ASVspoof21LA-eval', 'ASVspoof21LA-hidden', 'ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden']:
                # ASVspoof21LA_all = ASVspoof21LA_eval + ASVspoof21LA_hidden + ASVspoof21LA_progress
                data_types = DatasetName.split('-')[-1]
                line_ = line.strip().split(' ')
                utt, label = line_[1], line_[5]
                # utt, label, data_mode = line_[1], line_[5], line_[7]
                # if data_types == 'all':
                #     utt, label = utt_, label_
                # elif data_types == data_mode:
                #     utt, label = utt_, label_
            elif DatasetName == 'InTheWild':
                utt, label = line[0], line[2]
            elif DatasetName in ['LibriSeVoc']:
                utt, label = line.strip().split()
            elif DatasetName in ['DECRO-en', 'DECRO-ch', 'DECRO-all']:
                _, utt, _, _, label = line.strip().split(' ')
            elif DatasetName in ['FakeAVCeleb', 'ODSS-en', 'ODSS-all']:
                utt, _, _, label = line.strip().split(' ')
            elif DatasetName == 'ASVspoof5':
                utt = line.strip() 
            elif DatasetName in ['WaveFake-all', 'WaveFake-en', 'CVoiceFake-en', 'CVoiceFake-all', 
                                 'CVoiceFake-zh', 'CVoiceFake-it', 'CVoiceFake-fr', 'CVoiceFake-de', 'CVoiceFake-en', 'TIMIT-TTS-clean', 'Codecfake', 'CodecFake', 'Diffusion', 'DFADD', 'RFP-original', 'MLAADv3-en', 'VoiceWukong-ori-en', 'VoiceWukong-ori-all']:
                utt, _, label = line.strip().split(' ')
                
            name_list.append(utt)

            if DatasetName not in ['ASVspoof5']:
                label_list[utt] = Label_Mapping[label]
    return label_list, name_list

class Get_Data_dict(Dataset):
    def __init__(self, Config, NameList, LabelList, DatasetName=None, ModeType=None):
        self.mode_type = ModeType
        self.name_list = NameList
        self.label_list = LabelList
        self.DatasetName = DatasetName
        self.is_hand_feature = Config['is_hand_feature']
        self.base_wav_dir = Dataset_Info[self.DatasetName][f'{self.mode_type}_path']
        if self.is_hand_feature:
            self.hand_feature_dir = Dataset_Info[self.DatasetName][f'{Config.type_hand_feature}_{self.mode_type}_path']
        self.cut = Config['cut_length']
        # Get sampling rate from config if available
        self.sampling_rate = Config.get('sampling_rate', 16000)
        # Note: data_augment_type is read but not currently used
        # self.data_augment_type = Config.get('data_augment_type', None)
        self.max_wav_value = 32768.0
        # Get mel spectrogram parameters from config if available
        self.filter_length = Config.get('filter_length', 1280)
        self.n_mel_channels = Config.get('n_mel_channels', 80)
        self.hop_length = Config.get('hop_length', 320)
        self.win_length = Config.get('win_length', 1280)
        self.mel_fmin = Config.get('mel_fmin', 0.0)
        self.mel_fmax = Config.get('mel_fmax', None)
        
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.DatasetName not in ['ASVspoof5']:
            label = self.label_list[name]
        if self.is_hand_feature:
            x = torch.load(self.hand_feature_dir + name + '.npy')
            x_pad = tensor_padding_2d(x, self.cut)
        else:
            if self.DatasetName in ['ASVspoof19LA', 'ASVspoof19LA-Trim', 'ASVspoof21LA-all',
                                    'ASVspoof21LA-eval', 'ASVspoof21LA-hidden',
                                    'ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden', 'ASVspoof5']:
                x, sr = librosa.load(self.base_wav_dir + name + '.flac', sr=self.sampling_rate)
            elif self.DatasetName in ['InTheWild', 'LibriSeVoc', ]:
                x, sr = librosa.load(self.base_wav_dir + name, sr=self.sampling_rate)
            elif self.DatasetName in ['DECRO-en', 'DECRO-ch', 'DECRO-all', 'FakeAVCeleb']:
                x, sr = librosa.load(self.base_wav_dir + name + '.wav', sr=self.sampling_rate)
            elif self.DatasetName in ['WaveFake-all', 'WaveFake-en', 'MLAAD', 'Fake-or-Real-ori',
                                      'Fake-or-Real-norm',
                                      'TIMIT-TTS-clean', 'Codecfake', 'CodecFake', 'ODSS-en', 'ODSS-all', 'Diffusion', 'DFADD', 'VoiceWukong-ori-en', 'VoiceWukong-ori-all', 
                                      'RFP-original', 'MLAADv3-en']:
                x, _ = librosa.load(name, sr=self.sampling_rate)
            elif self.DatasetName in ['CVoiceFake-zh', 'CVoiceFake-it', 'CVoiceFake-fr', 
                         'CVoiceFake-de', 'CVoiceFake-en', 'CVoiceFake-all']:
                x, _ = librosa.load(name, sr=self.sampling_rate)
                # x, sr = torchaudio.load(name)
                # x = x.squeeze(0)
                # x, _ = librosa.load(name, sr=None)
                # x = AudioSegment.from_mp3(name) 
            # x_norm = x / self.max_wav_value
            x_pad = tensor_padding_1d(x, self.cut)
            # mel = mel_spectrogram_torch(x_pad.unsqueeze(0), self.filter_length, self.n_mel_channels, sr, self.hop_length, self.win_length, self.mel_fmin, self.mel_fmax)
            # mel = torch.squeeze(mel, 0)
        utt = Tensor(x_pad)
        if self.DatasetName not in ['ASVspoof5']:
            return utt, name, label
        else:
            return utt, name
        # return utt, mel, name, label
    



        # 19LA
        # _, name, _, domain_label, label = dev_line.strip().split(' ')
        # if domain_label == '-':
        #     domain_label = 'bonafide'
        
        # 21DF
        # domain_label = dev_line.strip().split(' ')[-5]
        
        # CVoiceFake and DFADD and Diffusion and VoiceWukong and WaveFake
        name, domain_label, label = dev_line.strip().split(' ')
        
        # DECRO
        # _, name, _, domain_label, label = dev_line.strip().split(' ')
        # if domain_label == 'asv19':
        #     domain_label = 'bonafide'        
        
        # LibriSeVoc
        # name, label = dev_line.strip().split(' ')
        # domain_label = name.strip().split('_')[-1][:-4]
        
        # ODSS
        # name, domain_label, _, label = dev_line.strip().split(' ')







