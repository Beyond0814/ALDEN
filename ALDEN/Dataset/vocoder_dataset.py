#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import os
import librosa
import torch
from torch import Tensor
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, ConcatDataset

sys.path.append('./')
from Script.data_padding import tensor_padding_1d, tensor_padding_2d
from Script.load_dataset_path import Dataset_Info, Label_Mapping
import logging
from torch.utils.data.distributed import DistributedSampler
from Pretrained.FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder


hann_window = {}

def Get_DDP_Dataset(data_config, mode_type=None):
    """
    Get distributed dataset for training or validation.
    
    Args:
        data_config: Data configuration dictionary
        mode_type: Dataset mode ('train' or 'dev')
        
    Returns:
        sampler: Distributed sampler
        data_loader: Data loader
    """
    if mode_type == 'train':
        protocol = Dataset_Info[data_config['data_type']][f'{mode_type}_real_protocol']
    else:
        protocol = Dataset_Info[data_config['data_type']][f'{mode_type}_protocol']
    label_list, name_list = Get_Data_List(dir_meta=protocol, mode_type=mode_type)
    # Pass model paths from config if available
    if 'model_paths' in data_config:
        data_config = {**data_config, **data_config['model_paths']}
    data_set = Get_Data_Speak_dict(config=data_config, name_list=name_list, 
                            label_list=label_list, mode_type=mode_type)
    sampler = DistributedSampler(data_set)
    data_loader = DataLoader(data_set, 
                             batch_size=data_config[f'{mode_type}_bs'],
                             drop_last=data_config['drop_last'],
                             sampler=sampler)
    if dist.get_rank() == 0:
        logging.info('Load {} {} dataset, total {} samples.'.format(data_config['data_type'], mode_type, len(data_set)))
    return sampler, data_loader

def Get_Data_List(dir_meta, mode_type=None):
    label_list, name_list = {}, []
    # domain_label_list = {}
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    if mode_type in ['train', 'dev', 'eval']:
        for line in l_meta:
        # for line in l_meta[:1024]:
            _, utt, _, domain_label, label = line.strip().split(' ')
            name_list.append(utt)
            label_list[utt] = Label_Mapping[label]
        return label_list, name_list
    else:
        for line in l_meta:           
            key = line.strip()
            name_list.append(key)
            label_list[key] = 0
        return label_list, name_list

class Get_Data_Speak_dict(Dataset):
    """
    Dataset class for vocoder training data.
    """
    def __init__(self, config, name_list, label_list, mode_type=None):
        """
        Initialize dataset.
        
        Args:
            config: Configuration dictionary
            name_list: List of utterance names
            label_list: Dictionary mapping utterance names to labels
            mode_type: Dataset mode ('train', 'dev', or 'eval')
        """
        self.mode_type = mode_type
        self.name_list = name_list
        self.label_list = label_list
        self.is_hand_feature = config['is_hand_feature']
        self.voc_dir = Dataset_Info[config['data_type']]['voc4_dir']
        self.voc_type = ['hifigan_', 'hn-sinc-nsf-hifi_', 'hn-sinc-nsf_', 'waveglow_']
        if self.is_hand_feature:
            self.hand_feature_dir = Dataset_Info[config['data_type']]["{}_{}_path".format(config['type_hand_feature'], mode_type)]
        self.cut = config['cut_length']
        # Get sampling rate from config if available
        self.sampling_rate = config.get('sampling_rate', 16000)
        # Get speaker encoder path from config if available
        self.speaker_encoder_path = config.get('speaker_encoder_path', 
                                               'Pretrained/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

        if self.mode_type in  ['dev', 'eval']:
            self.base_wav_dir = Dataset_Info[config['data_type']][f'{self.mode_type}_path']
        elif self.mode_type ==  'train':
            self.base_wav_dir = Dataset_Info[config['data_type']][f'{self.mode_type}_real_path']

        # Initialize speaker encoder
        self.smodel = SpeakerEncoder(self.speaker_encoder_path)
        self.smodel.eval()
        self.smodel.cuda()

    def LoadAudio(self, audio_path):
        """
        Load audio file and extract speaker embedding.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio_tensor: Audio tensor
            speak_emb: Speaker embedding
        """
        # Get sampling rate from config if available
        sampling_rate = getattr(self, 'sampling_rate', 16000)
        audio_ori, fs = librosa.load(audio_path, sr=sampling_rate)
        speak_emb = self.smodel.embed_utterance(audio_ori) # (BS, 256)
        audio_pad = tensor_padding_1d(audio_ori, self.cut)
        audio_tensor = Tensor(audio_pad)
        return audio_tensor, speak_emb
    
    def LoadAudio_Eval(self, audio_path):
        """
        Load audio file for evaluation (without speaker embedding).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio_tensor: Audio tensor
        """
        # Get sampling rate from config if available
        sampling_rate = getattr(self, 'sampling_rate', 16000)
        audio_ori, _ = librosa.load(audio_path, sr=sampling_rate)
        audio_pad = tensor_padding_1d(audio_ori, self.cut)
        audio_tensor = Tensor(audio_pad)
        return audio_tensor 
    
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        label = self.label_list[name]
        if self.is_hand_feature:
            x = torch.load(self.hand_feature_dir + name + '.npy')
            x_pad = tensor_padding_2d(x, self.cut)
            x_tensor = Tensor(x_pad)
            return x_tensor, name, label
        else:
            if self.mode_type in  ['dev', 'eval']:
                audio_ori = self.LoadAudio_Eval(self.base_wav_dir + name + '.flac')
                return audio_ori, name, label
            else:
                audio_ori, ori_speak = self.LoadAudio(self.base_wav_dir + name + '.flac')
                audio_hf, hf_speak = self.LoadAudio(self.voc_dir + self.voc_type[0] + name + '.wav')
                audio_hnhf, hnhf_speak = self.LoadAudio(self.voc_dir + self.voc_type[1] + name + '.wav')
                audio_hnsf, hnsf_speak = self.LoadAudio(self.voc_dir + self.voc_type[2] + name + '.wav')
                audio_wglow, wglow_speak = self.LoadAudio(self.voc_dir + self.voc_type[3] + name + '.wav')
                return audio_ori, audio_hf, audio_hnhf, audio_hnsf, audio_wglow, ori_speak


