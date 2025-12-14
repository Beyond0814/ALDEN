#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load_dataset_path.py
@Date    :   2024/06/26 15:11:32
@Author  :   Yuxiong Xu
@Email   :   yuxiongxu1996@gmail.com 
@Description:   load dataset path
'''
# Update this path to point to your dataset directory
dataset_path = '/path/to/your/datasets'  # Change this to your actual dataset path

Dataset_Info = {
    'ASVspoof19LA': {
        'train_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt',
        'dev_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'eval_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'train_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_train/flac/',
        'dev_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_dev/flac/',
        'eval_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_eval/flac/',
        'eval_trim_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_Slience/eval/',
        'LFCC_train_path': f'{dataset_path}/ASVspoof2019LA/Feature/LFCC_train/',
        'LFCC_dev_path': f'{dataset_path}/ASVspoof2019LA/Feature/LFCC_dev/',
        'LFCC_eval_path': f'{dataset_path}/ASVspoof2019LA/Feature/LFCC_eval/',
        'CQT_train_path': f'{dataset_path}/ASVspoof2019LA/Feature/CQT_train/',
        'CQT_dev_path': f'{dataset_path}/ASVspoof2019LA/Feature/CQT_dev/',
        'CQT_eval_path': f'{dataset_path}/ASVspoof2019LA/Feature/CQT_eval/',
        'sampling_rate': 16000,
    },
    'ASVspoof19LA-Trim': {
        'eval_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'eval_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_Slience/eval/',
        'sampling_rate': 16000,
    },
    'ASVspoof21LA-all': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/trial_metadata.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof21LA-eval': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/ASVspoof2021_LA_trial_eval.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof21LA-hidden': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/ASVspoof2021_LA_trial_hidden.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_LA_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof21DF-all': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/trial_metadata.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof21DF-eval': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/ASVspoof2021_DF_trial_eval.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof21DF-hidden': {
        'eval_protocol': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/ASVspoof2021_DF_trial_hidden.txt',
        'eval_path': f'{dataset_path}/ASVspoof2021/ASVspoof2021_DF_eval/flac/',
        'sampling_rate': 16000,
    },
    'ASVspoof5': {
        'train_protocol': f'{dataset_path}/ASVspoof5/ASVspoof5.train.metadata.txt',
        'train_mix_protocol': f'{dataset_path}/ASVspoof5/flac_T_mix.txt',
        'train_compress_protocol': f'{dataset_path}/ASVspoof5/flac_T_compress_conv.txt',
        'dev_protocol': f'{dataset_path}/ASVspoof5/ASVspoof5.dev.metadata.txt',
        'eval_prog_protocol': f'{dataset_path}/ASVspoof5/ASVspoof5.track_1.progress.trial.txt',
        'eval_protocol': f'{dataset_path}/ASVspoof5/eval/ASVspoof5.track_1.eval.trial.txt',
        'train_path': f'{dataset_path}/ASVspoof5/flac_T/',
        'train_mix_path': f'{dataset_path}/ASVspoof5/flac_T_mix/',
        'train_compress_path': f'{dataset_path}/ASVspoof5/flac_T_compress_conv/',
        'dev_path': f'{dataset_path}/ASVspoof5/flac_D/',
        'eval_prog_path': f'{dataset_path}/ASVspoof5/flac_E_prog/',
        'eval_path': f'{dataset_path}/ASVspoof5/eval/flac_E_eval/',
        'sampling_rate': 16000,
    },
    'Voc4': {
        'train_real_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train_bonafide.txt',
        # 'train_fake_protocol': f'{dataset_path}/voc.v4/scp/train.lst',
        'train_fake_protocol': f'{dataset_path}/voc.v4/train_protocol.txt', 
        'dev_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'dev_real_protocol': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev_bonafide.txt',
        'dev_fake_protocol': f'{dataset_path}/voc.v4/scp/dev.lst',
        'voc4_dir': f'{dataset_path}/voc.v4/wav/',
        'train_wavlm_content_path': f'{dataset_path}/voc.v4/train_wavlm_content/',
        'train_real_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_train/flac/',
        'dev_path': f'{dataset_path}/ASVspoof2019LA/ASVspoof2019_LA_dev/flac/',
        'sampling_rate': 16000,
    },
    'InTheWild': {
        'eval_protocol': f'{dataset_path}/In_the_Wild/meta.csv',
        'eval_path': f'{dataset_path}/In_the_Wild/eval/',
        'sampling_rate': 16000,
    },
    'LibriSeVoc': {
        'eval_protocol': f'{dataset_path}/LibriSeVoc/test.txt',
        'eval_path': f'{dataset_path}/LibriSeVoc/data/',
        'sampling_rate': 24000,
    },
    'WaveFake-all': {
        'train_protocol': f'{dataset_path}/WaveFake/all_train.txt',
        'dev_protocol': f'{dataset_path}/WaveFake/all_dev.txt',
        'eval_path': f'{dataset_path}/WaveFake/',
        'all_label_protocol': f'{dataset_path}/WaveFake/all_label.txt',
        'sampling_rate': 16000,
    },
    'WaveFake-en': {
        'train_protocol': f'{dataset_path}/WaveFake/en_train.txt',
        'dev_protocol': f'{dataset_path}/WaveFake/en_val.txt',
        'eval_path': f'{dataset_path}/WaveFake/',
        # 'en_label_protocol': f'{dataset_path}/WaveFake/english_label.txt',
        'en_label_protocol': f'{dataset_path}/WaveFake/english_label.txt',
        'sampling_rate': 16000,
    },
    'DECRO-en': {
        'en_eval_protocol': f'{dataset_path}/DECRO/en_eval.txt',
        'en_eval_path': f'{dataset_path}/DECRO/en_eval/',
        'sampling_rate': 16000,
    },
    'DECRO-ch': {
        'ch_eval_protocol': f'{dataset_path}/DECRO/ch_eval.txt',
        'ch_eval_path': f'{dataset_path}/DECRO/ch_eval/',
        'sampling_rate': 16000,
    },
    'DECRO-all': {
        'en_eval_protocol': f'{dataset_path}/DECRO/en_eval.txt',
        'ch_eval_protocol': f'{dataset_path}/DECRO/ch_eval.txt',
        'en_eval_path': f'{dataset_path}/DECRO/en_eval/',
        'ch_eval_path': f'{dataset_path}/DECRO/ch_eval/',
        'sampling_rate': 16000,
    },
    'Fake-or-Real-ori': {
        'eval_protocol': f'{dataset_path}/Fake-or-Real/for-original/testing.txt',
        'eval_path': f'{dataset_path}/Fake-or-Real/for-original/testing/',
        'sampling_rate': 16000,
    },
    'Fake-or-Real-norm': {
        'eval_protocol': f'{dataset_path}/Fake-or-Real/for-norm/testing.txt',
        'eval_path': f'{dataset_path}/Fake-or-Real/for-norm/testing/',
        'sampling_rate': 16000,
    },
    'FakeAVCeleb': {
        'eval_protocol': f'{dataset_path}/FakeAVCeleb_v1_2/audio_test.txt',
        'eval_path': f'{dataset_path}/FakeAVCeleb_v1_2/audio_test/',
        'sampling_rate': 16000,
    },
    'FAD': {
        'train_real': f'{dataset_path}/FAD/train_aishell3.lst',
        'dev_real': f'{dataset_path}/FAD/dev_aishell3.lst',
        'train_fake': f'{dataset_path}/FAD/train_fake.lst',
        'dev_fake': f'{dataset_path}/FAD/dev_fake.lst',
        'train_real_path': f'{dataset_path}/FAD/train/real/aishell3/',
        'dev_real_path': f'{dataset_path}/FAD/dev/real/aishell3/',
        'train_fake_path': f'{dataset_path}/FAD/train/fake/',
        'dev_fake_path': f'{dataset_path}/FAD/dev/fake/',
    },
    'CVoiceFake-en': {
        'train_protocol': f'{dataset_path}/CVoiceFake/en_train.txt',
        'dev_protocol': f'{dataset_path}/CVoiceFake/en_dev.txt',
        # 'eval_protocol': f'{dataset_path}/CVoiceFake/en/en_test.txt',
        'eval_protocol': f'{dataset_path}/CVoiceFake/en_label.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    },
    'CVoiceFake-zh': {
        'eval_protocol': f'{dataset_path}/CVoiceFake/zh-CN_all_label.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    },
    'CVoiceFake-it': {
        'eval_protocol': f'{dataset_path}/CVoiceFake/it_all_label.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    }, 
    'CVoiceFake-fr': {
        'eval_protocol': f'{dataset_path}/CVoiceFake/fr_all_label.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    },    
    'CVoiceFake-de': {
        'eval_protocol': f'{dataset_path}/CVoiceFake/de_all_label.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    },   
    'CVoiceFake-all': {
        'train_protocol': f'{dataset_path}/CVoiceFake/all_train.txt',
        'dev_protocol': f'{dataset_path}/CVoiceFake/all_dev.txt',
        'eval_protocol': f'{dataset_path}/CVoiceFake/all_label_test.txt',
        'eval_path': f'{dataset_path}/CVoiceFake/',
    },
    'TIMIT-TTS-clean': {
        'eval_protocol': f'{dataset_path}/TIMIT-TTS/clean_label.txt',
        'eval_path': f'{dataset_path}/TIMIT-TTS/CLEAN/',
    },
    'Codecfake': {
        'eval_protocol': f'{dataset_path}/Codecfake/test_label.txt',
        'eval_path': f'{dataset_path}/Codecfake/',
    },
    'CodecFake': {
        'eval_protocol': f'{dataset_path}/CodecFake/test_label.txt',
        'eval_path': f'{dataset_path}/CodecFake/',
    },
    'ODSS-en': {
        'eval_protocol': f'{dataset_path}/ODSS/all_english_label.txt',
        'eval_path': f'{dataset_path}/ODSS/',
    },
    'ODSS-all': {
        'eval_protocol': f'{dataset_path}/ODSS/all_label.txt',
        'eval_path': f'{dataset_path}/ODSS/',
    },
    'Diffusion': {
        'eval_protocol': f'{dataset_path}/Diffusion/eval.txt',
        'eval_path': f'{dataset_path}/Diffusion/',
    },
    'DFADD': {
        'eval_protocol': f'{dataset_path}/DFADD/test_label.txt',
        'eval_path': f'{dataset_path}/DFADD/',
    },
    'RFP-original': {
        'eval_protocol': f'{dataset_path}/RFP_original/TTS_VC_Real.txt',
        'eval_path': f'{dataset_path}/RFP_original/',
    },
    'MLAADv3-en': {
        'eval_protocol': f'{dataset_path}/MLAADv3/en_label.txt',
        'eval_path': f'{dataset_path}/MLAADv3/',
    },
    'VoiceWukong-ori-en': {
        'eval_protocol': f'{dataset_path}/VoiceWukong/Alldataset/Alldataset_en_label.txt',
        'eval_path': f'{dataset_path}/VoiceWukong/Alldataset/',
    },
    'VoiceWukong-ori-all': {
        'eval_protocol': f'{dataset_path}/VoiceWukong/Alldataset/Alldataset_all_label.txt',
        'eval_path': f'{dataset_path}/VoiceWukong/Alldataset/',
    },
}

Label_Mapping = {'bonafide': 1, 
                 'genuine': 1,
                 'bona-fide': 1,
                 'Real': 1,
                 'real': 1,
                 'Fake': 0,
                 'fake': 0,
                 'spoof': 0}

Domain_Label_Mapping = {'hifigan': 1, 
                        'hn-sinc-nsf': 2,
                        'hn-sinc-nsf-hifi': 3,
                        'waveglow': 4}

