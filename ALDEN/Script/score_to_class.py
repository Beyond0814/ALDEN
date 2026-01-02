'''
文件路径         : /Ensemble_20240627/Script/score_to_class.py
作者名称         : xyx7
文件版本         : V1.0.0
创建日期         : 2024-09-25 21:57:34
简要说明         : 

版权信息         : 2024 by ${Beyond0814}, All Rights Reserved.
'''
import os
import torch
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from Metric.metric_eer_tdcf import compute_eer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Note: ASVspoof2021_eval_score module is required for ASVspoof2021 evaluation
# Uncomment the following line if you have the ASVspoof2021 evaluation script
# from Metric.ASVspoof2021_eval_score import LA21_score_attack, DF21_score_attack

torch.manual_seed(1234)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

Label_Mapping = {'bonafide': 1, 
                 'genuine': 1,
                 'bona-fide': 1,
                 'Real': 1,
                 'real': 1,
                 'Fake': 0,
                 'fake': 0,
                 'spoof': 0}

NumToName = {0: 'Spoof', 1: 'Bonafide'}

# This is for the dataset of class configuration 
Config = {
    'ASVspoof19LA': {'attack': True, 'distribution': True, 'confusion': True},
    'ASVspoof19LA-Trim': {'attack': True, 'distribution': True, 'confusion': True},
    'ASVspoof21LA-all': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'all'},
    'ASVspoof21LA-eval': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'eval'},
    'ASVspoof21LA-hidden': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'hidden'},
    'ASVspoof21DF-all': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'all'},
    'ASVspoof21DF-eval': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'eval'},
    'ASVspoof21DF-hidden': {'attack': True, 'distribution': True, 'confusion': True, 'eval_set': 'hidden'},
    'DECRO-en': {'attack': True, 'distribution': True, 'confusion': True},
    'DECRO-ch': {'attack': True, 'distribution': True, 'confusion': True},
    'DECRO-all': {'attack': True, 'distribution': True, 'confusion': True},
    'LibriSeVoc': {'attack': True, 'distribution': True, 'confusion': True},
    'Fake-or-Real-ori': {'attack': False, 'distribution': True, 'confusion': True},
    'Fake-or-Real-norm': {'attack': False, 'distribution': True, 'confusion': True},
    'InTheWild': {'attack': False, 'distribution': True, 'confusion': True},
    'WaveFake-all': {'attack': True, 'distribution': True, 'confusion': True},
    'WaveFake-en': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-en': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-zh': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-it': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-fr': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-de': {'attack': True, 'distribution': True, 'confusion': True},
    'CVoiceFake-all': {'attack': True, 'distribution': True, 'confusion': True},
    'TIMIT-TTS-en': {'attack': True, 'distribution': True, 'confusion': True},
    'Codecfake': {'attack': True, 'distribution': True, 'confusion': True},
    'Codecfake': {'attack': True, 'distribution': True, 'confusion': True},
    'CodecFake': {'attack': True, 'distribution': True, 'confusion': True},
    'ODSS-en': {'attack': True, 'distribution': True, 'confusion': True},
    'ODSS-all': {'attack': True, 'distribution': True, 'confusion': True},
    'Diffusion': {'attack': True, 'distribution': True, 'confusion': True},
    'DFADD': {'attack': True, 'distribution': True, 'confusion': True},
    'RFP-original': {'attack': True, 'distribution': True, 'confusion': True},
    'MLAADv3-en': {'attack': True, 'distribution': True, 'confusion': True},
    'VoiceWukong-ori-en': {'attack': True, 'distribution': True, 'confusion': True},
    'VoiceWukong-ori-all': {'attack': True, 'distribution': True, 'confusion': True},
}

def LoadSourceFile(data_name):
    if data_name in ['ASVspoof19LA', 'ASVspoof19LA-Trim']:
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/ASVspoof2019LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", sep=" ", header=None)
        eval_df.columns = ["sid", "utt", "phy", "attack", "seflabel"]
    elif data_name == 'DECRO-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/DECRO/en_eval.txt", sep=" ", header=None)
        eval_df.columns = ["sid", "utt","phy", "attack", "seflabel"]
    elif data_name == 'DECRO-ch':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/DECRO/ch_eval.txt", sep=" ", header=None)
        eval_df.columns = ["sid", "utt","phy", "attack", "seflabel"]
    elif data_name == 'DECRO-all':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/DECRO/fusion_eval.txt", sep=" ", header=None)
        eval_df.columns = ["sid", "utt","phy", "attack", "seflabel"]
    elif data_name == 'LibriSeVoc':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/LibriSeVoc/test.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "seflabel"]
        eval_df['attack'] = eval_df['utt'].apply(lambda x: x.split("_")[-1][:-4])
    elif data_name == 'WaveFake-all':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/WaveFake/all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]        
    elif data_name == 'WaveFake-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/WaveFake/english_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-en':
        # eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/en/en_test.txt", sep=" ", header=None)
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/en_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-it':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/it_all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-zh':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/zh-CN_all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-fr':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/fr_all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-de':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/de_all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CVoiceFake-all':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CVoiceFake/all_label_test.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'TIMIT-TTS-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/TIMIT-TTS/clean_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'Codecfake':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/Codecfake/test_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]
    elif data_name == 'CodecFake':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/CodecFake/test_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'ODSS-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/ODSS/all_english_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "language", "seflabel"]   
    elif data_name == 'ODSS-all':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/ODSS/all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "language", "seflabel"]   
    elif data_name == 'Diffusion':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/Diffusion/eval.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'DFADD':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/DFADD/test_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'RFP-original':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/RFP_original/TTS_VC_Real.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'MLAADv3-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/MLAADv3/en_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'VoiceWukong-ori-en':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/VoiceWukong/Alldataset/Alldataset_en_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    elif data_name == 'VoiceWukong-ori-all':
        eval_df = pd.read_csv("/pubdata/xuyx/Datasets/VoiceWukong/Alldataset/Alldataset_all_label.txt", sep=" ", header=None)
        eval_df.columns = ["utt", "attack", "seflabel"]   
    return eval_df

def Score_AtackDistributionConfusion(score_file, dir_path):
    pred_df = pd.read_csv(score_file, sep=" ", header=None)
    pred_df.columns = ["utt", "label", "score"]
    file_name = os.path.basename(score_file)[:-4]
    data_name = os.path.basename(score_file).split('_')[0]
    pred_df['label'] = pred_df['label'].map(NumToName)
    
    if Config[data_name]['attack']:
        if data_name in ['ASVspoof21LA-all', 'ASVspoof21LA-eval', 'ASVspoof21LA-hidden', 
                         'ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden']:
            if not HAS_ASVSPOOF2021:
                print(f"Warning: Skipping {data_name} evaluation - ASVspoof2021_eval_score module not available")
                continue
            print(f"Score attack for {data_name}")
            if data_name in ['ASVspoof21LA-all', 'ASVspoof21LA-eval', 'ASVspoof21LA-hidden']:
                text_buffer = LA21_score_attack(score_file, eval_set=Config[data_name]['eval_set'])
            elif data_name in ['ASVspoof21DF-all', 'ASVspoof21DF-eval', 'ASVspoof21DF-hidden']:
                text_buffer = DF21_score_attack(score_file, eval_set=Config[data_name]['eval_set'])
            
            # Use StringIO to simulate a file-like object for the text buffer
            buffer = StringIO(text_buffer)
            # Create DataFrame
            trans_df = pd.read_csv(buffer, sep='\s+', on_bad_lines='skip')
            xlsx_path = os.path.join(dir_path, 'xlsx', file_name + '_class.xlsx')
            # Create corresponding subdirectory if it doesn't exist
            os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
            trans_df.to_excel(xlsx_path, index=True, header=True)
        else:
            # load eval set
            eval_df = LoadSourceFile(data_name)
            # merge eval_df and pred_df on utt
            res_df = pd.merge(eval_df, pred_df, on='utt')
            # calcuate EER by attack type
            # eer_dict = {}
            results = []
            for attack in res_df['attack'].unique():
                # print(attack)
                spoof_scores = res_df[(res_df['attack'] == attack) & (res_df['label'] == 'Spoof')]['score']
                bonafide_scores = res_df[(res_df['label'] == 'Bonafide')]['score']
                eer_attack, _ = compute_eer(bonafide_scores, spoof_scores)
                # eer_dict[attack] = eer * 100
                # print("{}: EER: {:.4f}%, threshold: {:.4f}".format(attack, eer*100, threshold))
                results.append([attack, "{:.2f}".format(eer_attack * 100)])

            if data_name == 'ASVspoof19LA':
                # 过滤出有效的条目（A07-A19）
                valid_results = [entry for entry in results if entry[0] != '-' and entry[0] >= 'A07' and entry[0] <= 'A19']

                # 按照标识符排序
                sorted_results = sorted(valid_results, key=lambda x: x[0])
                results = sorted_results
            df = pd.DataFrame(results)
            trans_df = df.T


            xlsx_path = os.path.join(dir_path, 'xlsx', file_name + '_class.xlsx')
            # 创建相应的子文件夹（如果不存在）
            os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
            trans_df.to_excel(xlsx_path, index=False, header=False)
            
    if Config[data_name]['distribution']:
        # compute EER
        dist_spoof_scores = pred_df[pred_df['label'] == 'Spoof']['score']
        dist_bonafide_scores = pred_df[pred_df['label'] == 'Bonafide']['score']
        dist_eer, dist_threshold = compute_eer(dist_bonafide_scores, dist_spoof_scores)
        dist_eer = dist_eer * 100 

        plt.figure()
        plt.hist(dist_bonafide_scores, histtype='step', density=True, bins=50, label='Bonafide')
        plt.hist(dist_spoof_scores, histtype='step', density=True, bins=50, label='Spoof')
        plt.plot(dist_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label=f'Threshold: {dist_threshold:.4f}')
        plt.legend()
        plt.xlabel('CM score')
        plt.title(f'{file_name} score Distribution')
        # save_name = file_name + '_Distribution.png'  
        # plt.savefig(os.path.join(os.getcwd(), 'fig/' + save_name), dpi=900)
        # 目标路径
        dist_save_path = os.path.join(dir_path, 'dist', file_name + '_Distri.png')
        # 创建相应的子文件夹（如果不存在）
        os.makedirs(os.path.dirname(dist_save_path), exist_ok=True)
        # 保存图像
        plt.savefig(dist_save_path, dpi=900)
        plt.show()

    if Config[data_name]['confusion']:
        cfs_spoof_scores = pred_df[pred_df['label'] == 'Spoof']['score']
        cfs_bonafide_scores = pred_df[pred_df['label'] == 'Bonafide']['score']
        _, cfs_threshold = compute_eer(cfs_bonafide_scores, cfs_spoof_scores)

        pred_df['pred'] = pred_df['score'].apply(lambda x: 'Spoof' if x < cfs_threshold else 'Bonafide')
        # confusion matrix
        # pred_df['label'] = pred_df['label'].map(NumToName)
        cm = confusion_matrix(pred_df["label"], pred_df["pred"], labels=["Spoof", "Bonafide"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Spoof", "Bonafide"])
        disp.plot(cmap='Greens', values_format='g')
        # plt.title(f'{file_name} Confusion Matrix')
        # 目标路径
        cfs_save_path = os.path.join(dir_path, 'cfs', file_name + '_Cfs.png')
        # 创建相应的子文件夹（如果不存在）
        os.makedirs(os.path.dirname(cfs_save_path), exist_ok=True)
        # 保存图像
        plt.savefig(cfs_save_path, dpi=900)
        plt.show()







