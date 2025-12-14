"""
Test/Evaluation script for the ALDEN model.
This script evaluates the trained model on multiple datasets and generates evaluation reports.
"""
import argparse
import os
import sys
import glob
import logging
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
import torch.nn as nn
from tqdm import tqdm
import warnings

# Set CUDA device order (optional, can be configured via environment variable)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Set this according to your GPU setup

sys.path.append('./')
warnings.filterwarnings("ignore")
from Dataset.load_test_dataset import Get_Eval_Dataset
from Metric.metric_eer_tdcf import eval_base_numpy_array
from Script.load_colors import print_random_color
from Script.merge_xlsx import copy_data
from Script.save_score import save_scores
from Script.score_to_class import Score_AtackDistributionConfusion
from Trainer.Model import LoadModel, VocoderEncoder

class ModelClass:
    """
    Model class for evaluation phase.
    Contains only the models needed for inference: E_T (WavLM Encoder), E_a (Vocoder-agnostic Encoder), and C_a (Forgery Classifier).
    Note: In some pretrained checkpoints, E_a may be named as E_g.
    """
    def __init__(self, wavlm_path=None):
        """
        Initialize ModelClass.
        
        Args:
            wavlm_path: Path to WavLM model (default: None, uses default path)
        """
        # E_T: WavLM Encoder
        self.E_T = LoadModel(pretrained=False, model_path=wavlm_path)
        # E_a: Vocoder-agnostic Encoder (may be named E_g in some checkpoints)
        self.E_a = VocoderEncoder(512, 384, 256, 0.7)
        # C_a: Forgery Classifier
        self.C_a = nn.Linear(256, 2)

def EvalPhase(cfg, log_path):
    """
    Evaluation phase function.
    
    Args:
        cfg: Configuration dictionary
        log_path: Path to save evaluation results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_config = cfg['evaluate']
    
    # Get model paths from config if available
    wavlm_path = None
    if 'model_paths' in cfg:
        wavlm_path = cfg['model_paths'].get('wavlm_path')

    # Initialize model instance
    model = ModelClass(wavlm_path=wavlm_path)
    # Load checkpoint
    checkpoint = torch.load(eval_config['test_model'], map_location=device)
       
    # Load parameters for each component
    # Note: Some checkpoints may use 'E_g' instead of 'E_a' for the Vocoder-agnostic Encoder
    model.E_T.load_state_dict(checkpoint['E_T'])
    if 'E_a' in checkpoint:
        model.E_a.load_state_dict(checkpoint['E_a'])
    elif 'E_g' in checkpoint:
        # Handle checkpoint with E_g naming
        model.E_a.load_state_dict(checkpoint['E_g'])
        logging.info("Loaded E_g from checkpoint (mapped to E_a)")
    else:
        raise KeyError("Neither 'E_a' nor 'E_g' found in checkpoint. Please check checkpoint keys.")
    model.C_a.load_state_dict(checkpoint['C_a'])

    # Move to device and parallelize
    model.E_T = model.E_T.to(device)
    model.E_a = model.E_a.to(device)
    model.C_a = model.C_a.to(device)
    if torch.cuda.device_count() > 1:
        model.E_T = nn.DataParallel(model.E_T)
        model.E_a = nn.DataParallel(model.E_a)
        model.C_a = nn.DataParallel(model.C_a)

    data_dict = {}
    dataset_name_list = ['ASVspoof19LA', 'ASVspoof21DF-eval', 'WaveFake-en', 'LibriSeVoc', 'CVoiceFake-en', 'InTheWild']

    for dataset_name in dataset_name_list:
        eval_loader = Get_Eval_Dataset(eval_config, dataset_name, mode_type='eval')
        # Model evaluation
        score_list, name_list, label_list = [], [], []
        inf_model_name = os.path.basename(eval_config['test_model'])[:-3]
        model.E_T.eval()
        model.E_a.eval()
        model.C_a.eval()
        with torch.no_grad():
            for batch_x, batch_name, batch_label in tqdm(eval_loader, desc=f'{dataset_name}_{inf_model_name}', ncols=100):
                batch_x, batch_label = batch_x.to(device), batch_label.to(device)

                # Forward pass: E_T (WavLM Encoder) -> E_a (Vocoder-agnostic Encoder) -> C_a (Forgery Classifier)
                dev_f_T = model.E_T(batch_x)
                dev_f_a = model.E_a(dev_f_T)
                logit = model.C_a(dev_f_a)
                
                batch_score = (logit[:, 1]).data.cpu().numpy().ravel()
                score_list.extend(batch_score.tolist())
                label_list.extend(batch_label.data.cpu().numpy().ravel())
                name_list.extend(batch_name)
                
        cm_score = np.array(score_list).ravel()
        label = np.array(label_list)
        eval_eer = 100. * eval_base_numpy_array(cm_score, label)
        # Save the score to the txt file
        save_path = os.path.join(log_path, 'txt', '{}_{}_score.txt'.format(dataset_name, inf_model_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_scores(cm_score, name_list, label_list, save_path)
        logging.info(f'{dataset_name} database of Inference EER: {eval_eer:.2f}')
        print_random_color(f'{dataset_name} database of Inference EER: {eval_eer:.2f}')

        data_dict[dataset_name] = [round(eval_eer, 2)]
    
    # Save the evaluation results (EER) to the xlsx file
    eval_all_data = pd.DataFrame(data=data_dict)
    csv_save_path = os.path.join(log_path, 'eval_all_dataset.xlsx')
    eval_all_data.to_excel(csv_save_path, index=False)
    
    # Read all the txt files in the txt folder and calculate the confusion matrix
    txt_path = glob.glob(os.path.join(log_path, 'txt', '*.txt'))
    for txt in txt_path:
        Score_AtackDistributionConfusion(txt, log_path)
    
    # Merge xlsx files
    xlsx_path = glob.glob(os.path.join(log_path, 'xlsx', '*.xlsx'))
    tgt_file = os.path.join(log_path, 'class_merge.xlsx')
    # Iterate through each file and copy data
    for file_name in xlsx_path:
        copy_data(file_name, tgt_file)


parser = argparse.ArgumentParser(description='Starting to evaluate the model.')
parser.add_argument('--config_path', type=str,
                    default='Config/config.yaml',
                    help='path to detector YAML file')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Main entry point for evaluation.
    Loads configuration, sets up logging, and runs evaluation on multiple datasets.
    """
    try:
        with open(args.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Define required parameters
        required_params = ['mode', 'model_name', 'random_seed', 'train_config.data_type', 
                        'train_config.optimizer.det_lr', 'train_config.train_bs']

        # Check for required parameters
        for param in required_params:
            current = cfg
            for key in param.split('.'):
                if key not in current:
                    raise KeyError(f"Missing required parameter: {param}")
                current = current[key]
        
        # Create folder name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Extract necessary values
        mode = cfg['mode']
        model_name = cfg['model_name']
        
        test_model = os.path.basename(cfg['evaluate']['test_model'])[:-3]
        test_model = re.sub(r'_', '-', test_model)
        folder_name = f"{timestamp}_{mode}_{model_name}_{test_model}"

        # Sanitize folder name
        folder_name = re.sub(r'[\\/:*?"<>|]', '_', folder_name)

        # Create folder
        try:
            log_path = os.path.join(os.getcwd(), f'Experiment/{folder_name}/')
            os.makedirs(log_path, exist_ok=True)
            logging.info(f"Experiment folder '{log_path}' created.")
        except Exception as e:
            logging.error(f"Failed to create folder '{log_path}': {e}")

        # Reconfigure logging to write to the folder if creation is successful
        if os.path.exists(log_path):
            log_file = os.path.join(log_path, 'experiment.log')
            # Remove all handlers to avoid duplicate logging
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                                # format='%(asctime)s - %(levelname)s - %(message)s')
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

            logging.info(f"Logging redirected to '{log_file}'.")
            # Log the entire config
            config_str = yaml.dump(cfg)
            logging.info("Config:\n" + config_str)
        else:
            logging.error("Folder creation failed. Unable to redirect logging.")
        
        # Update the configuration with the command line arguments
        cfg.update(vars(args))

        print_random_color('Inference Phase')
        EvalPhase(cfg, log_path)
    except Exception as e:
        logging.error(f'Error: {e}')
        import traceback 
        logging.error(traceback.format_exc())
        raise e





