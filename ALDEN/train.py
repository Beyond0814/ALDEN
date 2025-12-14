"""
Training script for the ALDEN model.
This script handles distributed training with meta-learning components.
"""
import argparse
import copy
import logging
import os
import re
from datetime import datetime
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dataset.vocoder_dataset import Get_DDP_Dataset
from Metric.LogitNorm_loss import LogitNormLoss
from Metric.metric_eer_tdcf import eval_base_numpy_array
from Script.fig_spectrogram import TenBoard_V1, update_loss_tensorboard
from Script.load_colors import print_random_color
from Script.load_mel_and_spec import load_mel_spectrogram
from Script.set_randomseed import Set_RandomSeed
from Trainer.Model import LoadModel, VocoderEncoder, Discriminator, ReconModel
warnings.filterwarnings("ignore")

class Trainer:
    """
    Main trainer class for the ALDEN model.
    Handles model initialization, training loop, and evaluation.
    """
    def __init__(self, cfg, log_dir):
        """
        Initialize the trainer.
        
        Args:
            cfg: Configuration dictionary
            log_dir: Directory to save logs and models
        """
        self.device_id = dist.get_rank() % torch.cuda.device_count()
        self.cfg = cfg
        self.tra_conf = cfg['train_config']
        self.opt_conf = self.tra_conf['optimizer']
        self.log_dir = log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, 'logs'))

        # Model setup
        self._init_model()

        # Optimizer setup
        self._init_opt()

        # Data loaders
        self._init_dataload()

        # Loss functions
        self.cls_loss = LogitNormLoss(0.01)
        self.domain_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.disent_loss = nn.MSELoss()
        
        # Labels
        self._init_label()

        # Training parameters
        self.best_eer = float('inf')
        self.max_no_improve = self.tra_conf.get('early_stopping_patience', 10)
        self.no_improve_epoch = 0

    def _init_model(self):
        """Initialize all model components."""
        # Get model paths from config
        model_paths = self.cfg.get('model_paths', {})
        wavlm_path = model_paths.get('wavlm_path')
        freevc_path = model_paths.get('freevc_path')
        freevc_config = model_paths.get('freevc_config')
        domain_num = self.tra_conf.get('domain_num', 5)
        
        # E_T: WavLM Encoder (WavLM-based feature extractor)
        self.E_T = LoadModel(pretrained=False, model_path=wavlm_path).to(self.device_id)
        self.E_T = torch.nn.parallel.DistributedDataParallel(self.E_T, 
                                                             device_ids=[self.device_id],
                                                             find_unused_parameters=True)
        
        # E_a: Vocoder-agnostic Encoder (extracts artifact features from audio)
        self.E_a = VocoderEncoder(512, 384, 256, 0.7).to(self.device_id)
        self.E_a = torch.nn.parallel.DistributedDataParallel(self.E_a,
                                                             device_ids=[self.device_id],
                                                             find_unused_parameters=True)
        
        # E_d: Vocoder-specific Encoder (extracts domain-specific features)
        self.E_d = VocoderEncoder(512, 384, 256, 0.7).to(self.device_id)
        self.E_d = torch.nn.parallel.DistributedDataParallel(self.E_d, 
                                                             device_ids=[self.device_id],
                                                             find_unused_parameters=True)

        # D: Discriminator (domain discriminator for adversarial training)
        self.D = Discriminator(256, domain_num).to(self.device_id)
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, 
                                                           device_ids=[self.device_id],
                                                           find_unused_parameters=True)    
        
        # C_d: Domain Classifier (classifies vocoder types)
        self.C_d = nn.Linear(256, domain_num).to(self.device_id)
        self.C_d = torch.nn.parallel.DistributedDataParallel(self.C_d, 
                                                             device_ids=[self.device_id],
                                                             find_unused_parameters=True)
        
        # C_a: Forgery Classifier (binary classifier for real/fake detection)
        self.C_a = nn.Linear(256, 2).to(self.device_id)
        self.C_a = torch.nn.parallel.DistributedDataParallel(self.C_a, 
                                                             device_ids=[self.device_id],
                                                             find_unused_parameters=True)
        
        # ReconModel: Reconstruction model for disentanglement
        self.rec_model = ReconModel(wavlm_path=wavlm_path, 
                                   freevc_path=freevc_path, 
                                   freevc_config=freevc_config,
                                   domain_num=domain_num).to(self.device_id)
        self.rec_model = torch.nn.parallel.DistributedDataParallel(self.rec_model, 
                                                                   device_ids=[self.device_id],
                                                                   find_unused_parameters=True)

    def _init_opt(self):
        """Initialize optimizers and schedulers."""
        # Main optimizer for all components
        opt_params = (
            list(self.E_T.parameters()) 
            + list(self.E_a.parameters()) 
            + list(self.E_d.parameters()) 
            + list(self.D.parameters()) 
            + list(self.C_d.parameters()) 
            + list(self.C_a.parameters())
            + list(self.rec_model.parameters()))
        
        self.Det_opt = torch.optim.Adam(opt_params, 
                                        lr=float(self.opt_conf['det_lr']), 
                                        weight_decay=float(self.opt_conf['det_weight_decay']), 
                                        betas=(float(self.opt_conf['det_beta1']),  
                                            float(self.opt_conf['det_beta2'])), 
                                        eps=float(self.opt_conf['det_eps']), 
                                        amsgrad=self.opt_conf['det_amsgrad'])
        self.Det_sched = torch.optim.lr_scheduler.StepLR(self.Det_opt, 
                                                    step_size=self.opt_conf['det_lr_step'], 
                                                    gamma=self.opt_conf['det_lr_gamma']) 
        
        # Meta-optimizer for E_a (Vocoder-agnostic Encoder) and C_a (Forgery Classifier)
        Meta_opt_params = (list(self.E_a.parameters()) + list(self.C_a.parameters()))
        self.Meta_opt = torch.optim.Adam(Meta_opt_params, 
                                        lr=float(self.opt_conf['det_lr']), 
                                        weight_decay=float(self.opt_conf['det_weight_decay']), 
                                        betas=(float(self.opt_conf['det_beta1']), 
                                                float(self.opt_conf['det_beta2'])), 
                                        eps=float(self.opt_conf['det_eps']), 
                                        amsgrad=self.opt_conf['det_amsgrad'])
        self.Meta_sched = torch.optim.lr_scheduler.StepLR(self.Meta_opt, 
                                                    step_size=self.opt_conf['det_lr_step'], 
                                                    gamma=self.opt_conf['det_lr_gamma']) 

    def _init_dataload(self):
        """Initialize data loaders for training and validation."""
        # Pass model paths to data config
        tra_config = self.tra_conf.copy()
        if 'model_paths' in self.cfg:
            tra_config['model_paths'] = self.cfg['model_paths']
            tra_config['speaker_encoder_path'] = self.cfg['model_paths'].get('speaker_encoder_path')
            tra_config['sampling_rate'] = self.tra_conf.get('sampling_rate', 16000)
        self.tra_sampler, self.tra_loader = Get_DDP_Dataset(tra_config, mode_type='train')
        _, self.dev_loader = Get_DDP_Dataset(tra_config, mode_type='dev')

    def _init_label(self):
        """Initialize labels for training."""
        self.loss_keys = ['ForgeryCls', 'Adv', 'DomainCls', 'Disentangle_D', 'Recon', 'Consistency', 
                          'Disentangle_S', 'MtraCls', 'MteCls', 'Total_All']
        self.batch_size = self.tra_conf['train_bs']
        # Classification labels: 1 for real, 0 for fake
        self.cls_label = torch.cat([torch.ones(self.batch_size, dtype=torch.long, device=self.device_id),
                            torch.zeros(self.batch_size*4, dtype=torch.long, device=self.device_id)], dim=0)
        # Domain labels: 0 to domain_num-1
        domain_num = self.tra_conf.get('domain_num', 5)
        self.domain_label = torch.cat([torch.full((self.batch_size,), i, dtype=torch.long, device=self.device_id) 
                                       for i in range(domain_num)], dim=0)

    def _train_step(self, data, loss_totals):
        """
        Perform one training step.
        
        Args:
            data: Training batch data
            loss_totals: Dictionary to accumulate losses
            
        Returns:
            Updated loss_totals, original mel spectrograms, reconstructed mel spectrograms, 
            original waveforms, and reconstructed waveforms
        """
        real, hf, hnnf, hnsf, glow, real_s = [x.to(self.device_id) for x in data]
        tra_data = torch.cat([real, hf, hnnf, hnsf, glow], dim=0)
        tra_s = real_s.repeat(5, 1)
        
        # Forward pass through model components
        # E_T: WavLM Encoder extracts SSL features
        f_T = self.E_T(tra_data)
        # E_d: Vocoder-specific Encoder extracts domain features
        f_d = self.E_d(f_T)
        # C_d: Domain Classifier classifies vocoder types
        logit_d = self.C_d(f_d)
        
        # E_a: Vocoder-agnostic Encoder extracts artifact features
        f_a = self.E_a(f_T)
        # D: Discriminator for adversarial training
        logit_D = self.D(f_a)
        # C_a: Forgery Classifier for real/fake detection
        logit_a = self.C_a(f_a)
        
        # Loss calculations
        loss_cls_a = self.cls_loss(logit_a, self.cls_label)
        loss_cls_d = self.domain_loss(logit_d, self.domain_label)
        loss_adv = self.adv_loss(logit_D, self.domain_label)
        
        loss_dis_D = self._cal_mi_loss(f_a, f_d)
        
        loss_train = loss_cls_a + self.tra_conf['w_adv'] * loss_adv + self.tra_conf['w_cls_d'] * loss_cls_d + \
                     self.tra_conf['w_dis_D'] * loss_dis_D
        
        # ReconModel: Forward pass through reconstruction model
        rec_dict = self.rec_model(tra_data, f_a)
        loss_con = F.l1_loss(rec_dict['f_s'], tra_s)
        loss_dis_S = self._cal_mi_loss(f_a, rec_dict['f_s'])
        loss_rec, ori_mel, rec_mel = self._cal_rec_loss(tra_data, rec_dict)
        loss_rec = self.tra_conf['w_rec']*loss_rec + self.tra_conf['w_dis_S']*loss_dis_S + \
            self.tra_conf['w_con']*loss_con
        
        loss_train += loss_rec  
        self.Det_opt.zero_grad()
        loss_train.backward()
        self.Det_opt.step()
        
        # Meta-training
        self.Meta_opt.zero_grad()
        loss_train, inner_loss, final_loss = self._meta_EaCa(tra_data, loss_train)
        self.Meta_opt.step()
        
        loss_values = [
            loss_cls_a.item(), loss_adv.item(), loss_cls_d.item(), loss_dis_D.item(), 
            loss_rec.item(), loss_con.item(), loss_dis_S.item(), 
            inner_loss.item(), final_loss.item()]

        for key, value in zip(self.loss_keys, loss_values):
            loss_totals[key] += value
        loss_totals['Total_All'] += loss_train.item()
        
        return loss_totals, ori_mel, rec_mel, (tra_data[0], tra_data[self.tra_conf['train_bs']]), (rec_dict['rec_audio'][0].squeeze(1)[0], rec_dict['rec_audio'][1].squeeze(1)[0])

    def _cal_mi_loss(self, f_a, f_s):
        """
        Calculate mutual information loss between artifact features and speaker features.
        
        Args:
            f_a: Artifact features
            f_s: Speaker features
            
        Returns:
            Mutual information loss
        """
        d_a_n = f_a - torch.mean(f_a, 0)[None, :]
        d_s_n = f_s - torch.mean(f_s, 0)[None, :]
        dis_C = d_a_n[:, :, None] * d_s_n[:, None, :]
        target_cr = torch.zeros(dis_C.shape[0], dis_C.shape[1], dis_C.shape[2]).to(self.device_id)
        return self.disent_loss(dis_C, target_cr)

    def _cal_rec_loss(self, tra_data, rec_dict):
        """
        Calculate reconstruction loss.
        
        Args:
            tra_data: Training audio data
            rec_dict: Reconstruction dictionary containing reconstructed audio
            
        Returns:
            Reconstruction loss, original mel spectrograms, and reconstructed mel spectrograms
        """
        tra_s_mel = load_mel_spectrogram(tra_data, self.tra_conf)
        recon_audio = torch.cat(rec_dict['rec_audio'], dim=0)
        recon_mel = load_mel_spectrogram(recon_audio.squeeze(1).float(), self.tra_conf)
        return F.l1_loss(tra_s_mel, recon_mel), (tra_s_mel[0], tra_s_mel[self.tra_conf['train_bs']]), (recon_mel[0], recon_mel[self.tra_conf['train_bs']])
    
    def _meta_EaCa(self, data, loss_train):
        """
        Meta-learning step for E_a (Vocoder-agnostic Encoder) and C_a (Forgery Classifier).
        
        Args:
            data: Training data
            loss_train: Current training loss
            
        Returns:
            Updated loss, inner loss, and final loss
        """
        # Meta-training
        self_param = list(self.E_a.parameters()) + list(self.C_a.parameters())
        for p in self_param:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # Randomly select a test domain index
        domain_num = self.tra_conf.get('domain_num', 5)
        metaTestID = random.randint(0, domain_num - 1)
        # Calculate start and end indices for test domain data
        start, end = metaTestID * self.batch_size, (metaTestID + 1) * self.batch_size
        # Meta-training data
        mtr_data = torch.cat([data[:start], data[end:]], dim=0)
        mtr_label = torch.cat([self.cls_label[:start], self.cls_label[end:]], dim=0)

        # Meta-test data
        mte_data = data[start:end]
        mte_label = self.cls_label[start:end]

        inner_E_a = copy.deepcopy(self.E_a)
        inner_C_a = copy.deepcopy(self.C_a)

        inner_param = list(inner_E_a.parameters()) + list(inner_C_a.parameters())
        inner_opt = torch.optim.Adam(inner_param, lr=float(self.opt_conf['det_lr']))
        
        mtr_f_T = self.E_T(mtr_data)
        mtr_f_a = inner_E_a(mtr_f_T)
        mtr_logit_a = inner_C_a(mtr_f_a)
        inner_loss_cls = self.cls_loss(mtr_logit_a, mtr_label)
        
        inner_opt.zero_grad()
        inner_loss_cls.backward()
        inner_opt.step()
        for p_tgt, p_src in zip(self_param, inner_param):
            if p_src.grad is not None:
                p_tgt.grad.data.add_(p_src.grad.data / 5)
        loss_train += (0.1 * inner_loss_cls).item()
        
        mte_f_T = self.E_T(mte_data)
        mte_f_a = inner_E_a(mte_f_T)
        mte_logit_a = inner_C_a(mte_f_a)
        final_loss_cls = self.cls_loss(mte_logit_a, mte_label)
        
        grad_inner_j = torch.autograd.grad(final_loss_cls, inner_param, allow_unused=True)
        loss_train += (0.1 * final_loss_cls).item()

        for p, g_j in zip(self_param, grad_inner_j):
            if g_j is not None:
                p.grad.data.add_(1.0 * g_j.data / 5)
        return loss_train, inner_loss_cls, final_loss_cls

    def _save_model(self, epoch, eer):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            eer: Equal error rate
        """
        model_path = os.path.join(self.log_dir, f'Det_{epoch}_{eer:.2f}.pt')
        torch.save({
            'E_T': self.E_T.module.state_dict(),
            'E_a': self.E_a.module.state_dict(),
            'E_d': self.E_d.module.state_dict(),
            'D': self.D.module.state_dict(),
            'C_d': self.C_d.module.state_dict(),
            'C_a': self.C_a.module.state_dict(),
            'rec_model': self.rec_model.module.state_dict()
        }, model_path)

    def _tensorboard_log(self, loss_values, ori_mel, rec_mel, ori_wav, rec_wav, epoch):
        """
        Log training information to TensorBoard.
        
        Args:
            loss_values: Dictionary of loss values
            ori_mel: Original mel spectrograms
            rec_mel: Reconstructed mel spectrograms
            ori_wav: Original waveforms
            rec_wav: Reconstructed waveforms
            epoch: Current epoch number
        """
        # Log losses
        logging.info(f"Epoch [{epoch}] | " + " | ".join([f"{k}: {v:.4f}" 
                                                        for k, v in loss_values.items()]))
        update_loss_tensorboard(self.writer, epoch, 'Train', loss_values)
        ori_mel_1, ori_mel_2 = ori_mel
        rec_mel_1, rec_mel_2 = rec_mel
        ori_wav_1, ori_wav_2 = ori_wav
        rec_wav_1, rec_wav_2 = rec_wav
        TenBoard_V1(self.writer, epoch, 'Train', loss_values, {
            'Ori_1': ori_mel_1, 'Rec_S_1': rec_mel_1,
            'Ori_2': ori_mel_2, 'Rec_S_2': rec_mel_2 }, {
                'Ori_Wav_F': ori_wav_1, 'Rec_Wav_S_F': rec_wav_1,
                'Ori_Wav_R': ori_wav_2, 'Rec_Wav_S_R': rec_wav_2, })

    def _eval_step(self, epoch):
        """
        Perform evaluation on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Equal error rate (EER) on validation set
        """
        self.E_T.eval()
        self.E_a.eval()
        self.E_d.eval()
        self.D.eval()
        self.C_d.eval()
        self.C_a.eval()
        self.rec_model.eval()

        score_list, label_list = [], []
        total_loss = 0.0

        for dev_x, _, dev_y in tqdm(self.dev_loader, desc=f'val_{epoch}', ncols=100):
            dev_x, dev_y = dev_x.to(self.device_id), dev_y.to(self.device_id)
            
            # Forward pass: E_T (WavLM Encoder) -> E_a (Vocoder-agnostic Encoder) -> C_a (Forgery Classifier)
            dev_f_T = self.E_T(dev_x)
            dev_f_a = self.E_a(dev_f_T)
            logit = self.C_a(dev_f_a)
            
            batch_score = (logit[:, 1]).data.cpu().numpy().ravel()
            batch_loss = self.cls_loss(logit, dev_y)
            score_list.extend(batch_score.tolist())
            label_list.extend(dev_y.tolist())
            total_loss += batch_loss.item() * dev_x.size(0)

        avg_loss = total_loss / len(self.dev_loader.dataset)
        if self.device_id == 0:
            cm_score = np.array(score_list).ravel()
            label = np.array(label_list)
            dev_eer = 100. * eval_base_numpy_array(cm_score, label)
            
            logging.info(f'Epoch [{epoch}] | DevLoss: {avg_loss:.4f} | DevEER: {dev_eer:.2f}')    
            losses = {'DevLoss': avg_loss, 'DevEER': dev_eer}
            update_loss_tensorboard(self.writer, epoch, 'Dev', losses)
            return dev_eer

    def _update_sched(self):
        """Update learning rate schedulers."""
        self.Det_sched.step()
        self.Meta_sched.step()

    def train(self):
        """Main training loop."""
        for epoch in range(self.tra_conf['num_epoch']):
            self.tra_sampler.set_epoch(epoch)
            loss_totals = {key: 0.0 for key in self.loss_keys}
            
            self.E_T.train()
            self.E_a.train()
            self.E_d.train()
            self.D.train()
            self.C_d.train()
            self.C_a.train()
            self.rec_model.train()

            for data in tqdm(self.tra_loader, desc=f'Train_{epoch}', ncols=100):
                loss_totals_, ori_mel, rec_mel, ori_wav, rec_wav = self._train_step(data, loss_totals)
            self._update_sched()
            avg_loss_total = {k: v / len(self.tra_loader) for k, v in loss_totals_.items()}
            with torch.no_grad():
                dev_eer = self._eval_step(epoch)

            if self.device_id == 0:
                self._tensorboard_log(avg_loss_total, ori_mel, rec_mel, ori_wav, rec_wav, epoch)
                
                if dev_eer < self.best_eer:
                    self.best_eer = dev_eer
                    self.no_improve_epoch = 0
                    self._save_model(epoch, dev_eer)
                else:
                    self.no_improve_epoch += 1

                if self.no_improve_epoch >= self.max_no_improve:
                    logging.info(f"No improvement for {self.max_no_improve} consecutive epochs. Stopping training.")
                    break

parser = argparse.ArgumentParser(description='Starting to train the model.')
parser.add_argument('--config_path', type=str,
                    default='Config/config.yaml',
                    help='path to detector YAML file')
args = parser.parse_args()

def setup():
    """Initialize distributed training."""
    dist.init_process_group('nccl')

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

if __name__ == '__main__':
    try:
        with open(args.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if cfg['mode'] == 'train':
            setup()
        # Define required parameters
        required_params = ['mode', 'model_name', 'random_seed', 'train_config.data_type', 'train_config.optimizer.det_lr', 'train_config.train_bs']

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
        if cfg['mode'] == 'train':
            random_seed = cfg['random_seed']
            data_type = cfg['train_config']['data_type']        
            lr = cfg['train_config']['optimizer']['det_lr']
            batch_size = cfg['train_config']['train_bs']

            folder_name = f"{timestamp}_{mode}_{model_name}_rs{random_seed}_lr{lr}_bs{batch_size}_Tra{data_type}"
        elif cfg['mode'] == 'test':
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
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            
            if cfg['mode'] == 'train':
                if dist.get_rank() == 0:
                    logging.info(f"Logging redirected to '{log_file}'.")
                    # Log the entire config
                    config_str = yaml.dump(cfg)
                    logging.info("Config:\n" + config_str)
            else:
                logging.info(f"Logging redirected to '{log_file}'.")
                # Log the entire config
                config_str = yaml.dump(cfg)
                logging.info("Config:\n" + config_str)                
        else:
            logging.error("Folder creation failed. Unable to redirect logging.")
        
        # Update the configuration with the command line arguments
        cfg.update(vars(args))
        # Create trainer instance
        trainer = Trainer(cfg, log_path)
        
        if cfg['mode'] == 'train':
            print_random_color('Training Phase')
            # Set random seed
            Set_RandomSeed(cfg['random_seed'])
            # Enter training phase
            trainer.train()
    except Exception as e:
        logging.error(f'Error: {e}')
        import traceback 
        logging.error(traceback.format_exc())
        raise e
