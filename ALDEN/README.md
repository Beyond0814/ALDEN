# ALDEN: Dual-Level Disentanglement with Meta-learning for Generalizable Audio Deepfake Detection


## Installation

### Prerequisites

- Python 3.7+
- CUDA 10.2+ (for GPU support)
- PyTorch 1.10+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Beyond0814/ALDEN.git
cd ALDEN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained models:

   - **WavLM-Large**: Download from [WavLM repository](https://github.com/microsoft/unilm/tree/master/wavlm) and place it in `Pretrained/wavlm/WavLM-Large.pt`
   
   - **FreeVC**: Download FreeVC checkpoint and place it in `Pretrained/FreeVC/freevc.pth`
   
   - **Speaker Encoder**: Download speaker encoder checkpoint and place it in `Pretrained/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt`

4. Configure dataset paths:

   Edit `Script/load_dataset_path.py` and update the `dataset_path` variable to point to your dataset directory:
   ```python
   dataset_path = '/path/to/your/datasets'
   ```

## Configuration

Edit `Config/config.yaml` to configure training and evaluation parameters:

### Key Configuration Parameters

- **model_paths**: Paths to pretrained models (WavLM, FreeVC, Speaker Encoder)
- **train_config**: Training configuration including:
  - `data_type`: Dataset type (e.g., 'Voc4')
  - `num_epoch`: Number of training epochs
  - `train_bs`: Training batch size
  - `domain_num`: Number of domains (vocoder types)
  - Loss weights: `w_adv`, `w_cls_d`, `w_dis_D`, `w_rec`, `w_dis_S`, `w_con`
  - Optimizer settings: learning rate, weight decay, etc.
- **evaluate**: Evaluation configuration including:
  - `data_type`: Evaluation dataset
  - `test_model`: Path to trained model checkpoint

## Usage

### Training

1. Configure your training settings in `Config/config.yaml`

2. Run training:
```bash
# Single GPU
python train.py --config_path Config/config.yaml

# Multi-GPU (using run.sh)
bash run.sh
```

The training script will:
- Create an experiment directory with timestamp
- Save model checkpoints based on validation EER
- Log training progress to TensorBoard

### Evaluation

1. Update `test_model` path in `Config/config.yaml` to point to your trained checkpoint

2. Run evaluation:
```bash
python test.py --config_path Config/config.yaml
```

The evaluation script will:
- Evaluate on multiple datasets (ASVspoof19LA, ASVspoof21DF-eval, WaveFake-en, etc.)
- Generate score files and confusion matrices
- Save evaluation results to Excel files
- Automatically handle checkpoint naming differences (E_a vs E_g) when loading pretrained models

## Project Structure

```
ALDEN/
├── Config/                 # Configuration files
│   └── config.yaml        # Main configuration file
├── Dataset/               # Dataset loading modules
│   ├── vocoder_dataset.py # Training dataset loader
│   └── load_test_dataset.py # Evaluation dataset loader
├── Metric/                # Evaluation metrics
│   ├── LogitNorm_loss.py  # Logit normalization loss
│   └── metric_eer_tdcf.py  # EER and t-DCF calculation
├── Pretrained/            # Pretrained models
│   ├── FreeVC/           # FreeVC vocoder
│   └── wavlm/            # WavLM SSL model
├── Script/                # Utility scripts
│   ├── data_padding.py    # Data padding utilities
│   ├── fig_spectrogram.py # Visualization utilities
│   ├── load_colors.py     # Colored output utilities
│   ├── load_dataset_path.py # Dataset path configuration
│   ├── load_mel_and_spec.py # Mel spectrogram utilities
│   ├── merge_xlsx.py      # Excel file merging
│   ├── save_score.py      # Score saving utilities
│   ├── score_to_class.py  # Score analysis utilities
│   └── set_randomseed.py  # Random seed setting
├── Trainer/               # Model definitions
│   └── Model.py           # Main model architectures
├── train.py               # Training script
├── test.py                # Evaluation script
├── run.sh                 # Multi-GPU training script
└── requirements.txt       # Python dependencies
```

## Model Architecture

The ALDEN framework consists of several key components:

1. **E_T (WavLM Encoder)**: WavLM-based feature extractor
2. **E_a (Vocoder-agnostic Encoder)**: Extracts artifact features from audio
   - **Note**: In pretrained model checkpoints, this component may be named as `E_g` instead of `E_a`. The evaluation script (`test.py`) automatically handles this naming difference when loading checkpoints.
3. **E_d (Vocoder-specific Encoder)**: Extracts domain-specific features
4. **D (Discriminator)**: Domain discriminator for adversarial training
5. **C_d (Domain Classifier)**: Classifies vocoder types
6. **C_a (Forgery Classifier)**: Binary classifier for real/fake detection
7. **ReconModel**: Reconstruction model for disentanglement

## Supported Datasets

The framework supports evaluation on multiple deepfake detection datasets:

- ASVspoof 2019 LA
- ASVspoof 2021 DF-eval
- WaveFake-en
- LibriSeVoc
- CVoiceFake-en
- InTheWild

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{xu2025alden,
author = {Xu, Yuxiong and Li, Bin and Li, Weixiang and Mandelli, Sara and Negroni, Viola and Li, Sheng},
title = {ALDEN: Dual-Level Disentanglement with Meta-learning for Generalizable Audio Deepfake Detection},
year = {2025},
url = {https://doi.org/10.1145/3746027.3754741},
doi = {10.1145/3746027.3754741},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {7277–7286},
numpages = {10},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) for the SSL backbone
- [FreeVC](https://github.com/OlaWod/FreeVC) for One-Shot Voice Conversion model

## Contact

Due to code repository redundancy and multiple historical versions, this codebase has been organized and cleaned with the assistance of Cursor AI to ensure code quality and maintainability.

For questions or issues, please open an issue on GitHub or contact xuyuxiong2022@email.szu.edu.cn.

