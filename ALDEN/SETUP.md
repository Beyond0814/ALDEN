# Setup Guide

This guide will help you set up the ALDEN framework for training and evaluation.

## Model Component Naming

The ALDEN framework consists of several key components:

1. **E_T (WavLM Encoder)**: WavLM-based feature extractor
2. **E_a (Vocoder-agnostic Encoder)**: Extracts artifact features from audio
   - **Note**: In pretrained model checkpoints, this component may be named as `E_g` instead of `E_a`. The evaluation script (`test.py`) automatically handles this naming difference when loading checkpoints.
3. **E_d (Vocoder-specific Encoder)**: Extracts domain-specific features
4. **D (Discriminator)**: Domain discriminator for adversarial training
5. **C_d (Domain Classifier)**: Classifies vocoder types
6. **C_a (Forgery Classifier)**: Binary classifier for real/fake detection
7. **ReconModel**: Reconstruction model for disentanglement

## Prerequisites

1. **Python Environment**: Python 3.7 or higher
2. **CUDA**: CUDA 10.2+ (for GPU training)
3. **PyTorch**: PyTorch 1.10+ with CUDA support

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Beyond0814/ALDEN/tree/main/ALDEN
cd ALDEN
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pretrained Models

#### 4.1 WavLM-Large

1. Download WavLM-Large from the [WavLM repository](https://github.com/microsoft/unilm/tree/master/wavlm)
2. Place the model file in: `Pretrained/wavlm/WavLM-Large.pt`

#### 4.2 FreeVC

1. Download FreeVC checkpoint from the [FreeVC repository](https://github.com/OlaWod/FreeVC)
2. Place the checkpoint in: `Pretrained/FreeVC/freevc.pth`
3. Ensure the config file is at: `Pretrained/FreeVC/configs/freevc.json`

#### 4.3 Speaker Encoder

1. Download the speaker encoder checkpoint from the [FreeVC repository](https://github.com/OlaWod/FreeVC)
2. Place it in: `Pretrained/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt`

#### 4.4 ALDEN Pretrained Model (Optional, for Evaluation)

1. Download the pretrained ALDEN model checkpoint from [Google Drive](https://drive.google.com/file/d/1YmdASztbbDwF0PZeg43kOvpdZVVAsg-_/view?usp=drive_link)
2. This checkpoint can be used for evaluation without training from scratch
3. Update the `test_model` path in `Config/config.yaml` to point to the downloaded checkpoint

### 5. Configure Dataset Paths

Edit `Script/load_dataset_path.py` and update the dataset path:

```python
dataset_path = '/path/to/your/datasets'  # Change this to your actual path
```

Ensure your dataset directory structure matches the expected format in `Dataset_Info`.

### 6. Configure Training/Evaluation

Edit `Config/config.yaml`:

1. **Update model paths** (if different from defaults):
   ```yaml
   model_paths:
     wavlm_path: Pretrained/wavlm/WavLM-Large.pt
     freevc_path: Pretrained/FreeVC/freevc.pth
     speaker_encoder_path: Pretrained/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt
   ```

2. **Configure training parameters**:
   - `data_type`: Dataset type (e.g., 'Voc4')
   - `train_bs`: Training batch size (adjust based on GPU memory)
   - `num_epoch`: Number of training epochs
   - `domain_num`: Number of domains (vocoder types)
   - Loss weights: `w_adv`, `w_cls_d`, `w_dis_D`, `w_rec`, `w_dis_S`, `w_con`
   - Optimizer settings: learning rate, weight decay, etc.

3. **For evaluation**, update:
   - `data_type`: Evaluation dataset
   - `test_model`: Path to trained model checkpoint
     - You can download the pretrained ALDEN checkpoint from [Google Drive](https://drive.google.com/file/d/1YmdASztbbDwF0PZeg43kOvpdZVVAsg-_/view?usp=drive_link) for evaluation without training

### 7. Verify Installation

Run a quick test to verify everything is set up correctly:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Issues

### Issue: CUDA out of memory

**Solution**: Reduce `train_bs` in `Config/config.yaml` or use gradient accumulation.

### Issue: Module not found errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Dataset path errors

**Solution**: Verify that:
1. Dataset path in `Script/load_dataset_path.py` is correct
2. Dataset directory structure matches `Dataset_Info` in `load_dataset_path.py`
3. Protocol files exist and are readable

### Issue: Model path errors

**Solution**: Verify that all pretrained models are downloaded and placed in the correct locations as specified in `Config/config.yaml`.

## Next Steps

After setup, you can:
1. Start training: `python train.py --config_path Config/config.yaml`
2. Evaluate a model: `python test.py --config_path Config/config.yaml`

For more details, see the main [README.md](README.md).

