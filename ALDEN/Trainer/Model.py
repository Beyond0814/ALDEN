"""
Model definitions for the ALDEN framework.
Includes encoders, discriminators, classifiers, and reconstruction models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from Pretrained.wavlm.WavLM import WavLM, WavLMConfig
from Script.layer_weighted_sum import LayerWeightedSum
from Pretrained.FreeVC.models import SynthesizerTrn
import Pretrained.FreeVC.utils as utils

# Speaker encoder parameters
SpeakerEncoder_para = {
    "c_in": 201,
    "c_h": 256,
    "c_out": 256,
    "kernel_size": 5,
    "bank_size": 8,
    "bank_scale": 1,
    "c_bank": 256,
    "n_conv_blocks": 6,
    "n_dense_blocks": 6,
    "subsample": [1, 2, 1, 2, 1, 2],
    "act": 'relu',
    "dropout_rate": 0.0,
}

class DropoutForMC(nn.Module):
    """
    Dropout layer for Bayesian model.
    The difference is that we do dropout even in eval stage.
    """
    def __init__(self, p, dropout_flag=True):
        """
        Initialize DropoutForMC.
        
        Args:
            p: Dropout probability
            dropout_flag: Whether to apply dropout (default: True)
        """
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        """
        Forward pass with dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with dropout applied
        """
        return F.dropout(x, self.p, training=self.flag)
    
class VocoderEncoder(nn.Module):
    """
    Vocoder encoder for extracting features from SSL features.
    Used as E_a (Vocoder-agnostic Encoder) or E_d (Vocoder-specific Encoder) in the ALDEN framework.
    """
    def __init__(self, in_f=512, hidden_f=384, out_f=256, dp_rate=0.5):
        """
        Initialize VocoderEncoder.
        
        Args:
            in_f: Input feature dimension (default: 512)
            hidden_f: Hidden feature dimension (default: 384)
            out_f: Output feature dimension (default: 256)
            dp_rate: Dropout rate (default: 0.5)
        """
        super(VocoderEncoder, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.hidden_f = hidden_f
        self.m_mcdp_rate = dp_rate
        self.m_mcdp_flag = True

        self.LL = nn.Linear(1024, self.in_f)
        self.frame_level = nn.Sequential(
            nn.Linear(self.in_f, self.hidden_f),
            nn.LeakyReLU(inplace=True),
            DropoutForMC(self.m_mcdp_rate, self.m_mcdp_flag),

            nn.Linear(self.hidden_f, self.out_f),
            nn.LeakyReLU(inplace=True),
            DropoutForMC(self.m_mcdp_rate, self.m_mcdp_flag))

    def forward(self, x_ssl_feat):
        """
        Forward pass of VocoderEncoder.
        
        Args:
            x_ssl_feat: SSL features from backbone
            
        Returns:
            Utterance-level features (mean pooled)
        """
        feat_ = self.frame_level(self.LL(x_ssl_feat))
        feat_utt = feat_.mean(1)
        return feat_utt

class GradReverse(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: return input as-is.
        
        Args:
            ctx: Context object
            x: Input tensor
            
        Returns:
            Input tensor unchanged
        """
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse gradient.
        
        Args:
            ctx: Context object
            grad_output: Gradient from next layer
            
        Returns:
            Negated gradient
        """
        return -grad_output

class Discriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    """
    def __init__(self, feature_dim, domain_classes):
        """
        Initialize Discriminator.
        
        Args:
            feature_dim: Input feature dimension
            domain_classes: Number of domain classes
        """
        super(Discriminator, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )

    def forward(self, di_z):
        """
        Forward pass with gradient reversal.
        
        Args:
            di_z: Input features
            
        Returns:
            Domain classification logits
        """
        y = self.class_classifier(GradReverse.apply(di_z))
        return y

class LoadModel(nn.Module):
    """
    Wrapper for loading WavLM pretrained model.
    """
    def __init__(self, pretrained=False, model_path=None):
        """
        Initialize LoadModel.
        
        Args:
            pretrained: Whether to freeze the backbone (default: False)
            model_path: Path to WavLM model checkpoint (default: None, uses default path)
        """
        super(LoadModel, self).__init__()
        if model_path is None:
            # Default path, should be set in config.yaml
            raise ValueError("WavLM model path must be provided either as argument or in config.yaml")
        cp = torch.load(model_path)
        cfg = WavLMConfig(cp['cfg'])
        self.backbone = WavLM(cfg)
        self.backbone.load_state_dict(cp['model'])
        
        if pretrained:  # If flag is True, then the backbone is frozen
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass: extract features from WavLM.
        
        Args:
            x: Input audio tensor
            
        Returns:
            Extracted features
        """
        return self.backbone.extract_features(x)[0]

def get_act(act: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        act: Activation function name ('lrelu' or 'relu')
        
    Returns:
        Activation function module
    """
    if act == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()

class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        n_dense_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = ConvBank(c_in, c_bank, bank_size, bank_scale, act)
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size),
                )
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub),
                )
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp: Tensor) -> Tensor:
        out = inp
        for idx, (first_layer, second_layer) in enumerate(
            zip(self.first_conv_layers, self.second_conv_layers)
        ):
            y = first_layer(out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[idx] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[idx], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp: Tensor) -> Tensor:
        out = inp
        for first_layer, second_layer in zip(
            self.first_dense_layers, self.second_dense_layers
        ):
            y = first_layer(out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_bank(x)
        out = self.in_conv_layer(out)
        out = self.act(out)
        out = self.conv_blocks(out)
        out = self.pooling_layer(out).squeeze(-1)
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out

class ConvBank(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_bank: int, bank_scale: int, act: str):
        super(ConvBank, self).__init__()
        self.conv_bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d((k // 2, k // 2 - 1 + k % 2)),
                    nn.Conv1d(c_in, c_out, kernel_size=k),
                )
                for k in range(bank_scale, n_bank + 1, bank_scale)
            ]
        )
        self.act = get_act(act)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.act(layer(x)) for layer in self.conv_bank]
        out = torch.cat(outs + [x], dim=1)
        return out

class AffineLayer(nn.Module):
    def __init__(self, c_cond: int, c_h: int):
        super(AffineLayer, self).__init__()
        self.c_h = c_h
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: Tensor, x_cond: Tensor) -> Tensor:
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, :self.c_h], x_cond[:, self.c_h:]
        # mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
        x = self.norm_layer(x)
        x = x * std + mean
        return x

class FreeVC(nn.Module):
    """
    FreeVC model for audio reconstruction.
    """
    def __init__(self, ptfile=None, hpfile=None):
        """
        Initialize FreeVC model.
        
        Args:
            ptfile: Path to FreeVC checkpoint (default: None, uses default path)
            hpfile: Path to FreeVC config file (default: None, uses default path)
        """
        super(FreeVC, self).__init__()
        if ptfile is None:
            ptfile = 'Pretrained/FreeVC/freevc.pth'
        if hpfile is None:
            hpfile = 'Pretrained/FreeVC/configs/freevc.json'
        hps = utils.get_hparams_from_file(hpfile)

        self.net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, 
                                    hps.train.segment_size // hps.data.hop_length, 
                                    **hps.model).cuda()

        _ = utils.load_checkpoint(ptfile, self.net_g, None, True)

        for param in self.net_g.parameters():
            param.requires_grad = False
        
    def get_content(self, c):
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        z, m, logs, mask = self.net_g.enc_p(c, c_lengths)
        return z

    def forward(self, c, spk_emb):
        g = spk_emb.unsqueeze(-1)
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        z_p, m, logs, mask = self.net_g.enc_p(c, c_lengths)
        z = self.net_g.flow(z_p, mask, g=g, reverse=True)
        rec_audio = self.net_g.dec(z * mask, g=g)
        return rec_audio 
    
class ReconModel(nn.Module):
    """
    Reconstruction model for audio reconstruction.
    Combines speaker features and artifact features to reconstruct audio.
    """
    def __init__(self, wavlm_path=None, freevc_path=None, freevc_config=None, domain_num=5):
        """
        Initialize ReconModel.
        
        Args:
            wavlm_path: Path to WavLM model (default: None, uses default path)
            freevc_path: Path to FreeVC checkpoint (default: None, uses default path)
            freevc_config: Path to FreeVC config (default: None, uses default path)
            domain_num: Number of domains (default: 5)
        """
        super(ReconModel, self).__init__()
        self.wavlm_path = wavlm_path
        self.freevc_path = freevc_path
        self.freevc_config = freevc_config
        # Freeze pretrained model as feature extractor
        self.E_F = self.build_backbone(frozen=True)
        # Weighted sum of the last 24 layer features
        self.Sum_layer = LayerWeightedSum()
        self.Weight_s = nn.Parameter(torch.randn(24), requires_grad=True)
        # Speaker feature encoder
        self.E_s = SpeakerEncoder(**SpeakerEncoder_para)
        # Use AdaIN to fuse speaker features and artifact features
        self.affine = AffineLayer(256, 256)
        # FreeVC model for audio reconstruction
        self.Generator = FreeVC(ptfile=freevc_path, hpfile=freevc_config)
        for param in self.Generator.parameters():
            param.requires_grad = False
        self.domain_num = domain_num

    def build_backbone(self, frozen=True):
        """
        Build WavLM backbone model.
        
        Args:
            frozen: Whether to freeze the backbone parameters (default: True)
            
        Returns:
            WavLM backbone model
        """
        if self.wavlm_path is None:
            # Default path, should be set in config.yaml
            raise ValueError("WavLM model path must be provided either as argument or in config.yaml")
        # Load WavLM pretrained model
        checkpoint = torch.load(self.wavlm_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        backbone = WavLM(cfg)
        backbone.load_state_dict(checkpoint['model'])

        if frozen:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone

    def forward(self, x, f_a):
        """
        Forward pass of reconstruction model.
        
        Args:
            x: Input audio tensor
            f_a: Artifact features
            
        Returns:
            Dictionary containing reconstructed audio and speaker features
        """
        # Feature extraction
        _, f_F = self.E_F.extract_features(x, 
                                           output_layer=self.E_F.cfg.encoder_layers, 
                                           ret_layer_results=True)[0]
        # Speaker features
        f_FS = self.Sum_layer(f_F[1:], self.Weight_s)
        f_s = self.E_s(f_FS)
        # Content features
        f_Fc = f_F[-1][0].permute(1, 2, 0)
        
        # Split features by domain
        f_s_ind = f_s.size(0) // self.domain_num
        s_r, s_f = f_s[:f_s_ind], f_s[f_s_ind:]
        a_r, a_f = f_a[:f_s_ind], f_a[f_s_ind:]
        c_r, c_f = f_Fc[:f_s_ind,:,:], f_Fc[f_s_ind:,:,:]
        f_as_11 = self.affine(a_r, s_r)
        f_as_22 = self.affine(a_f, s_f)
        
        with torch.no_grad():
            s_rec_1 = self.Generator(c_r, f_as_11)
            s_rec_2 = self.Generator(c_f, f_as_22)
        # Build the prediction dict for each output 
        rec_dict = {'rec_audio': (s_rec_1, s_rec_2), 'f_s': f_s}
        return rec_dict


