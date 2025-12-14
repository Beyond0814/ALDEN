"""
Data preprocessing utilities for audio data padding.
Includes functions for 1D (raw audio) and 2D (feature maps) padding operations.
"""
import numpy as np
import torch


## For raw audio data processing
def zero_padding_1d(wav, ref_len):
    """
    Zero-pad 1D audio signal to reference length.
    
    Args:
        wav: 1D audio signal tensor
        ref_len: Target length
        
    Returns:
        Zero-padded audio signal
    """
    wav_len = wav.shape[0]
    assert ref_len > wav_len
    padd_len = ref_len - wav_len
    return torch.cat((wav, torch.zeros(padd_len, dtype=wav.dtype)), 0)

def repeat_padding_1d(wav, ref_len):
    """
    Repeat-pad 1D audio signal to reference length.
    
    Args:
        wav: 1D audio signal tensor
        ref_len: Target length
        
    Returns:
        Repeat-padded audio signal
    """
    mul = int(np.ceil(ref_len / wav.shape[0]))
    wav = wav.repeat(mul)[:ref_len]
    return wav

def tensor_padding_1d(feat_map, feat_len=64600, padding='repeat'):
    """
    Pad or truncate 1D tensor to specified length.
    
    Args:
        feat_map: 1D feature map (audio signal)
        feat_len: Target length (default: 64600)
        padding: Padding mode, 'zero' or 'repeat' (default: 'repeat')
        
    Returns:
        Padded or truncated tensor
    """
    feat_mat = torch.Tensor(feat_map)
    this_feat_len = feat_mat.shape[0]  # dim
    if this_feat_len > feat_len:
        startp = np.random.randint(this_feat_len - feat_len)
        feat_mat = feat_mat[startp:startp + feat_len]
    elif this_feat_len < feat_len:
        if padding == 'zero':
            feat_mat = zero_padding_1d(feat_mat, feat_len)
        elif padding == 'repeat':
            feat_mat = repeat_padding_1d(feat_mat, feat_len)
        else:
            raise ValueError('Padding should be zero or repeat!')
    return feat_mat

## For audio feature map processing
def padding_2d(spec, ref_len):
    """
    Zero-pad 2D spectrogram to reference length.
    
    Args:
        spec: 2D spectrogram tensor (width, length)
        ref_len: Target length
        
    Returns:
        Zero-padded spectrogram
    """
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding_2d(spec, ref_len):
    """
    Repeat-pad 2D spectrogram to reference length.
    
    Args:
        spec: 2D spectrogram tensor (width, length)
        ref_len: Target length
        
    Returns:
        Repeat-padded spectrogram
    """
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

def tensor_padding_2d(feat_map, feat_len=750, padding='repeat'):
    """
    Pad or truncate 2D tensor to specified length.
    
    Args:
        feat_map: 2D feature map (spectrogram)
        feat_len: Target length (default: 750)
        padding: Padding mode, 'zero' or 'repeat' (default: 'repeat')
        
    Returns:
        Padded or truncated tensor
    """
    feat_mat = torch.Tensor(feat_map)
    this_feat_len = feat_mat.shape[1] # frame, dim
    if this_feat_len > feat_len:
        startp = np.random.randint(this_feat_len - feat_len)
        feat_mat = feat_mat[:, startp:startp+feat_len]
    if this_feat_len < feat_len:
        if padding == 'zero':
            feat_mat = padding_2d(feat_mat, feat_len)
        elif padding == 'repeat':
            feat_mat = repeat_padding_2d(feat_mat, feat_len)
        else:
            raise ValueError('Padding should be zero or repeat!')
    return feat_mat
