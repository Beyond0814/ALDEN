"""
Utility functions for computing mel spectrograms and spectrograms from audio signals.
"""
import torch
from librosa.filters import mel as librosa_mel_fn

# Global caches for mel basis and hann windows
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression to input tensor.
    
    Args:
        x: Input tensor
        C: Compression factor (default: 1)
        clip_val: Minimum value for clipping (default: 1e-5)
        
    Returns:
        Compressed tensor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """
    Apply spectral normalization to magnitude spectrogram.
    
    Args:
        magnitudes: Magnitude spectrogram tensor
        
    Returns:
        Normalized spectrogram
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """
    Compute spectrogram from audio signal using STFT.
    
    Args:
        y: Audio signal tensor
        n_fft: FFT window size
        hop_size: Hop length for STFT
        win_size: Window size for STFT
        center: Whether to center the signal (default: False)
        
    Returns:
        Magnitude spectrogram
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Compute mel spectrogram from audio signal.
    
    Args:
        y: Audio signal tensor
        n_fft: FFT window size
        num_mels: Number of mel filter banks
        sampling_rate: Audio sampling rate
        hop_size: Hop length for STFT
        win_size: Window size for STFT
        fmin: Minimum frequency for mel filter bank
        fmax: Maximum frequency for mel filter bank
        center: Whether to center the signal (default: False)
        
    Returns:
        Mel spectrogram
    """
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_mel_spectrogram(audio, train_config):
    """
    Load mel spectrogram from audio signal using training configuration.
    
    Args:
        audio: Audio signal tensor
        train_config: Training configuration dictionary
        
    Returns:
        Mel spectrogram
    """
    audio_mel = mel_spectrogram_torch(audio, train_config['filter_length'], train_config['n_mel_channels'], 
                                      train_config['sampling_rate'], train_config['hop_length'], 
                                      train_config['win_length'], train_config['mel_fmin'], train_config['mel_fmax'])
    return audio_mel

def load_spectrogram(audio, train_config):
    """
    Load spectrogram from audio signal using training configuration.
    
    Args:
        audio: Audio signal tensor
        train_config: Training configuration dictionary
        
    Returns:
        Magnitude spectrogram
    """
    audio_spec = spectrogram_torch(audio, train_config['filter_length'], train_config['hop_length'], 
                                   train_config['win_length'], train_config['center'])
    return audio_spec
