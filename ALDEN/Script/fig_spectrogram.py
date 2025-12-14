
import logging
import sys

import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

def summarize(writer, epoch, images={}):
  for k, v in images.items():
    writer.add_image(k, v, epoch, dataformats='HWC')

def log_audio_to_tensorboard(writer, audio_tensor, sample_rate, global_step, tag="Audio"):
    """
    将音频数据写入TensorBoard。

    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - audio_tensor: 音频数据的Tensor，形状可以是(num_samples,)或(num_samples, num_channels)。
    - sample_rate: 音频的采样率。
    - global_step: 当前的全局步数，用于跟踪训练进程。
    - tag: 用于TensorBoard中显示的标签。
    """
    # 音频数据需要是1D或2D张量，且数据类型为float32
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # 从1D转为2D，形状变为(1, num_samples)
    
    # 确保音频数据的类型是float32
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.to(torch.float32)

    # 写入TensorBoard
    writer.add_audio(tag, audio_tensor, global_step, sample_rate=sample_rate)

def update_loss_tensorboard(writer, global_step, mode, losses):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

def update_tensorboard(writer, global_step, mode, losses, spectrograms, audios):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

    # 处理并记录频谱图
    images = {f'Mel/{mode}_ori': plot_spectrogram_to_numpy(spectrograms['Ori_mel'].data.cpu().numpy()), 
              f'Mel/{mode}_recon': plot_spectrogram_to_numpy(spectrograms['Recon_mel'].data.cpu().numpy())}
    summarize(writer, global_step, images)

    audio = {f'Audio/{mode}_ori': audios['Ori_wav'], 
             f'Audio/{mode}_recon': audios['Recon_wav']}
    for k, v in audio.items():
        log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)

'''def TenBoard(writer, global_step, mode, losses, spectrograms, audios):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

    # 处理并记录频谱图
    # images = {
    #     f'Mel/{mode}_Ori_Fake': plot_spectrogram_to_numpy(spectrograms['Ori_Fake_Mel'].data.cpu().numpy()),
    #     f'Mel/{mode}_Recon_SelfFake': plot_spectrogram_to_numpy(spectrograms['Recon_SelfFake_Mel'].data.cpu().numpy()),
    #     # f'Mel/{mode}_Recon_RevFake': plot_spectrogram_to_numpy(spectrograms['Recon_RevFake_Mel'].data.cpu().numpy()),

    #     f'Mel/{mode}_Ori_Real': plot_spectrogram_to_numpy(spectrograms['Ori_Real_Mel'].data.cpu().numpy()),
    #     f'Mel/{mode}_Recon_SelfReal': plot_spectrogram_to_numpy(spectrograms['Recon_SelfReal_Mel'].data.cpu().numpy()),
    #     # f'Mel/{mode}_Recon_RevReal': plot_spectrogram_to_numpy(spectrograms['Recon_RevReal_Mel'].data.cpu().numpy())
    # }
    # summarize(writer, global_step, images)

    # audio = {f'Audio/{mode}_Ori_Fake': audios['Ori_Fake_Wav'],
    #         f'Audio/{mode}_Recon_SelfFake': audios['Recon_SelfFake_Wav'],
    #         f'Audio/{mode}_Ori_Real': audios['Ori_Real_Wav'],
    #         f'Audio/{mode}_Recon_SelfReal': audios['Recon_SelfReal_Wav']}
    images = {f'Mel/{mode}_Recon_SelfReal': plot_spectrogram_to_numpy(spectrograms['Recon_SelfReal_Mel'].data.cpu().numpy())}
    summarize(writer, global_step, images)

    audio = {f'Audio/{mode}_Recon_SelfReal': audios['Recon_SelfReal_Wav']}
    for k, v in audio.items():
        log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)'''

def TenBoard(writer, global_step, mode, losses, spectrograms, audios):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

    # 处理并记录频谱图
    images = {
        f'Mel/{mode}_Ori_Fake': plot_spectrogram_to_numpy(spectrograms['Ori_Fake_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_SelfFake': plot_spectrogram_to_numpy(spectrograms['Recon_SelfFake_Mel'].data.cpu().numpy()),
        # f'Mel/{mode}_Recon_RevFake': plot_spectrogram_to_numpy(spectrograms['Recon_RevFake_Mel'].data.cpu().numpy()),

        f'Mel/{mode}_Ori_Real': plot_spectrogram_to_numpy(spectrograms['Ori_Real_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_SelfReal': plot_spectrogram_to_numpy(spectrograms['Recon_SelfReal_Mel'].data.cpu().numpy()),
        # f'Mel/{mode}_Recon_RevReal': plot_spectrogram_to_numpy(spectrograms['Recon_RevReal_Mel'].data.cpu().numpy())
    }
    
    summarize(writer, global_step, images)

    audio = {f'Audio/{mode}_Ori_Fake': audios['Ori_Fake_Wav'],
            f'Audio/{mode}_Recon_SelfFake': audios['Recon_SelfFake_Wav'],
            # f'Audio/{mode}_Recon_RevFake': audios['Recon_RevFake_Wav'],
            f'Audio/{mode}_Ori_Real': audios['Ori_Real_Wav'],
            f'Audio/{mode}_Recon_SelfReal': audios['Recon_SelfReal_Wav'],
            # f'Audio/{mode}_Recon_RevReal': audios['Recon_RevReal_Wav']
            }
    
    for k, v in audio.items():
        log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)


def TenBoard_V1(writer, global_step, mode, losses, spectrograms, audios):
    """
    更新 TensorBoard 的日志记录函数。
    
    参数:
    - writer: TensorBoard 的 SummaryWriter 实例。
    - global_step: 当前的全局步数。
    - mode: 当前模式 (如 "Train" 或 "Validation")。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的 Mel 频谱图 tensors。
    - audios: 包含原始和生成的音频数据。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

    # 自动生成并记录频谱图
    images = {}
    for key, spec in spectrograms.items():
        images[f'Mel/{mode}_{key}'] = plot_spectrogram_to_numpy(spec.data.cpu().numpy())
    
    summarize(writer, global_step, images)

    # 自动生成并记录音频
    audio = {}
    for key, wav in audios.items():
        audio[f'Audio/{mode}_{key}'] = wav

    for k, v in audio.items():
        log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)


def TenBoard_noaudio(writer, global_step, mode, losses, spectrograms):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in losses.items():
        writer.add_scalar(f'Loss/{mode}_{k}', v, global_step)

    # 处理并记录频谱图
    images = {
        f'Mel/{mode}_Ori_Fake': plot_spectrogram_to_numpy(spectrograms['Ori_Fake_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_SelfFake': plot_spectrogram_to_numpy(spectrograms['Recon_SelfFake_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_RevFake': plot_spectrogram_to_numpy(spectrograms['Recon_RevFake_Mel'].data.cpu().numpy()),

        f'Mel/{mode}_Ori_Real': plot_spectrogram_to_numpy(spectrograms['Ori_Real_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_SelfReal': plot_spectrogram_to_numpy(spectrograms['Recon_SelfReal_Mel'].data.cpu().numpy()),
        f'Mel/{mode}_Recon_RevReal': plot_spectrogram_to_numpy(spectrograms['Recon_RevReal_Mel'].data.cpu().numpy())
    }
    summarize(writer, global_step, images)

# def update_tensorboard(writer, global_step, losses, spectrograms, audios):
#     """
#     更新TensorBoard的日志记录函数。
    
#     参数:
#     - writer: TensorBoard的SummaryWriter实例。
#     - global_step: 当前的全局步数。
#     - losses: 包含所有损失数据的字典。
#     - spectrograms: 包含原始和生成的Mel频谱图tensors。
#     """
#     # 记录损失数据
#     for k, v in losses.items():
#         writer.add_scalar(f'Loss/{k}', v, global_step)

#     # 处理并记录频谱图
#     images = {'Mel/Train_ori': plot_spectrogram_to_numpy(spectrograms['Y_ori_mel'].data.cpu().numpy()), 
#               'Mel/Train_recon': plot_spectrogram_to_numpy(spectrograms['Y_recon_mel'].data.cpu().numpy())}
#     summarize(writer, global_step, images)

#     audio = {'Audio/Train_ori': audios['Audio_ori'], 
#              'Audio/Train_recon': audios['Audio_recon']}
#     for k, v in audio.items():
#         log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)
    
def update_eval_tensorboard(writer, global_step, eer, spectrograms, audios):
    """
    更新TensorBoard的日志记录函数。
    
    参数:
    - writer: TensorBoard的SummaryWriter实例。
    - global_step: 当前的全局步数。
    - losses: 包含所有损失数据的字典。
    - spectrograms: 包含原始和生成的Mel频谱图tensors。
    """
    # 记录损失数据
    for k, v in eer.items():
        writer.add_scalar(f'Loss/{k}', v, global_step)

    # 处理并记录频谱图
    images = {'Mel/Eval_ori': plot_spectrogram_to_numpy(spectrograms['Y_ori_mel'].data.cpu().numpy()), 
              'Mel/Eval_recon': plot_spectrogram_to_numpy(spectrograms['Y_recon_mel'].data.cpu().numpy())}
    summarize(writer, global_step, images)

    audio = {'Audio/Eval_ori': audios['Audio_ori'], 
             'Audio/Eval_recon': audios['Audio_recon']}
    for k, v in audio.items():
        log_audio_to_tensorboard(writer, v, sample_rate=16000, global_step=global_step, tag=k)











