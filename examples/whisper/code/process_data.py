import torch
from onnxruntime_extensions import util

# the flags for pre-processing
USE_ONNX_STFT = True

# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH

class CustomOpStftNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(g, self, n_fft, hop_length, window):
        t_n_fft = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        t_hop_length = g.op("Constant", value_t=torch.tensor(hop_length, dtype=torch.int64))
        t_frame_size = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
        return g.op("ai.onnx.contrib::StftNorm", self, t_n_fft, t_hop_length, window, t_frame_size)

    @staticmethod
    def forward(ctx, audio, n_fft, hop_length, window):
        win_length = window.shape[0]
        stft = torch.stft(
            audio,
            n_fft,
            hop_length,
            win_length,
            window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return stft.abs() ** 2


class WhisperPrePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(N_FFT)
        self.mel_filters = torch.from_numpy(util.mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        stft_norm = CustomOpStftNorm.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)
        magnitudes = stft_norm[:, :, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        spec_min = log_spec.max() - 8.0
        log_spec = torch.maximum(log_spec, spec_min)
        spec_shape = log_spec.shape
        padding_spec = torch.ones(
            spec_shape[0], spec_shape[1], (N_SAMPLES // HOP_LENGTH - spec_shape[2]), dtype=torch.float
        )
        padding_spec *= spec_min
        log_spec = torch.cat((log_spec, padding_spec), dim=2)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec