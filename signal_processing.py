"""
Modifications in signal processing function
# Original copyright: https://github.com/speechbrain

Reimplemented: Wooseok Shin
"""
import torch

def get_spec_and_phase(signal):
    stft = torch.stft(
        signal,
        512,
        256,
        512,
        torch.hamming_window(512).to(signal.device),
        center=True,
        pad_mode='constant',
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    stft = stft.transpose(2, 1)

    phase = torch.atan2(stft[:, :, :, 1], stft[:, :, :, 0])
    feat = spectral_magnitude(stft, power=0.5)
    feat = torch.log1p(feat)
    return feat, phase

def spectral_magnitude(stft, power=1, log=False, eps=1e-14):
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr

def transform_spec_to_wav(mag, phase, signal_length=None):
    # Combine with enhanced magnitude
    complex_predictions = torch.mul(
        torch.unsqueeze(mag, -1),
        torch.cat(
            (
                torch.unsqueeze(torch.cos(phase), -1),
                torch.unsqueeze(torch.sin(phase), -1),
            ),
            -1,
        ),
    )
    complex_predictions = complex_predictions.permute(0, 2, 1, 3)
    complex_predictions = torch.complex(complex_predictions[..., 0], complex_predictions[..., 1])
    
    pred_wavs = torch.istft(
        input=complex_predictions,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hamming_window(512).to(complex_predictions.device),
        center=True,
        onesided=True,
        normalized=False,
        length=signal_length,
    )
    return pred_wavs