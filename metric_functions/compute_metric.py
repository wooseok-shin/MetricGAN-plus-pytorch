import numpy as np
from pesq import pesq
from .metric_helper import wss, llr, SSNR, trim_mos

def PESQ_normalize(x):
    # Obtained from: https://github.com/nii-yamagishilab/NELE-GAN/blob/master/intel.py (def mapping_PESQ_harvard)
    a = -1.5
    b = 2.5
    y = 1/(1+np.exp(a *(x - b)))
    return y

# def PESQ_normalize(x):
#     y = (x + 0.5) / 5
#     return y

def CMOS_normalize(x):
    y = (x - 1.0) / 4
    return y

def compute_pesq(target_wav, pred_wav, fs, norm=False):
    # Compute the PESQ
    Pesq = pesq(fs, target_wav, pred_wav, 'wb')

    if norm:
        return PESQ_normalize(Pesq)
    else:
        return Pesq

def compute_csig(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = float(trim_mos(Csig))
    
    if norm:
        return CMOS_normalize(Csig)
    else:
        return Csig

def compute_cbak(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, 16000)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Cbak
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)

    if norm:
        return CMOS_normalize(Cbak)
    else:
        return Cbak

def compute_covl(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Covl
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    if norm:
        return CMOS_normalize(Covl)
    else:
        return Covl

