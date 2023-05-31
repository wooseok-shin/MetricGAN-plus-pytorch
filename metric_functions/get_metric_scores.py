import os 
import librosa
from joblib import Parallel, delayed, wrap_non_picklable_objects
from .compute_metric import compute_pesq, compute_csig, compute_cbak, compute_covl

fs = 16000

# PESQ
@wrap_non_picklable_objects
def get_pesq_score(clean_path, enhanced_file, norm):
    clean_wav, enhanced_wav = load_wavs(clean_path, enhanced_file)
    score = compute_pesq(clean_wav, enhanced_wav, fs, norm)
    return score

def get_pesq_parallel(clean_path, enhanced_list, norm=True, n_jobs=16):
    score = Parallel(n_jobs=n_jobs)(delayed(get_pesq_score)(clean_path, enhanced_file, norm) for enhanced_file in enhanced_list)
    return score


# CSIG
@wrap_non_picklable_objects
def get_csig_score(clean_path, enhanced_file, norm):
    clean_wav, enhanced_wav = load_wavs(clean_path, enhanced_file)
    score = compute_csig(clean_wav, enhanced_wav, fs, norm)
    return score

def get_csig_parallel(clean_path, enhanced_list, norm=True, n_jobs=16):
    score = Parallel(n_jobs=n_jobs)(delayed(get_csig_score)(clean_path, enhanced_file, norm) for enhanced_file in enhanced_list)
    return score


# CBAK
@wrap_non_picklable_objects
def get_cbak_score(clean_path, enhanced_file, norm):
    clean_wav, enhanced_wav = load_wavs(clean_path, enhanced_file)
    score = compute_cbak(clean_wav, enhanced_wav, fs, norm)
    return score

def get_cbak_parallel(clean_path, enhanced_list, norm=True, n_jobs=16):
    score = Parallel(n_jobs=n_jobs)(delayed(get_cbak_score)(clean_path, enhanced_file, norm) for enhanced_file in enhanced_list)
    return score


# COVL
@wrap_non_picklable_objects
def get_covl_score(clean_path, enhanced_file, norm):
    clean_wav, enhanced_wav = load_wavs(clean_path, enhanced_file)
    score = compute_covl(clean_wav, enhanced_wav, fs, norm)
    return score

def get_covl_parallel(clean_path, enhanced_list, norm=True, n_jobs=16):
    score = Parallel(n_jobs=n_jobs)(delayed(get_covl_score)(clean_path, enhanced_file, norm) for enhanced_file in enhanced_list)
    return score


def load_wavs(clean_path, enhanced_file):
    name = os.path.basename(enhanced_file)
    if '#' in name:
        wave_name = name.split('#')[0] + '.wav'
    else:
        wave_name = name
    
    clean_wav, sr    = librosa.load(clean_path+wave_name, sr=fs) 
    enhanced_wav, _ = librosa.load(enhanced_file, sr=fs)
    min_length = min(len(clean_wav), len(enhanced_wav))
    
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]

    return clean_wav, enhanced_wav