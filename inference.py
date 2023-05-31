import os
import time
import numpy as np
import soundfile as sf
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as opj

from model import Generator
from metric_functions.get_metric_scores import get_pesq_parallel, get_csig_parallel, get_cbak_parallel, get_covl_parallel
from signal_processing import get_spec_and_phase, transform_spec_to_wav

fs = 16000


def main():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--weight_path', type=str, default='results/exp1/model', help='Model path')
    parser.add_argument('--weight_file', type=str, default='best_model.pth')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu device')
    parser.add_argument('--save_wavs_path', type=str, default='../pred_wav')
    parser.add_argument('--base_path', type=str, default='data/VCTK_DEMAND', help='Data base path')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--causal', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    
    args = parser.parse_args()

    test_noisy_path = f'{args.base_path}/test/noisy/'
    test_clean_path = f'{args.base_path}/test/clean/'

    model_weight_path = Path(args.weight_path)
    model_weight_path.mkdir(parents=True, exist_ok=True)

    data_paths = {'test_noisy':test_noisy_path, 'test_clean':test_clean_path, 'model_weight':model_weight_path}

    tester = Tester(args, data_paths)
    tester.test()


class Tester:
    def __init__(self, args, data_paths):
        self.args = args
        self.device = torch.device(args.device)
        
        self.test_noisy_path = data_paths['test_noisy']
        self.test_clean_path = data_paths['test_clean']        
        self.model_weight_path = data_paths['model_weight']
        self.generator_test_paths = glob(self.test_clean_path + '/*.wav')

        self.init_model_optim()

        print(f'Test set:{len(self.generator_test_paths)}')
        print(f'Total parameters:{sum(p.numel() for p in self.G.parameters())/10**6:.3f}M')

        os.makedirs(opj(self.args.weight_path, self.args.save_wavs_path), exist_ok=True)

    def init_model_optim(self):
        self.G = Generator(causal=self.args.causal).to(self.device)

    def load_checkpoint(self):
        checkpoint = torch.load(opj(self.model_weight_path, self.args.weight_file), map_location=self.args.device)
        self.epoch = checkpoint['epoch']
        self.G.load_state_dict(checkpoint['generator'], strict=True)
        print(f'---{self.epoch}Epoch loaded: model weigths and optimizer---')


    def test(self):
        self.start_time = time.time()
        # Best validation scores
        self.load_checkpoint()
        print(f'--------Model Test score--------')
        self.evaluation(data_list=self.generator_test_paths, phase='best')

    
    def evaluation(self, data_list, phase='valid'):
        print(f'Evaluation on {phase} data')
        test_enhanced_name = []
        self.G.eval()
        if (phase == 'test') or (phase == 'best'):    # Test path
            clean_path = self.test_clean_path
            noise_path = self.test_noisy_path
        else:                                         # phase == valid
            clean_path = self.train_clean_path
            noise_path = self.train_noisy_path

        with torch.no_grad():
            for i, path in enumerate(tqdm(data_list)):
                wave_name = os.path.basename(path)
                name = Path(wave_name).stem
                suffix = Path(wave_name).suffix

                clean_wav, _ = torchaudio.load(path)
                noise_wav, _ = torchaudio.load(opj(noise_path, wave_name))
                noise_mag, noise_phase = get_spec_and_phase(noise_wav.to(self.device))
                assert noise_mag.size(2) == 257, 'eval'
                assert noise_phase.size(2) == 257, 'eval'

                mask = self.G(noise_mag)
                mask = mask.clamp(min=0.05)

                enh_mag = torch.mul(mask, noise_mag)
                enh_wav = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length=clean_wav.size(1)).detach().cpu().numpy().squeeze()

                enhanced_name=opj(self.args.weight_path, self.args.save_wavs_path, f'{name}{suffix}')                
                sf.write(enhanced_name, enh_wav, fs)
                test_enhanced_name.append(enhanced_name)

        # Calculate True PESQ
        test_PESQ = get_pesq_parallel(clean_path, test_enhanced_name, norm=False)
        test_CSIG = get_csig_parallel(clean_path, test_enhanced_name, norm=False)
        test_CBAK = get_cbak_parallel(clean_path, test_enhanced_name, norm=False)
        test_COVL = get_covl_parallel(clean_path, test_enhanced_name, norm=False)
        test_PESQ, test_CSIG, test_CBAK, test_COVL = np.mean(test_PESQ), np.mean(test_CSIG), np.mean(test_CBAK), np.mean(test_COVL)

        end = time.time()
        print(f'Test PESQ:{test_PESQ:.3f} | Test CSIG:{test_CSIG:.3f} | Test CBAK:{test_CBAK:.3f} | Test COVL:{test_COVL:.3f}')
        print(f'Total process time:{(end-self.start_time)/60:.2f}M')


if __name__ == "__main__":
    main()
