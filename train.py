"""
PyTorch-style pipeline of MetricGAN+
Original authors: Szu-Wei Fu 2021
Github repo: https://github.com/speechbrain

Reimplemented: Wooseok Shin
Modifications:
 - PyTorch-style implementation
 - Provide other target metrics (CSIG, CBAK, COVL)
"""
import os
import time
import json
import shutil
import random
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as opj

from model import Generator, Discriminator
from dataloader import create_dataloader
from metric_functions.get_metric_scores import get_pesq_parallel, get_csig_parallel, get_cbak_parallel, get_covl_parallel
from signal_processing import get_spec_and_phase, transform_spec_to_wav
fs = 16000

class Trainer:
    def __init__(self, args, data_paths):
        self.args = args
        self.device = torch.device(args.device)
        self.target_metric = args.target_metric
        self.num_samples = args.num_of_sampling
        
        self.train_noisy_path = data_paths['train_noisy']
        self.train_clean_path = data_paths['train_clean']
        self.train_enhan_path = data_paths['train_enhan']
        self.test_noisy_path = data_paths['test_noisy']
        self.test_clean_path = data_paths['test_clean']
        
        self.model_output_path = data_paths['model_output']
        self.log_output_path = data_paths['log_output']

        os.makedirs(self.train_enhan_path, exist_ok=True)
        os.makedirs(opj(self.args.output_path, self.args.exp_name, 'tmp'), exist_ok=True)
        
        self.generator_train_paths = glob(self.train_clean_path + '/*.wav')
        self.generator_valid_paths = []
        for sample in self.generator_train_paths:
            for speaker in args.val_speaker:
                if speaker in sample:
                    self.generator_valid_paths.append(sample)
        self.generator_train_paths = list(set(self.generator_train_paths) - set(self.generator_valid_paths))
        self.generator_test_paths = glob(self.test_clean_path + '/*.wav')
        random.shuffle(self.generator_train_paths)

        self.init_model_optim()
        self.init_target_metric()
        self.init_noisy_score()
        self.best_scores = {'pesq':-0.5, 'csig':0, 'cbak':0, 'covl':0, 'avg':0}

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Train MetricGAN+-{self.target_metric}\n')
            f.write(f'Train set:{len(self.generator_train_paths)}, Valid set:{len(self.generator_valid_paths)}, Test set:{len(self.generator_test_paths)}\n')
            f.write(f'Model parameters:{sum(p.numel() for p in self.G.parameters())/10**6:.3f}M\n')

        shutil.copy('train.py', opj(self.args.output_path, self.args.exp_name, 'train.py'))

        with open(opj(self.args.output_path, self.args.exp_name, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_model_optim(self):
        self.G = Generator(causal=self.args.causal).to(self.device)
        self.D = Discriminator().to(self.device)
        self.MSELoss = nn.MSELoss().to(self.device)

        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.args.lr)
        self.optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.args.lr)
        
    def init_target_metric(self):
        if self.target_metric == 'pesq':
            self.target_metric_func = get_pesq_parallel
        elif self.target_metric == 'csig':
            self.target_metric_func = get_csig_parallel
        elif self.target_metric == 'cbak':
            self.target_metric_func = get_cbak_parallel
        elif self.target_metric == 'covl':
            self.target_metric_func = get_covl_parallel
    
    def init_noisy_score(self):
        self.noisy_set_scores = {}
        Noised_name = glob(self.train_noisy_path + '/*.wav')
        train_score_C_N = self.target_metric_func(self.train_clean_path, Noised_name)
        assert len(Noised_name) == len(train_score_C_N), 'must same length'
        for path, score in tqdm(zip(Noised_name, train_score_C_N)):
            self.noisy_set_scores[path] = score

    def load_checkpoint(self, ver='latest'):
        checkpoint = torch.load(opj(self.model_output_path, f'{ver}_model.pth'))
        self.epoch = checkpoint['epoch']
        self.G.load_state_dict(checkpoint['generator'])
        self.D.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['g_optimizer'])
        self.optimizer_d.load_state_dict(checkpoint['d_optimizer'])
        if ver == 'best':
            print(f'---{self.epoch}Epoch loaded: model weigths and optimizer---')
        else:
            print(f'---load latest model weigths and optimizer---')

    def train(self):
        start_time = time.time()
        self.epoch = 1
        self.historical_set = []
        for epoch in np.arange(self.epoch, self.args.epochs+1):
            self.epoch = epoch
            print(f'{epoch}Epoch start')
            
            # random sample some training data  
            random.shuffle(self.generator_train_paths)
            genloader = create_dataloader(self.generator_train_paths[0 : round(1*self.args.num_of_sampling)], self.train_noisy_path,
                                          num_target_metric=1, loader='G', batch_size=self.args.batch_size, num_workers=self.args.num_workers)
            self.train_one_epoch(genloader)

        end_time = time.time()
        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Total training time:{(end_time-start_time)/60:.2f}Minute\n')

        # Best validation scores
        self.load_checkpoint('best')
        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'--------Model Best score--------')
        self.evaluation(data_list=self.generator_test_paths, phase='best')

    def train_one_epoch(self, genloader):
        if self.epoch >= 2:
            self.train_generator(genloader)

        if self.epoch >= self.args.skip_val_epoch:
            if self.epoch % self.args.eval_per_epoch == 0:
                self.evaluation(data_list=self.generator_test_paths[0:self.args.num_of_val_sample], phase='test')

        self.train_discriminator()  
        
    def train_generator(self, data_loader):
        self.G.train()
        print('Generator training phase')
        for clean_mag, noise_mag, target, length in tqdm(data_loader):          
            clean_mag = clean_mag.to(self.device)  # [B, T, F]
            noise_mag = noise_mag.to(self.device)  # [B, T, F]
            target = target.to(self.device)

            mask = self.G(noise_mag, length)
            mask = mask.clamp(min=0.05)
            enh_mag = torch.mul(mask, noise_mag).unsqueeze(1)

            ref_mag = clean_mag.detach().unsqueeze(1)
            d_inputs = torch.cat([ref_mag, enh_mag], dim=1)
            assert noise_mag.size(2) == 257, 'gen'
            assert clean_mag.size(2) == 257, 'gen'

            score = self.D(d_inputs)
            loss = self.MSELoss(score, target)
            self.optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5.0)
            self.optimizer_g.step()
    
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

                enhanced_name=opj(self.args.output_path, self.args.exp_name, 'tmp', f'{name}#{self.epoch}{suffix}')
                
                sf.write(enhanced_name, enh_wav, fs)
                test_enhanced_name.append(enhanced_name)

        # Calculate True PESQ
        test_PESQ = get_pesq_parallel(clean_path, test_enhanced_name, norm=False)
        test_CSIG = get_csig_parallel(clean_path, test_enhanced_name, norm=False)
        test_CBAK = get_cbak_parallel(clean_path, test_enhanced_name, norm=False)
        test_COVL = get_covl_parallel(clean_path, test_enhanced_name, norm=False)
        test_PESQ, test_CSIG, test_CBAK, test_COVL = np.mean(test_PESQ), np.mean(test_CSIG), np.mean(test_CBAK), np.mean(test_COVL)

        test_scores = {'pesq':test_PESQ, 'csig':test_CSIG, 'cbak':test_CBAK, 'covl':test_COVL}

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Epoch:{self.epoch} | Test PESQ:{test_PESQ:.3f} | Test CSIG:{test_CSIG:.3f} | Test CBAK:{test_CBAK:.3f} | Test COVL:{test_COVL:.3f}\n')

        if (phase == 'valid') or (phase == 'test'):    # Test path
            checkpoint = {  'epoch':self.epoch,
                            'stats': test_scores,
                            'generator': self.G.state_dict(),
                            'discriminator': self.D.state_dict(),
                            'g_optimizer': self.optimizer_g.state_dict(),
                            'd_optimizer': self.optimizer_d.state_dict(),
                            }

            if test_scores['pesq'] >= self.best_scores['pesq']:
                print('----------------------------------------')
                print('-----------------SAVE-------------------')
                self.best_scores = test_scores

                torch.save(checkpoint, opj(self.model_output_path, f'best_model.pth'))
                print('----------------------------------------')
            torch.save(checkpoint, opj(self.model_output_path, f'latest_model.pth'))

    def get_score(self):
        print('Get scores for discriminator training')
        D_paths = self.generator_train_paths[0:self.args.num_of_sampling]

        Enhanced_name = []
        Noised_name = []

        self.G.eval()
        with torch.no_grad():
            for path in tqdm(D_paths):
                wave_name = os.path.basename(path)
                name = Path(wave_name).stem
                suffix = Path(wave_name).suffix
                
                clean_wav, _ = torchaudio.load(path)
                noise_wav, _ = torchaudio.load(self.train_noisy_path+wave_name)
                noise_mag, noise_phase = get_spec_and_phase(noise_wav.to(self.device))

                mask = self.G(noise_mag)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag)
                enh_wav = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length=clean_wav.size(1)).detach().cpu().numpy().squeeze()

                assert noise_mag.size(2) == 257, 'get_score'
                assert noise_phase.size(2) == 257, 'get_score'
                assert enh_mag.size(2) == 257, 'get_score'

                enhanced_name=opj(self.train_enhan_path, name+'#'+str(self.epoch)+suffix)
                sf.write(enhanced_name, enh_wav, fs)
                Enhanced_name.append(enhanced_name)
                Noised_name.append(self.train_noisy_path+wave_name)
    
        # Calculate true score
        train_scores = []
        train_score_C_E = self.target_metric_func(self.train_clean_path, Enhanced_name)
        train_score_C_N = [self.noisy_set_scores[path] for path in Noised_name]
        train_score_C_C = [1.0] * self.num_samples
        train_scores.append(train_score_C_E)
        train_scores.append(train_score_C_N)
        train_scores.append(train_score_C_C)

        train_scores.append(Enhanced_name)
        current_set = np.array(train_scores).T.tolist()   # [num_sampling, 4]

        # Training for current list
        random.shuffle(current_set)
        return current_set
    
    def subtrain_discriminator(self, data_loader, hist=False):
        for i, (enhance_mag, noisy_mag, clean_mag, target) in enumerate(tqdm(data_loader)):
            target = target.to(self.device)

            if hist:
                inputs = torch.cat([clean_mag, enhance_mag], dim=1).to(self.device)  # (clean, enhanced)
                score = self.D(inputs)    
                loss = self.MSELoss(score, target[:, 0])             # target: [C-E score, C-N score, C-C score]
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                self.optimizer_d.step()
            else:
                for j, target_mag in enumerate([enhance_mag, noisy_mag, clean_mag]):
                    inputs = torch.cat([clean_mag, target_mag], dim=1).to(self.device)  # (clean, enhanced), (clean, noisy), (clean, clean)
                    score = self.D(inputs)
                    loss = self.MSELoss(score, target[:, j])
                    self.optimizer_d.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                    self.optimizer_d.step()


    def train_discriminator(self):
        print("Discriminator training phase")
        self.D.train()
        # Get true score of train data
        current_set = self.get_score()
        # Training current list
        disc_loader = create_dataloader(current_set, self.train_noisy_path, clean_path=self.train_clean_path, loader='D', 
                                        batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        self.subtrain_discriminator(disc_loader)

        random.shuffle(self.historical_set)
        
        # Training hist list
        train_hist_length = int(len(self.historical_set) * self.args.hist_portion)
        train_concat_set = self.historical_set[0 : train_hist_length] + current_set
        random.shuffle(train_concat_set)
        disc_loader_hist = create_dataloader(train_concat_set, self.train_noisy_path, clean_path=self.train_clean_path, loader='D', 
                                             batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        self.subtrain_discriminator(disc_loader_hist, hist=True)
        
        # Update the history list
        self.historical_set = self.historical_set + current_set

        # Training current list again
        self.subtrain_discriminator(disc_loader)