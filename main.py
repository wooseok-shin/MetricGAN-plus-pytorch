import random
import shutil
import argparse
import warnings
import numpy as np
import torch
from pathlib import Path
from os.path import join as opj
from train import Trainer
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--exp_name', type=str, default='exp1', help='name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu device')
    parser.add_argument('--output_path', type=str, default='results', help='Model and Log path')
    parser.add_argument('--base_path', type=str, default='data/VCTK_DEMAND', help='Data base path')

    parser.add_argument('--target_metric', type=str, default='pesq', help='pesq or csig or cbak or covl')
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_of_sampling', type=int, default=100)
    parser.add_argument('--num_of_val_sample', type=int, default=824)
    parser.add_argument('--hist_portion', type=float, default=0.2, help='history portion of replay buffer')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--skip_val_epoch', type=int, default=600, help='skip early epoch evaluation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--causal', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_speaker', type=str, nargs='*', default=[], help='select validation speaker (e.g., p226, p227, ..., etc)')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    
    train_noisy_path = f'{args.base_path}/train/noisy/'
    train_clean_path = f'{args.base_path}/train/clean/'
    train_enhan_path = f'{args.output_path}/{args.exp_name}/enhanced_wavs/'

    test_noisy_path = f'{args.base_path}/test/noisy/'
    test_clean_path = f'{args.base_path}/test/clean/'

    model_output_path = Path(args.output_path, args.exp_name, 'model')
    log_output_path = Path(args.output_path, args.exp_name)

    model_output_path.mkdir(parents=True, exist_ok=True)
    log_output_path.mkdir(parents=True, exist_ok=True)

    data_paths = {'train_noisy':train_noisy_path, 'train_clean':train_clean_path, 'train_enhan':train_enhan_path,
                  'test_noisy':test_noisy_path, 'test_clean':test_clean_path,
                  'model_output':model_output_path, 'log_output':log_output_path}


    trainer = Trainer(args, data_paths)
    trainer.train()
    shutil.rmtree(opj(args.output_path, args.exp_name, 'tmp'))
    shutil.rmtree(train_enhan_path)


if __name__ == "__main__":
    main()
