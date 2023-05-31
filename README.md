## MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement (PyTorch Implementation)

This is a PyTorch implementation of "[MetricGAN+](https://arxiv.org/abs/2104.03538)" (Szu-Wei Fu, Cheng Yu, Tsun-An Hsieh, Peter Plantinga, Mirco Ravanelli, Xugang Lu, Yu Tsao, 2021 Interspeech).
This repository is implemented only with PyTorch based on the implementation of [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN).



## Setup Datasets
**VoiceBank-DEMAND:** Please download clean_trainset_28spk_wav.zip, noisy_trainset_28spk_wav.zip, clean_testset_wav.zip, and noisy_testset_wav.zip from [here](https://datashare.ed.ac.uk/handle/10283/2791)
and extract them to `data/VCTK_DEMAND/train(or test)/clean(or noisy)`.


The final folder structure should look like this:
```none
MetricGAN+
├── ...
├── data
│   ├── VCTK_DEMAND
│   │   ├── train
│   │   │   ├── clean
│   │   │   ├── noisy
│   │   ├── test
│   │   │   ├── clean
│   │   │   ├── noisy
├── ...
```

## Training
```shell
python main.py ---exp_name=exp1 --target_metric pesq
```
You can change the hyperparameters (target_metric, epochs, batch_size, hist_portion, lr, ...).
```shell
python main.py ---exp_name=exp2_csig_hist0.1 --target_metric csig --hist_portion=0.1
```

## Testing & Inference
```shell
python inference.py --weight_path results/exp1/model/ --weight_file best_model.pth
```


## Results and Checkpoints
We provide results and checkpoints of MetricGAN+ on the VoiceBank-DEMAND dataset.

| Target Metric       | PESQ | CSIG | CBAK | COVL |
|---------------------|------|------|------|------|
| PESQ - trial 1      | 3.13 | 4.13 | 3.03 | 3.61 |
| PESQ - trial 2      | 3.08 | 4.07 | 3.15 | 3.56 |
| PESQ - trial 3      | 3.15 | 4.09 | 3.16 | 3.61 |
| CSIG - trial 1      | 3.12 | 4.26 | 3.07 | 3.68 |

Please download the weights file from [our release](https://github.com/wooseok-shin/MetricGAN-plus-pytorch/releases/tag/v1.weights), 
put them in your path, and run inference.
```shell
python inference.py --weight_path your/path/ --weight_file PESQ-GAN_trial1.pth
```