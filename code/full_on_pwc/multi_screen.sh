#!/bin/bash

# Fudan

python main.py 20201201_125827_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-1_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 1 --loi_level take_image --loi_maxing 0
python main.py 20201201_154822_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 2 --loi_level take_image --loi_maxing 0
python main.py 20201207_213704_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201202_001230_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-10_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 10 --loi_level take_image --loi_maxing 0
python main.py 20201202_030514_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-25_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 25 --loi_level take_image --loi_maxing 0


python main.py 20201121_093417_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201121_093417_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201121_092743_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201121_092743_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201207_213704_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201207_213704_dataset-fudan_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201124_165148_dataset-fudan_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p33small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201124_165148_dataset-fudan_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p33small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201125_152255_dataset-fudan_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p43small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201125_152255_dataset-fudan_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p43small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201130_233710_dataset-fudan_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p632small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201130_233710_dataset-fudan_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p632small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201130_130356_dataset-fudan_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p72small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201130_130356_dataset-fudan_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p72small --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

# # CrowdFlow
python main.py 20201126_105213_dataset-tub_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model baseline21 --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201126_105213_dataset-tub_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model baseline21 --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201126_124108_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-750_lr_setting-adam_2 --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201126_124108_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-750_lr_setting-adam_2 --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201126_112447_dataset-tub_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201126_112447_dataset-tub_model-p21small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201202_224804_dataset-tub_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p33small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201202_224804_dataset-tub_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p33small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201202_141502_dataset-tub_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p43small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201202_141502_dataset-tub_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p43small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201203_003126_dataset-tub_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p632small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201203_003126_dataset-tub_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p632small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201203_030222_dataset-tub_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p72small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201203_030222_dataset-tub_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-700_lr_setting-adam_9 --mode loi --model p72small --dataset tub --frames_between 5 --loi_level take_image --loi_maxing 1

# # AICity
python main.py 20201126_105213_dataset-tub_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model baseline21 --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201126_105213_dataset-tub_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model baseline21 --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201129_095114_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model csrnet --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201129_095114_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-2000_lr_setting-adam_2 --mode loi --model csrnet --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201201_044622_dataset-aicity_model-p21small_density_model-fixed-5_cc_weight-50_frames_between-2_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201201_044622_dataset-aicity_model-p21small_density_model-fixed-5_cc_weight-50_frames_between-2_epochs-350_lr_setting-adam_9 --mode loi --model p21small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201128_134718_dataset-aicity_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p33small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201128_134718_dataset-aicity_model-p33small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p33small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201128_013534_dataset-aicity_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p43small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201128_013534_dataset-aicity_model-p43small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p43small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201207_081056_dataset-aicity_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p632small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201207_081056_dataset-aicity_model-p632small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p632small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1

python main.py 20201207_081110_dataset-aicity_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p72small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 0
python main.py 20201207_081110_dataset-aicity_model-p72small_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p72small --dataset aicity --frames_between 5 --loi_level take_image --loi_maxing 1