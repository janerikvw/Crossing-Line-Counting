#!/bin/bash

python main.py 20201116_145333_dataset-fudan_model-p21base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201116_145345_dataset-fudan_model-p3base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p3base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201117_082909_dataset-fudan_model-p31base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p31base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201117_154221_dataset-fudan_model-pcustom_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model pcustom --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201117_154540_dataset-fudan_model-p4base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p4base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201118_034001_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-750_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201118_143201_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-750_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
python main.py 20201118_093140_dataset-fudan_model-p4base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_lr_setting-adam_9 --mode loi --model p4base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1


#python main.py 20201116_145333_dataset-fudan_model-p21base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p21base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
# python main.py 20201116_145345_dataset-fudan_model-p3base_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-350_lr_setting-adam_9 --mode loi --model p3base --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
# python main.py 20201110_022216_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_loss_focus-cc_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1

#python main.py 20201110_183517_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-400_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
#python main.py 20201110_022216_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_loss_focus-cc_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
#python main.py 20201110_125216_dataset-fudan_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-150_lr_setting-adam_8 --mode loi --model v5dilation --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
#python main.py 20201107_104351_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2 --mode loi --model v51flowfeatures --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 0
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level take_image --loi_maxing 1

#python main.py 20201111_204009_dataset-tub_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model baseline21 --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201111_150840_dataset-aicity_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-2_epochs-300_lr_setting-adam_2_pre --mode loi --model baseline21 --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0

#python main.py 20201110_183517_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-400_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1

#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201110_183517_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-400_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixedadam_2_resize_mode-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0

#python main.py 20201107_104351_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2 --mode loi --model v51flowfeatures --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201110_125216_dataset-fudan_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-150_lr_setting-adam_8 --mode loi --model v5dilation --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0

#python main.py 20201110_183517_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-400_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
##python main.py 20201110_183517_dataset-fudan_model-baseline21_density_model-fixed-8_cc_weight-1_frames_between-5_epochs-400_lr_setting-adam_2 --mode loi --model baseline21 --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201110_022216_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_loss_focus-cc_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
##python main.py 20201110_022216_dataset-fudan_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-500_loss_focus-cc_lr_setting-adam_2 --mode loi --model csrnet --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201110_125216_dataset-fudan_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-150_lr_setting-adam_8 --mode loi --model v5dilation --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
##python main.py 20201027_073844_dataset-fudan_model-v5dilation_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear --mode loi --model v5dilation --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201107_104351_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2 --mode loi --model v51flowfeatures --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
##python main.py 20201107_104351_dataset-fudan_model-v51flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2 --mode loi --model v51flowfeatures --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201111_055245_dataset-fudan_model-v55flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-300_lr_setting-adam_8 --mode loi --model v55flowwarping --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1


#python main.py 20201107_171156_dataset-tub_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset tub --frames_between 5 --loi_level moving_counting --loi_maxing 1
#
#python main.py 20201104_163749_dataset-aicity_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 0
#python main.py 20201107_034757_dataset-aicity_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 0
#python main.py 20201104_163749_dataset-aicity_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1
# python main.py 20201107_034757_dataset-aicity_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1

# python main.py 20201107_171204_dataset-aicity_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1


#python main.py 20201106_233644_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level moving_counting --loi_maxing 0
#python main.py 20201105_151431_dataset-tub_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset tub --frames_between 5 --loi_level moving_counting --loi_maxing 1

#python main.py 20201107_171204_dataset-aicity_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 0
#python main.py 20201106_180415_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 0
#python main.py 20201106_180415_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1
#python main.py 20201104_163749_dataset-aicity_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1
#python main.py 20201107_034757_dataset-aicity_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset aicity --frames_between 2 --loi_level moving_counting --loi_maxing 1

#python main.py 20201107_171156_dataset-tub_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201107_171156_dataset-tub_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201106_233644_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201106_233644_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201106_233644_dataset-tub_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset tub --frames_between 5 --loi_level crossed --loi_maxing 1
#python main.py 20201105_151431_dataset-tub_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201105_151431_dataset-tub_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset tub --frames_between 5 --loi_level crossed --loi_maxing 1
##python main.py 20201024_201737_dataset-fudan_model-v5flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear --mode loi --model v5flowfeatures --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201105_113739_dataset-tub_model-v51flowfeatures_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v51flowfeatures --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201105_132238_dataset-tub_model-v51flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v51flowwarping --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201107_015407_dataset-tub_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 1
#python main.py 20201107_015407_dataset-tub_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201105_113739_dataset-tub_model-v51flowfeatures_density_model-fixed-8_cc_weight-50_frames_between-5_epochs-250_lr_setting-adam_2_pre --mode loi --model v51flowfeatures --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0

#python main.py 20201107_171204_dataset-aicity_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0
#python main.py 20201107_171204_dataset-aicity_model-baseline2_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model baseline2 --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 1
#python main.py 20201106_180415_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0
#python main.py 20201106_180415_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 1
#python main.py 20201106_180415_dataset-aicity_model-csrnet_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_loss_focus-cc_lr_setting-adam_2_pre --mode loi --model csrnet --dataset aicity --frames_between 2 --loi_level crossed --loi_maxing 1
#python main.py 20201104_163749_dataset-aicity_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0
#python main.py 20201104_163749_dataset-aicity_model-v5dilation_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5dilation --dataset aicity --frames_between 2 --loi_level crossed --loi_maxing 1
#python main.py 20201024_201737_dataset-fudan_model-v5flowfeatures_cc_weight-50_epochs-500_lr_setting-adam_2_resize_mode-bilinear --mode loi --model v5flowfeatures --dataset tub --frames_between 5 --loi_level pixel --loi_maxing 0
#python main.py 20201105_165901_dataset-aicity_model-v51flowfeatures_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v51flowfeatures --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0
#python main.py 20201106_043800_dataset-aicity_model-v51flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v51flowwarping --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 1
#python main.py 20201107_034757_dataset-aicity_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 1
#
#python main.py 20201106_043800_dataset-aicity_model-v51flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v51flowwarping --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0
#python main.py 20201107_034757_dataset-aicity_model-v5flowwarping_density_model-fixed-8_cc_weight-50_frames_between-2_epochs-400_lr_setting-adam_2_pre --mode loi --model v5flowwarping --dataset aicity --frames_between 2 --loi_level pixel --loi_maxing 0