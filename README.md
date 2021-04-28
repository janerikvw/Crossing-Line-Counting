# Crossing-Line Counting with Deep Spatial-Temporal Feature Learning
 In this repository [the results (final-version-thesis.pdf)](final-version-thesis.pdf) and code are published regarding the master thesis of Jan Erik van Woerden for the Master Artificial Intelligence at the University of Amsterdam.


# How to run the code
All the training and testing code (Main loop) is in code/main.py. In the rest of the files there are utilities and other core modules with corresponding names. In code/datasets the files for loading the datasets are present and in code/data the different datasets need to be stored. The code/gd_runner is bash script which makes it possible to run multiple trainingruns from parameters stored in a Google Drive Spreadsheet.

Below there are several demo's and explainations for running code:

## Training a simple model:
Here is a simple example for training. The only required parameter is the name for results saving. The rest are all optional. (Check for defaults in the main.py. In this example this are the default values for the extra parameters)
```
python main.py unique_name_of_results_saving --mode train --model p21small --dataset fudan --density_model fixed-8 --frames_between 5
```

- `--model` selects the models. `p21small`=without context, `p33small`=without warping, `p43small`=flow as context, `p632small`=proposed/with warping, `p72small`=with warping/frame2 + flow as context.
- `--mode` is pure for training or testing. When mode is set to `loi` the model will be tested, otherwise training.
- `--dataset` selects the datasets, currently `fudan`, `uscd`, `aicity` and `tub` are supported for both testing and training.
- `--frames_between` The amount of frames between predictions. So if you have a input of 25 fps, then skipping 5 frames, brings the fps down to 5fps evaluation. Larger displacements, so larger accuracy, but more occlussion/error. Results in thesis show that 5FPS is optimal, but 1FPS is still more accurate, but much more costly. Would be ideal to variate this for training.
- `--density_model` Is the size of the density Guassian, it is `fixed-{size}`, with an arbitrary integer for `{size}`.

## Testing a simple model:
A simple example of evaluating for LOI performance. The results are saved at the end in `code/loi_results`.

Use only the LOI results, because the ROI results can be corrupted due to the use of multiple lines for a single example. Additionally for the LOI, we skip the evaluation for the inbetween frames (Of `--frames_between`), so not all the ROI performances are measured.
```
python main.py unique_name_of_results_saving --mode loi --model p21small --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1
```
- `--loi_level` Explains which type op LOI evaluation is performed, `pixel`, `region` or `cross`. Pixel works best in almost all cases.
- `--loi_maxing` Is maxing applied or not: `0` or `1`. The parameters for the maxing are edited inside the main.py per dataset.

### ROI performance
To accurately use the ROI performance, use `--eval_model roi`. This runs over all the samples without skipping and only 1 time.

```
python main.py unique_name_of_results_saving --mode loi --model p21small --dataset fudan --frames_between 5 --loi_level pixel --loi_maxing 1 --eval_mode roi
```
With the `--eval_mode` we can evaluate the ROI performance.

### Moving counting
To counting moving counting obstacles we can replace the `--loi_level` with `moving_counting`.
These tests work currently only on `aicity` and `tub`. Implementations of `ucsd` should be possible. (Due to the presence of pedestrian trajectories)
```
python main.py unique_name_of_results_saving --mode loi --model p21small --dataset tub --frames_between 5 --loi_level moving_counting --loi_maxing 1 --eval_mode roi
```

### Take several images from trained model:
To take images and store in `code/full_imgs` for presentations replace `--loi_level take_image`. Only 6 per sample videos are stored. When performing on multiple models, all results are grouped per video, so easily comparison between the models on the same sample.
```
python main.py trained_data_model --mode loi --model p21small --dataset fudan --frames_between 1 --loi_level take_image --loi_maxing 0
```


## Notes:
Normally the both flow and density are trained together. `--loss_focus full`. However CSRNet can't do this, so you first have to train CSRNet with `--loss_focus cc` and a seperate model as p21small with `--loss_focus fe`. In the loi_test function you can set the seperate flow model to test CSRNet.

Loss function needs to be edited in the code itself. New settings can be selected with `--lr_setting`. The default is `--lr_setting adam_9`.

With `--loi_width` you can select the width of the LOI. This is the width around the LOI used for performing the merge between Flow and Density. It can influence the performance, but `40` is a pretty safe bet as default. For smaller pedestrians, smaller size should be fine as well.