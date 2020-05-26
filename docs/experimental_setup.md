# Experimental setup

## Used datasets:

### Fudan-ShanghaiTech
- https://github.com/sweetyy83/Lstn_fdst_dataset
- For semi busy environments. Possibly be handled by Yolo as well. We'll see.
- No trajectories, so we need to label how many people crossed the line
- 100 videos of each 150 frames (6 seconds of 25fps) from 13 different scenes

### TUB Crowd Flow
- https://github.com/tsenst/CrowdFlow
- Very busy, so good for really showing the power of crowd counting model
- 5 scenes (25 fps, 12 seconds each)
- Has person trajectories, so unlimited line draw

### USCD Pedestrian
- http://www.svcl.ucsd.edu/projects/peoplecnt/ (Multiple links available)
- Very low quality, but standard for earlier datasets, so comparison with legacy papers
- Has person trajectories, so unlimited line draw



## Other current CC models:
https://github.com/gjy3035/Awesome-Crowd-Counting/blob/master/src/Light-weight_Model.md