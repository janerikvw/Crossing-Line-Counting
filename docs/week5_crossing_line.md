## Summary
At the moment we got two different solutions to solve the crossing line problem.
- One by taking each time a pair of frames and calculate density map and velocity (Two phase) map around the LOI, then based on that we can calculate how many people cross the line per frame. (parts of people, based on the gaussian in the density map) (Feels more computationally expensive, but better accuracy?)
- The other one takes several frames and takes a slice around the LOI and stitches to 1 image. We normalize the image on speed and angle. Finally we predict based on that individual image how many people crossed the frame in that timeline. (Feels more efficient computing wise)

FOCUS on the two above in method. Especially the first one has a lot more traction, because it works pretty well with the recent Crowd Counting papers and Flow Estimation papers. Also the second method is more old school, but recently there was still a paper which used this approach in combination with a CNN.

The first 3 are the three papers which come back every time on crossing the line. I think it is good to reproduce most of these to set some baselines. According to (2), Ma et al. (3) uses custom filters which are very hard to transfer to other scenes, so probably not a good idea. (1) is also not used with Neural Networks, so I doubt if they are robust enough to handle all the changes in light conditions. (2) is the only one based on Deep Learning,but the big problem is the lack of flow estimated data. (Which annotated is pretty shit)

The datasets which are available are only the ones from http://visal.cs.cityu.edu.hk/downloads/. There is a security-view tracking dataset, with one scene, which is used in all the cross-line papers to compare with each other.



## Interesting papers:
#### 1. Cross-Line Pedestrian Counting Based on Spatially-Consistent Two-Stage Local Crowd Density Estimation and Accumulation, Zheng et al. (2019)
Old school method with own filters on method 1, first introduced in (2). State-of-the-art and quick on the UCSD dataset.

#### 2. Crossing-line Crowd Counting with Two-phase Deep Neural Networks, Zhao et al. (2016)
The second paper who uses CNN's on the LOI counting problem. Instead of (4) using a stiched image and classify it, this paper actually uses flow estimation and density maps together to predict the LOI counts.
So they predict the crowd density and the flow estimation of the pedestrians. Then they multiply these together and predict based on those the counts. (Yet to read about velocity normalization)

The models has a shared architecture with only the last layer apart. One predict the density and the other predicts the flow. (Architecture is based on FlowNet) First those objectives are trained together to obtain good features. Then afterwards both maps are multiplied with each other and this result is lastly fine-tuned to optimize for shiftings between those two.

#### 3. Counting People Crossing a Line Using Integer Programming and Local Features, Ma et al. (2016)
Old school method which uses custom filters and a stiched image (Second approach) to predict the crossing count. Uses their own filters, so hard to use.

#### 4. Large scale crowd analysis based on convolutional neural network. Cao et al. (2015)
This is an old paper which try to calculate the LOI counts by applying aa CNN on method 2. So on the stiched image. But if we look at (2), then we see that the performance is worse than (2).



## Ideas
It feels that the approach in (2) would be very nice to extent. If we can obtain the data and code of them that would be a very nice starting point. If this is not possible, then I have to code it from scratch, which seems possible, but takes more time. Additionally for now we don't have good data yet to train the original model. So I can't start prototyping, which is unfortunate. For now better to focus on finding good unsupervised Flow Estimators and getting them to work on the existing footage.

(2) is also probably the approach we wanna aim for, because we want to distance our selves from custom filters, because it wouldn't fit in the system. So actually only 2 is still in the run. I will look a bit more into (1) as well, so maybe we can apply some ideas of (1) into approach (2) to improve it.

## Focus and planning.
Right now the focus should be on the unsupervised learning. We ofcourse need supervised learning for this as well as testset, but right now we can maybe take a shortcut and try to train on unsupervised methods.
What do I need:
- Some videos from Boen or security camera
- Some labeled Crowd Density data to train the density map.

So my extension would be to add the unsupervised method to improve the scalability to the real world, because we only need a couple of frames labeled.

Planning for next week will be doing this:
- Make a good overview of which unsupervised flow estimation methods there are for flow estimation. Find out if it makes sense to do it like that. (Computationally and training wise)
- Test on untrained things
- Training a Flow Estimator on the unlabeled dataset of the Gemeente


## Things for Thought
- What if traditional approaches work better? Or is it that they work particularly well in non-occluded environments, but we have a highly occluded environment, so less usefull (Lucas-Kanade (LK) check)
