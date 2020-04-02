# Important papers for Flow Estimation

The following papers will be important for Flow Estimation. The first 2 are methods to upgrade supervised Optical Flow Estimators to unsupervised ones. So we don't need as much data to trained the Flow Estimator.

(3) and (4) are the most popular Flow Estimator architectures. FlowNet is purely CNN based, but PWC-Net is a combination with traditional models, which makes it much easier to train, but harder to merge to for example a unified model for both predicting Flow Estimation and Crowd Counting

#### 1. DDFlow: Learning Optical Flow with Unlabeled Data Distillation (2019)
A method to learn flow estimators in an unsupervised way. Purely data-driven.

They find unoccluded pixels in a full image and take patches, so the unoccluded pixels of the next frame are outside the image. So we can obtain the ground-truth from those pixels, but the patch doesn't have that information, so it has to learn those.

Second in KITTI and Sintel behind SelFlow.

Uses PWC-Net (3) and wraps around a method to apply a unsupervised method on the supervised flow estimator, but can be used with every other Flow Estimator architecture.

#### 2. SelFlow: Self-Supervised Learning of Optical Flow (2019)
Need to read more how this on obtains the ground truths for the network, but was first in Sintel en KITTI benchmarks.

They market the method more as both unsupervised learning and afterwards supervised finetuning. (Would be perfect for Eagle Eye) But performs unsupervised only already better than DDFlow.

Both SelFlow and DDFlow have the same main author, but the 2nd and 3rd authors are swapped.

In DDFlow they crop the images in such a way that the non-occluded pixel appears outside the patch. With SelFlow they patch the non-occluded pixel with noise, so they are occluded as well.
As well they use multiple frames to better estimate the flow. Has performance improvements.

#### 3. PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume (2018)
A more efficient way to estimate the Flow. Uses way less parameters than FlowNet and is around 2x quicker. Combines some traditional methods to obtain this improvement.

#### 4. FlowNet: Learning Optical Flow with Convolutional Networks (2015)
First Optical Flow Estimator using CCN's. A year later FlowNet2 was created, but basically the same base architecture, but used a bigger network and some small improvements to obtain better results.

Zhao et al. (2016) used FlowNet as well and based on the architecture both the integrated Flow Estimator and the Crowd Density estimator.