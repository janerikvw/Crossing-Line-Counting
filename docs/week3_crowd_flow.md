## Extra research in:
- Thermal Diffusion Process (5)
- Modified Social Forces (5)
- Spatial Transformer (6)

## Crowd Counting summaries
__ROI:__ Region of Interest (Counting how many people are present in a certain area)
__LOI:__ Line of Interest (How many people cross the line over a certain amount of time)
__Cross Scene:__ More robust in different scenes

#### 1. Crossing the Line: Crowd Counting by Integer Programming with Local Features (2013)
Counting which people are crossing the line (LOI, Line of Interest) by: Taking a fixed width line and taking that line every frame and stitching it together, so we get a static image where the line over time is shown. The image is normalized for speed and orientation. Find HOG descriptors for the people and use those for Crowd Segementation. Then do Count Regression with Integer Programming to get a integer update on the people count crossing the line. (According to Deng et al. the integer programming is rather computationally expensive)

#### 2. Real Time and Scene Invariant Crowd Counting: Across a Line or Inside a Region (2015)
Tries to do LOI and ROI by counting crowds in stead of individual pedestrians. (Blobs are used to measure the amount what part of the crowd has crossed the line) They will introduce a quick and robust way to calculate velocity. Gaussian Process Regression is used to better interpreted the features and crowd count. 
First they seperate the background/foreground. 

#### 3. Vision-based Counting of Pedestrians and Cyclists (2016)
A partnership with City of Pittsburg to promote cycling and walking. (Interesting for introduction?)  Same as before. Foreground/background seperation by removing everything that not has a intense flow estimation. Not very interesting. Biggest contribution was adding a huge dataset (not available) and some extra filtering to detect unwanted small padestrains.

#### 4. Counting People Crossing a Line Using Integer Programming and Local Features (2016)
Switching from blobs to individual persons. Because this can help with accurately highly occludes blobs predict more accurately. Focus is on LOI, and ROI on both sides is a bad idea according to the paper.
They use temporal slicing to get a single image of a line over time. (So every frame take a slice around the line) Using Flow Mosaic they have variable width of the line, so according to perspective the line is thicker of thinner to normalize the velocity of the blobs.
Same method can be applied on individual people, but the HOG descriptors of individual people has more issues with occlusion.

Same authors as Crossing the Line (2013), but has some minor improvements. (Other loss function, which improves speed a lot with a slightly less high accuracy in high density videos)
Multiple ROI's possible. More experimentation.

#### 5. Crowd Behaviour Identification (2016)
This paper is more about how crowd behaves and finding possible problem spots. This is less focussed on actually tracking people, but more checking how the flow of people is moving in the space.
For optical flow the Farnebäck optical flow technique is used.

They use Thermal diffusion process as well. (Not sure what it does with energy, more research on this, maybe interesting for merging flows)
Modified social forces is also done, to exclude non-flow behaving people. These two combined are used from simple ROI to detecting possible spots and possible lanes. (Which is what we want)

#### 6. Locality-Constrained spatial transformer network for video crowd counting (2019)
Crowd counting for video. So the temporal information is used to keep track of people and improve the detection. Earlier LSTM's were used to transfer information from the earlier frames to the current frame prediction. In this paper a Tranformer is used to predict the next frame with used of the neighbouring frames. (LSTM save the temporal information in an implicit way, but for such local movements which have sometimes a lot of occlusion, explicit tranfering is better and can be done with a tranformer.)
Explains that there are not a lot of papers doing video crowd counting, but only 2 with LSTM's. (Xiong et al. 2017, Zhang et al. 2017)

Still mainly an crowd counting problem, but tries to improve by leveraging the temporal information. Looking at the results. Single image crowd counting is only marginally worse. (Question is how complex is the other model?)

Additionally this is the paper which published the big Shanghai part B dataset. :)

__INTERESTING!__


#### 7. An Efficient Crossing-Line Crowd Counting Algorithm with Two-Stage Detection (2019)
ROI counting can be devided into: object detection and regression. Where object detection is finding each exact pedestrain, but will loss accuracy when occlusion appears. (Boxing the objects) Regression is less about the exact position of the person. (For example density maps) But more about that there is a person and that we can count it.

This paper introduces crowd counting, but even in 2019 with more traditional methods to improve the speed without degrading the accuracy too much.
They use SVM's and affinity propagation clustering instead of CNN's which produce the density map and flow estimation.

I doubt if the model is very robust when a lot of people are in the frame. Then the tracking feels very hard to do and errors can occur.

__GOOD METHOD!__

#### 8. Cross-Line Pedestrian Counting Based on Spatially-Consistent Two-Stage Local Crowd Density Estimation and Accumulation
Good method with lots of details about different ROI/LOI methods and their advantages/disadvantages.

The method breaks the problem in two stages (Same as 7). It makes local regions around the Line of Interest, which are skewed, because of the angle of the camera. (Further away of the camera the smaller the region)
Then the first stage is to calculate the regions local density (Taking integral over the region). Second stage is to calculate the local velocity of the region. Combining these two give us an idea how many people are crossing the line.
METHOD TO LOOK FURTHER INTO!!!

MORE:
Use line slicing as well with a variable line width.
Using detected Shi-Tomasi corners which are tracked by Lucas-Kanade (LK) optical flow they can track small crowds in a scene. (Only usable in crowded scenes)
Each forground pixel is then assigned to one of the two directions based on kNN of the features (corners)

Looking at the results paper (4) is still very good in comparing to (8). (8) is only marginally beter, eventhough the other paper is 4 years old.


### Way to go?
Track people for a short amount of time (Only estimating the velocity and direction) and based on this we can come up with a very nice flow (Thermal Diffusion) But problem is when flows intertwine. Would be very cool to do Crowd Behaviour (5) on the current dataset. (If we have those)