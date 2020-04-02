# Meeting with Zenglin week 3

### Point to discuss
- Discuss which exact dataset they can provide. Feel more that we should make clear what kind of data good is for the model. I propose a flow tracking for a limited amount of frames. And maybe predicting where problems are in the crowds?
- Tommorrow meeting and presentation
- Starting to do crowd flow estimation implementation (Density map prediction and then with some flow map predict direction. Test on the dataset with tracking)
- Show baseline (YOLO). This should be easily improvable especially for small pedestrians.
- Found one dataset with tracking pedestrians, so we have a testset with semi busy area
- Show realistic images send by Maarten
- Next weeks: Trying to implement the full Crowd Counting MCNN (And maybe the other one as well) and starting with several flow estimators.
- Next weeks: More focus on Crowd Behavior (Reading more about this)

- Felt like a lot of papers were focussing on the feature extraction, because those are quicker and so possible to perform real time classification. Especially from the last years.


AFTER TODO:
- Get the Flow estimation to work (FOCUS) Try FlowNet otherwise KLT tracker.
- Get the Crowd Count to work
- KLT


In 2 months demo, and from then on we'll see


After next meeting:
- 

- Read beginning of method 3.1: Cross-scene Crowd Counting via Deep CNN's (2015), Cong Zhang et al. (For whole body moving direction)
- Read: Collective ensity Clustering for Coherent Motion Detection


