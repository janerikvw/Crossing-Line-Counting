import os
import numpy as np
from PIL import Image

"""
BasicVideo is an object used to store individual video's with frames.
It is the basic block for training and validating the model.
"""
class BasicVideo:
    def __init__(self, base_path, labeled=False):
        self.frames = []
        self.path = base_path
        self.pairs = None
        self.labeled = labeled

    # Get all the BasicFrame objects of the video
    def get_frames(self):
        return self.frames

    # Add an individual BasicFrame to the video
    def add_frame(self, frame):
        self.frames.append(frame)

    # Get the base path of the video. The directory which stores all of the frames
    def get_path(self):
        return self.path

    # Return all frame pairs of the video
    def get_frame_pairs(self):
        if self.pairs is None:
            self._generate_frame_pairs()

        return self.pairs

    def _generate_frame_pairs(self):
        self.pairs = []

        frames = self.get_frames()

        for i, frame1 in enumerate(frames):
            if i+1 >= len(frames):
                break

            frame2 = frames[i+1]
            self.pairs.append(BasicFramePair(frame1, frame2))

    def is_labeled(self):
        return self.labeled



"""
BasicFrame is the object which stores all the information of an individual frame
Both the frame path and the labeled information are stored here. 
"""
class BasicFrame:
    def __init__(self, image_path, labeled=False):
        self.centers = []
        self.image_path = image_path
        self.labeled = labeled

    # Get all the coordinates of the labeled heads in the frame
    def get_centers(self):
        return self.centers

    # Add an individual head location
    def add_point(self, xy, bbox = None):
        self.labeled = True
        self.centers.append(xy)

    # Retrieve the image path
    def get_image_path(self):
        return self.image_path

    # Retrieve the path where the generated density map is stored
    def get_density_path(self):
        return os.path.splitext(self.image_path)[0] + ".npy"

    # Return a Pillow link to the image file data.
    def get_image(self):
        return Image.open(self.get_image_path())

    # Return the BasicFrame's corresponding RAW image data as a Numpy array.
    def load_image_data(self):
        img = Image.open(self.get_image_path())
        image_data = self.image_data = np.asarray(img)
        img.close()
        return image_data

    # Get the numpy array of the density map
    def get_density(self):
        return np.load(self.get_density_path())

    def is_labeled(self):
        return self.labeled


"""
An object which holds two frames and the tracking information between the two frames (if available)
"""
class BasicFramePair:
    def __init__(self, frame1, frame2, labeled=False):
        self.frame1 = frame1
        self.frame2 = frame2
        self.pairs = {}
        self.labeled = labeled

    # Add the point pair to the dictionary
    def add_point_pair(self, frame1_id, frame2_id):
        self.labeled = True
        self.pairs[frame1_id] = frame2_id

    # Returns all the point pairs in a dictionary, where key is the id in frame 1 with the value the id in frame 2.
    def get_point_pairs(self):
        return self.pairs

    # Get the two frames, when only requiring 1, the index of the frame is given as parameter
    def get_frames(self, frame=None):
        ret = (self.frame1, self.frame2)

        if frame is None:
            return ret
        else:
            return ret[frame]

    def is_labeled(self):
        return self.labeled

