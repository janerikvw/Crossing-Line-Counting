import os
class BasicVideo:
    def __init__(self, base_path):
        self.frames = []
        self.path = base_path


    def get_frames(self):
        return self.frames

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_path(self):
        return self.path


class BasicFrame:
    def __init__(self, image_path, centers=None):
        if centers == None:
            centers = []
        self.centers = centers
        self.image_path = image_path

    def get_centers(self):
        return self.centers

    def add_point(self, xy):
        self.centers.append(xy)

    def get_image_path(self):
        return self.image_path

    def get_density_path(self):
        return os.path.splitext(self.image_path)[0] + ".npy"

