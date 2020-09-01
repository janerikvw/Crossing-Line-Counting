class RegionLOI():
    def __init__(self, point1, point2, img_width, img_height, ped_size,  width_peds, height_peds, crop_processing=False):
        pass

    def run(self):
        pass


class PixelLOI():
    def __init__(self):
        pass

    def run(self):
        pass


class CSRPredictor():
    def __init__(self):
        pass

    def run(self, frame2):
        pass


class PWCPredictor():
    def __init__(self):
        pass

    def run(self, frame1, frame2):
        pass


class CombinedPredictor():
    def __init__(self, cc_model, fe_model):
        self.cc = cc_model
        self.fe = fe_model

    def run(self, frame1, frame2):
        return self.cc.run(frame2), self.fe.run(frame1, frame2)


class FullPredictor():
    def __init__(self):
        pass

    def run(self, frame1, frame2):
        pass
