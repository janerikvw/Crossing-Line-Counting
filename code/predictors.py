class RegionLOI():
    def __init__(self):
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
        this.cc = cc_model
        this.fe = fe_model

    def run(self, frame1, frame2):
        return this.cc.run(frame2), this.fe.run(frame1, frame2)


class FullPredictor():
    def __init__(self):
        pass

    def run(self, frame1, frame2):
        pass
