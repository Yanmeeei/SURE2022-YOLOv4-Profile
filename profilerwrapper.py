import timer
import memorizer
import datascale


class ProfilerWrapper(object):
    def __init__(self):
        self.mr = memorizer.MemRec()
        self.tt = timer.Clock()
        self.scale = datascale.DataScale()

    def report(self, sample=False):
        self.mr.report(sample=sample)
        self.tt.report(sample=sample)
        self.scale.report()
