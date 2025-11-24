from sunpy.net.attr import AttrAnd, AttrComparison, AttrOr, AttrWalker, DataAttr, SimpleAttr
from sunpy.net import Fido, attrs as a

class Filter(DataAttr):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def collides(self, other):
        return isinstance(other, Filter)

class CycleID(DataAttr):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def collides(self, other):
        return isinstance(other, CycleID)

class Seq_num(DataAttr):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def collides(self, other):
        return isinstance(other, Seq_num)

class Acq_num(DataAttr):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def collides(self, other):
        return isinstance(other, Acq_num)

class Exp_num(DataAttr):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def collides(self, other):
        return isinstance(other, Exp_num)

walker = AttrWalker()

@walker.add_applier(Filter)
def _apply_filter(wlk, attr, imap):
    imap['filter'] = attr.value

@walker.add_applier(CycleID)
def _apply_cycle(wlk, attr, imap):
    imap['cycle_id'] = attr.value

@walker.add_applier(Seq_num)
def _apply_seq(wlk, attr, imap):
    imap['seq_num'] = attr.value

@walker.add_applier(Acq_num)
def _apply_seq(wlk, attr, imap):
    imap['acq_num'] = attr.value

@walker.add_applier(Exp_num)
def _apply_seq(wlk, attr, imap):
    imap['exp_num'] = attr.value
