from sunpy.net.dataretriever.client import GenericClient
from sunpy.net import attrs as a
from sunpy.net import Fido, attrs as a
from resources import aspiics_attrs

'''
Non lo uso
'''

class AspiicsClient(GenericClient):
    pattern = (
        "https://p3sc.oma.be/datarepfiles/L1/v2/"
        "aspiics_{filter}_l1_{cycle_id:08d}000{seq_num:01d}{acq_num:01d}{exp_num:01d}_"
        "{year:04d}{month:02d}{day:02d}T{hour:02d}{minute:02d}{second:02d}.fits"
    )

    @classmethod
    def register_values(cls):
        return {
            a.Instrument: [("aspiics", "Association of Spacecraft for Polarimetric and Imaging Investigation of the Corona of the Sun")],
        }
    
    @classmethod
    def _attrs_module(cls):
        return 'aspiics', 'resources.aspiics_attrs'
    
    @classmethod
    def _can_handle_query(cls, *query):
        query_attrs = set(type(x) for x in query)
        supported_attrs = {a.Instrument, aspiics_attrs.Filter, aspiics_attrs.CycleID,
                           aspiics_attrs.Seq_num, aspiics_attrs.Acq_num, aspiics_attrs.Exp_num, a.Time}
        return supported_attrs.issuperset(query_attrs)

def _make_url_for_query(cls, query_attrs):
    return cls.pattern.format(**query_attrs)

def search_aspiics(start_time, end_time, filter, cycle_id, seq, acq, exp):
    return Fido.search(
        a.Time(start_time, end_time),
        a.Instrument.aspiics,
        a.Attr("filter", filter),
        a.Attr("cycle_id", cycle_id),
        a.Attr("seq_num", seq),
        a.Attr("acq_num", acq),
        a.Attr("exp_num", exp)
    )
