import sunpy.util.net
from sunpy.net.attr import AttrWalker, AttrAnd, AttrOr, DataAttr
from sunpy.net.attrs import Time, Level
from sunpy.net.base_client import QueryResponseTable, BaseClient
from functools import partial

'''
non lo uso
'''

walker = AttrWalker()

# Creator functions
@walker.add_creator(AttrOr)
def create_or(wlk, tree):
    results = []
    for sub in tree.attrs:
        results.append(wlk.create(sub))
    return results

@walker.add_creator(AttrAnd, DataAttr)
def create_and(wlk, tree):
    result = dict()
    wlk.apply(tree, result)
    return [result]

# Applier functions
@walker.add_applier(Time)
def _(wlk, attr, params):
    return params.update({'startTime': attr.start.isot,
                          'endTime': attr.end.isot})

@walker.add_applier(Level)
def _(wlk, attr, params):
    return params.update({'level': attr.value})

# SCIPA Client
class ScipaClient(BaseClient):
    size_column = 'Filesize'

    url_pattern = (
        "https://p3sc.oma.be/datarepfiles/L1/v2/"
        "aspiics_{filter}_l1_{cycle_id:08d}000{seq_num:01d}{acq_num:01d}{exp_num:01d}_"
        "{year:04d}{month:02d}{day:02d}T{hour:02d}{minute:02d}{second:02d}.fits"
    )

    def search(self, *query):
        # Crea le query dal walker
        queries = walker.create(*query)
        results = []

        for query_parameters in queries:
            results.append(self._make_search(query_parameters))

        return QueryResponseTable(results, client=self)


    def _make_search(self, params):
        """
        Placeholder for actual search implementation.
        In a real client, here you would query the SCIPA database or an index.
        For now, return dummy info to match the pattern.
        """
        # Example: simulate a single file per query
        from datetime import datetime
        dt = datetime.fromisoformat(params['startTime'])
        return {
            'ID': f"{dt.strftime('%Y%m%dT%H%M%S')}_001",
            'filter': params.get('level', 'unknown'),
            'cycle_id': 1,
            'seq_num': 1,
            'acq_num': 1,
            'exp_num': 1,
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute,
            'second': dt.second,
        }

    def _make_filename(self, path, row, resp=None, url=None):
        # Build filename from the URL pattern
        return self.url_pattern.format(**row)

    def fetch(self, query_results, *, path, downloader, **kwargs):
        for row in query_results:
            filepath = partial(self._make_filename, path, row)
            url = self.url_pattern.format(**row)
            downloader.enqueue_file(url, filename=filepath)

    @classmethod
    def _can_handle_query(cls, *query):
        query_attrs = set(type(x) for x in query)
        supported_attrs = {Time, Level}
        return supported_attrs.issuperset(query_attrs)
