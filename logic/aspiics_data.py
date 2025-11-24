import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def aspiics_files(filter: str = None,
               cycle_id: str = None,
               seq_num: int = None,
               acq_num: int = None,
               exp_num: int = None,
               start_date: str = None,  
               end_date: str = None):
    """
    Search ASPIICS repository to find filenames and return the urls of the images. 
    Supports multiple filters separated by commas.
    If start_date and end_date are given, it filters the time.

    Args:
        filter: ASPIICS filter (also a list, ex: 'p1,p2')
        cycle_id: ID of the observation cycle
        seq_num: sequence number (at same filter and exp time)
        acq_num: number relative to the filter 
        exp_num: number relative to the exposure time
        start_date: date filter start. format: "YYYYMMDD'T'HHMMSS"
        end_date: date filter end. format: "YYYYMMDD'T'HHMMSS"

    Returns:
        list of urls of the images 
    """

    base_url = "https://p3sc.oma.be/datarepfiles/L1/v2/"

    r = requests.get(base_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    all_files = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".fits")]

    pattern = r"^aspiics_"

    if filter:
        filters = [f.strip() for f in filter.split(',')]
        pattern += '(?:' + '|'.join(map(re.escape, filters)) + ')'  
    else:
        pattern += r".+"

    pattern += r"_l1_"

    if cycle_id is not None:
        cycle_ids = [c.strip() for c in str(cycle_id).split(',')]
        pattern += '(?:' + '|'.join(map(re.escape, cycle_ids)) + ')'
    else:
        pattern += r"\d{8}"

    if seq_num is not None:
        pattern += f"000{seq_num:01d}"
    else:
        pattern += r"000\d"

    if acq_num is not None:
        pattern += f"{acq_num:01d}"
    else:
        pattern += r"\d"

    if exp_num is not None:
        pattern += f"{exp_num:01d}"
    else:
        pattern += r"\d"

    pattern += r"_(\d{8}T\d{6})\.fits$"

    regex = re.compile(pattern)
        
    fmt = "%Y%m%dT%H%M%S"
    start_dt = datetime.strptime(start_date, fmt) if start_date else None
    end_dt = datetime.strptime(end_date, fmt) if end_date else None

    matched_files = []
    for f in all_files:
        m = regex.match(f)
        if m:
            date_str = m.group(1)
            dt = datetime.strptime(date_str, fmt)

            if start_dt and dt < start_dt:
                continue
            if end_dt and dt > end_dt:
                continue

            matched_files.append(f'{base_url}{f}')

    return matched_files
