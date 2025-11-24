import json


def readparams(configfile, header_keys):
    """This function puts for each value in header_keys the header values in the
    dictionary params as key value pair. Then it adds to the dictionary params
    a key DC and value the data of the first hdul in the fits file,
    a key FLAT and value the data of the first hdul in the fits file,
    a key BIAS and value the data of the first hdul in the fits file,
    it overwrites each of the previous key value pairs with a values in the file OVERRULE_FILE,
    it adds all keys in the file OVERRULE_FILE to the dictionary params.
    """
    params = {}
    overrule_params(configfile, params)
    return params


def overrule_params(configfile, params):
    # fn = os.path.join(inputdir, OVERRULE_FILE)
    with open(configfile) as json_file:
        datarep_params = json.load(json_file)
    params.update(datarep_params)
