import numpy as np

def remove_duplicates(series):
    """ Remove identical consecutive observations
    """
    cleaned_series=series[np.insert(np.diff(series).astype(bool), 0, True)]
    
    return cleaned_series


