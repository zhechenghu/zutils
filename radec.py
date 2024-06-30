import numpy as np

def ra2alpha(ra, sep=":"):
    ra_sep = np.array(ra.split(sep)).astype(float)
    alpha = (ra_sep[0] + ra_sep[1] / 60.0 + ra_sep[2] / 3600.0) * 15.0
    return alpha

def dec2delta(dec, sep=":"):
    dec_sep = np.array(dec.split(sep)).astype(float)
    if dec_sep[0] < 0:
        delta = dec_sep[0] - dec_sep[1] / 60.0 - dec_sep[2] / 3600.0
    else:
        delta = dec_sep[0] + dec_sep[1] / 60.0 + dec_sep[2] / 3600.0
    return delta