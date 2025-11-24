import numpy as np
import sunpy.map
import warnings
import sunkit_image.enhance as enhance  
import astropy.units as u
import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins
from skimage.filters import unsharp_mask

def wow_filter(smap):
    """Apply WOW filter to SunPy Map."""
    if smap is None:
        raise ValueError("No solar map provided")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean_data = np.nan_to_num(smap.data, nan=np.nanmedian(smap.data))
        clean_map = sunpy.map.Map(clean_data, smap.meta)
        wow_map = enhance.wow(
            clean_map,
            bilateral=1,
            denoise_coefficients=[5, 2, 1],
            gamma=4,
            h=0.99
        )
    return wow_map

def mgn_filter(smap):
    """Apply MGN filter to SunPy Map."""
    if smap is None:
        raise ValueError("No solar map provided")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean_data = np.nan_to_num(smap.data, nan=np.nanmedian(smap.data))
        clean_map = sunpy.map.Map(clean_data, smap.meta)
        wow_map = enhance.mgn(
            clean_map,
        )
    return wow_map


def log_scale(smap):
    """Apply logscale to SunPy Map."""
    if smap is None:
        raise ValueError("No solar map provided")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smap.data[smap.data <= 0] = np.nan
        log_image = np.log10(smap.data)
        log_map = sunpy.map.Map(log_image, smap.meta)
    return log_map

def nrgf_filter(smap):
    radial_bin_edges = equally_spaced_bins(1.3, 3.5, smap.data.shape[0] // 4)
    radial_bin_edges *= u.R_sun
    base_nrgf = radial.nrgf(smap, radial_bin_edges=radial_bin_edges, application_radius=1.3 * u.R_sun)
    return base_nrgf

def unsharp_mask_filter(smap, radius=5, amount=2):
    unsh_image = unsharp_mask(smap.data, radius, amount)
    unsh_map = sunpy.map.Map(unsh_image, smap.meta)
    return unsh_map

def remove_dark_bias(smap, dark_map):
    """
    Applica la correzione dark+bias.

    Args:
        smap: SunPy Map da correggere
        dark_map: SunPy Map del dark frame

    Returns:
        SunPy Map corretta
    """
    if smap is None or dark_map is None:
        raise ValueError("Input map(s) missing")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected_data = smap.data - dark_map.data
        corrected_map = sunpy.map.Map(corrected_data, smap.meta)
    return corrected_map

def flat_field_correction(smap, flat_map):
    """
    Applica la correzione flat field.

    Args:
        smap: SunPy Map da correggere
        flat_map: SunPy Map del flat frame

    Returns:
        SunPy Map corretta
    """
    if smap is None or flat_map is None:
        raise ValueError("Input map(s) missing")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat_norm = flat_map.data / np.mean(flat_map.data)
        corrected_data = smap.data / flat_norm
        corrected_map = sunpy.map.Map(corrected_data, smap.meta)
    return corrected_map

def calculate_darkbias(bias, dark_A, dark_B, dark_C, header):
    '''
    Calcola dark+bias con modello quadratico: bias + (dark_A + dark_B*temp + dark_C*temp^2)/t_exp

    Args:
        bias: Array bias
        dark_A: Array dark_A 
        dark_B: Array dark_B 
        dark_C: Array dark_C
        header: header immagine considerata, per estrarre temp e t_exp

    Returns:
        Array di dark+bias da sottrarre all'immagine 

    '''
    temp = float(header['APS_TEMP'])
    t_exp = float(header['EXPTIME'])
    return (bias + (dark_A + dark_B*temp + dark_C*temp**2)*t_exp)

def calculate_dark(dark_A, dark_B, dark_C, header):
    '''
    Calcola dark con modello quadratico: (dark_A + dark_B*temp + dark_C*temp^2)/t_exp

    Args:
        dark_A: Array dark_A 
        dark_B: Array dark_B 
        dark_C: Array dark_C
        header: header immagine considerata, per estrarre temp e t_exp

    Returns:
        Array di dark da sottrarre all'immagine 

    '''
    temp = float(header['APS_TEMP'])
    t_exp = float(header['EXPTIME'])
    return (dark_A + dark_B*temp + dark_C*temp**2)*t_exp

def calculate_bias(bias_A, bias_B, header):
    '''
    Calcola bias tenendo conto di temperatura: (bias_A + bias_B*temp)

    Args:
        bias_A: Array bias_A 
        bias_B: Array bias_B 
        header: header immagine considerata, per estrarre temp

    Returns:
        Array di bias da sottrarre all'immagine 

    '''
    temp = float(header['APS_TEMP'])
    return (bias_A + bias_B*temp)

