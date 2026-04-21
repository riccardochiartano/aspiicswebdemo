# Owned
__author__ = "Aleksandr Burtovoi (UNIFI - Dipartimento di Fisica e Astronomia) & Giovanna Jerse (INAF - Osservatorio Astronomico di Trieste)"
__copyright__ = "TBD"
__credits__ = [""]
__license__ = "GPL"
__maintainer__ = "Aleksandr Burtovoi & Giovanna Jerse"
__email__ = "aleksandr.burtovoi@unifi.it & giovanna.jerse@inaf.it"
__status__ = "Released for test"

# Change log
# Date       ver      Description
# ---------- -------- ---------------------------------------------------------
# 17-02-2025    0.0.1 First release
# 19-02-2025    0.0.2 Second test
# 24-02-2025    0.0.3 Third test
# 25-02-2025    0.0.4 Fourth test
# 28-02-2025    0.0.5 Fifth test
# 02-04-2025    0.0.6 Adjusting colormaps, implementing cart_to_polar, 
#                     polar_to_cart, mask_bad_pix
# 10-06-2025    0.0.7 Writing documentation
# 03-09-2025    0.0.8 Implementing Giovanna's comments
__version__ = "0.0.8.20250915"


import warnings
import numpy as np
from scipy import ndimage
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize # modAB
from astropy.coordinates.representation import (
    CartesianRepresentation, 
    CylindricalRepresentation,
)
from sunpy.map import GenericMap



class METISMap(GenericMap):
    """
    Solar Orbiter Metis Image Map

    Metis is the multi-wavelength coronagraph for the Solar Orbiter mission that
    investigates the global corona in polarized visible light and in ultraviolet 
    light. In the ultraviolet band, the coronagraph obtains monochromatic images 
    in the Lyman-alpha line, at 121.6 nm, emitted by the few neutral-hydrogen 
    atoms surviving in the hot corona.

    By simulating a solar eclipse, Metis observes the faint coronal light in an 
    annular zone 1.6-2.9 deg wide, around the disk center. When Solar Orbiter 
    is at its closest approach to the Sun, at the minimum perihelion of 0.28 
    astronomical units, the annular zone is within 1.7 and 3.1 solar radii from 
    disk center.
    
    Solar Orbiter was successfully launched on February 10th, 2020. 

    References
    ----------
    * `Metis Instrument Page <https://metis.oato.inaf.it/index.html>`_
    * `Solar Orbiter Mission Page <https://sci.esa.int/web/solar-orbiter/>`_
    """

    def __init__(self, data, header, **kwargs):
        """
        Initialize the METISMap class with the provided data and header. 
        Validate that the header contains the required parameters.
        """

        if 'RSUN_OBS' in header or 'SOLAR_R' in header or 'RADIUS' in header:
            pass
        else:
            header['RSUN_OBS'] = header['RSUN_ARC']

        # Call the superclass (GenericMap) to initialize the map
        super().__init__(data, header, **kwargs)
        
        self._nickname = f"{self.instrument}/{self.meta['filter']}"
        self._prodtype = self.get_prodtype()
        self._contr_cut = self.get_contr_cut()
        self.update_plot_norm_settings() # modAB
        self.plot_settings['cmap'] = metis_color_table(self._get_cmap_name()) #modAB


    def get_prodtype(self):
        """
        Define the type of the Metis data product.

        Returns
        -------
        prodtype : `str`
            Name of the Metis data product.

        """
        
        btype_suff_dict = {
            'VL total brightness':             ('-TB', '-TB'), 
            'VL polarized brightness':         ('-PB', '-PB'), 
            'VL fixed-polarization intensity': ('-FP', '-Fix. Pol.'), 
            'VL polarization angle':           ('-PA', '-Pol. Angle'), 
            'Stokes I':                        ('-SI', '-Stokes I'), 
            'Stokes Q':                        ('-SQ', '-Stokes Q'), 
            'Stokes U':                        ('-SU', '-Stokes U'),
            'Pixel quality':                   ('-PQ', '-Pixel quality'), 
            'Absolute error':                  ('-AE', '-Abs. err.'),
            'Relative error':                  ('-RE', '-Rel. err.'), #modAB2
            'UV Lyman-alpha intensity':        ('', ''),
        }
        
        btype = self.meta['btype']
        prodtype = self.meta['filter']
        
        if btype in btype_suff_dict:
            suff, nickname_add = btype_suff_dict[btype]
            prodtype += suff
            self._nickname += nickname_add 
        else:
            raise ValueError(
                f"Error. self.meta['btype']='{btype}' is not known."
            ) 

        return prodtype


    @property
    def prodtype(self):
        return self._prodtype


    @prodtype.setter
    def prodtype(self, value):
        raise AttributeError('Cannot manually set prodtype for METISMap')


    def get_contr_cut(self):
        """
        Define the contrast of the Metis data product.

        Returns
        -------
        contr_cut : `float` or `None`
            Contrast of the Metis data product.

        """
        if 'L2' in self.meta['level']:
            if self.prodtype == 'VL-TB' or self.prodtype == 'VL-SI':
                contr_cut = 0.05
            elif self.prodtype == 'VL-PB':
                contr_cut = 0.005
            elif self.prodtype == 'VL-FP':
                contr_cut = 0.01
            elif self.prodtype == 'VL-PA':
                contr_cut = 0.01
            elif self.prodtype == 'VL-SQ':
                contr_cut = 0.01
            elif self.prodtype == 'VL-SU':
                contr_cut = 0.01
            elif self.prodtype == 'UV':
                contr_cut = 0.05  # 0.03
            elif self.prodtype == 'VL-PQ' or self.prodtype == 'UV-PQ':
                contr_cut = 0.0
            elif self.prodtype == 'VL-AE' or self.prodtype == 'UV-AE':
                contr_cut = 0.1
            elif self.prodtype == 'VL-RE' or self.prodtype == 'UV-RE':
                contr_cut = 0.02
            else:
                contr_cut = 0.0
        else:
            contr_cut = 0.0
        return contr_cut


    @property
    def contr_cut(self):
        return self._contr_cut

    
    @contr_cut.setter
    def contr_cut(self, value):
        self._contr_cut = value 

    #modAB
    def update_plot_norm_settings(self):
        img_vlim = self.get_img_vlim()
        self.plot_settings['norm'] = ImageNormalize(
            vmin=img_vlim[0], vmax=img_vlim[1]
        )
    #modAB

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """
        Determine whether the data is a Metis product.

        Returns
        -------
        `bool`
            ``True`` if data corresponds to a Metis product, otherwise 
            ``False``.

        """
        instrume = header.get('INSTRUME', '').strip().upper()
        obsrvtry = header.get('OBSRVTRY', '').strip().upper()
        return ('METIS' in instrume) and ('SOLAR ORBITER' in obsrvtry)


    def get_fov_rsun(self):
        """
        Return the Metis field of view in solar radii.

        Returns
        -------
        `tuple` : `(float, float, float)`
            Inner, outer radii of the field, determined by the internal 
            occulter, field stop and detector size, respectively.
        
        """
        rsun_deg = self.rsun_obs.value / 3600.0  # in deg
        rmin_rsun = self.meta['inn_fov'] / rsun_deg  # in rsun
        rmax_rsun = self.meta['out_fov'] / rsun_deg  # in rsun
        board_deg = 2.9  # deg
        board_rsun = board_deg / rsun_deg  # in rsun
        return rmin_rsun, rmax_rsun, board_rsun


    def mask_occs(self, mask_val=np.nan):
        """
        Mask the data in regions obscured by internal and external occulters.
    
        Parameters
        ----------
        mask_val : `float`, optional
            The values of masked pixels (outside the field of view). Default is 
            ``np.nan``.

        """
        if self.meta['cdelt1'] != self.meta['cdelt2']:
            warnings.warn('Warning: CDELT1 != CDELT2 for {fname}'.format(
                fname=self.meta['filename'])
            )
            print('\t>>> exiting mask_occs method.')
            return

        inn_fov = self.meta['inn_fov'] * 3600 / self.meta['cdelt1']  # in pix
        out_fov = self.meta['out_fov'] * 3600 / self.meta['cdelt2']  # in pix
        x = np.arange(0, self.meta['naxis1'], 1)
        y = np.arange(0, self.meta['naxis2'], 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        dist_suncen = np.sqrt(
            (xx-self.meta['sun_xcen'])**2 + (yy-self.meta['sun_ycen'])**2
        )
        dist_iocen = np.sqrt(
            (xx-self.meta['io_xcen'])**2 + (yy-self.meta['io_ycen'])**2
        )
        self.data[dist_iocen < inn_fov] = mask_val
        self.data[dist_suncen > out_fov] = mask_val
        self.update_plot_norm_settings() # modAB


    def mask_bad_pix(self, qmat, mask_val=np.nan):
        """
        Mask bad-quality pixels in the Metis image.
    
        Parameters
        ----------
        qmat : `numpy.ndarray`
            Pixel quality matrix with the size of original image and the 
            following values: ``1`` - linear range (good-quality), ``0`` - 
            close to 0 counts/close to saturation (bad-quality) and ``np.nan`` -
            0 count/saturated pixels (bad-quality).
        mask_val : `float`, optional
            The values of masked pixels. Default is ``np.nan``.

        """

        if qmat.shape != self.data.shape:
            warnings.warn('Warning: Pixel quality matrix and the METISMap data have different size')
            print('\t>>> exiting mask_bad_pix method.')
            return
        qmat_mask = qmat == 1
        self.data[~qmat_mask] = mask_val
        self.update_plot_norm_settings() # modAB


    def _get_cmap_name(self):
        """
        Override the default implementation to handle Metis color maps.

        Returns
        -------
        cmap_string : `str`
            Name of the Metis data product.
        
        """

        cmap_string = '{obsv}{instr}{prod}'.format(
            obsv=self.observatory, # self.observatory.replace(' ', '_'), 
            instr=self.instrument,
            prod=self.prodtype
        )
        cmap_string = cmap_string.lower()
        return cmap_string


    def get_img_vlim(self):
        """
        Return the intensity limits of the Metis image.

        Returns
        -------
        `tuple` : `(float, float)`
            The minimum and maximum intensity values
        
        """

        vlim = AsymmetricPercentileInterval(
            self.contr_cut*100, (1-self.contr_cut)*100
        ).get_limits(self.data)
        return vlim


    ''' modAB
    def plot(self, **kwargs):
        """
        Override the default implementation to handle Metis color maps and 
        contrast.
        
        """


        if self.contr_cut is None:
            clip_interval = None
        else:
            clip_interval = [self.contr_cut*100, (1-self.contr_cut)*100] * u.percent
        cmap_name = self._get_cmap_name()
        self.plot_settings['cmap'] = metis_color_table(cmap_name)  ### ??? temp solution until metis cmaps are not registred in sunpy
        return super().plot(clip_interval=clip_interval, **kwargs)
    '''        # modAB



#########################
### General functions ###
#########################


def get_linspace_arr(val1, val2, dval):
    """
    Return evenly spaced numbers over a specified interval and step.
    This function is aimed to overcome rounding problems when creating
    arrays with ``numpy.linspace`` or ``numpy.arange`` funcitons.
    
    Parameters
    ----------
    val1 : `float`
        The starting value of the sequence.
    val2 : `float`
        The end value of the sequence.
    dval : `float`
        Spacing between values.
    
    Returns
    -------
    polar_img_arr : `numpy.ndarray`
        Array of evenly spaced values.
    
    """
    
    num = (val2 - val1) / dval
    num = np.around(num, decimals=0)
    arr = np.linspace(val1, val2, int(num)+1)
    return arr


def cart_to_polar(img_arr, r_arr, pa_arr, xc, yc, rsun_pix, rot_angle=None, 
                  interp_order=1, cval=np.nan):
    """
    Convert image array from Cartesian (`Rsun`,`Rsun`) to polar (`Rsun`, `deg`) 
    coordinates by means of ``CylindricalRepresentation`` object of 
    ``astropy.coordinates.representation`` library.
    
    Parameters
    ----------
    img_arr : `numpy.ndarray`
        The input array with initial image in Cartesian coordinates.
    r_arr : `numpy.ndarray`
        The array of values of polar coordinate `Rsun`, at which the output 
        polar image array will be calculated.
    pa_arr : `numpy.ndarray`
        The array of values of polar coordinate `polar angle`, at which the 
        output polar image array will be calculated.
    xc : `float`
        The first Cartesian coordinate (along `x`-axis) of the pole of 
        the polar coordinate system.
    yc : `float`
        The second Cartesian coordinate (along `y`-axis) of the pole of 
        the polar coordinate system.
    rsun_pix : `float`
        The radius of the Sun in pixels of ``img_arr``.
    rot_angle : `float`, optional
        If not None, the rotation angle in degrees, applied counter clockwise. 
        Default is None.
    interp_order : `int`, optional
        The order of the spline interpolation used in function
        ``ndimage.map_coordinates``, default is 1. The order has to be in the 
        range 0-5.
    cval : `float`, optional
        The value to fill past edges of input. Default is 0.0.

    Returns
    -------
    polar_img_arr : `numpy.ndarray`
        The result of transforming the input ``img_arr`` to polar coordinates. 
    
    """
    
    r_pix_arr = r_arr * rsun_pix
    if rot_angle is not None:
        pa_arr = pa_arr - rot_angle
    pa_matrix, r_matrix = np.meshgrid(pa_arr, r_pix_arr)
    polar_repr = CylindricalRepresentation(
        r_matrix, pa_matrix*u.deg, np.zeros(r_matrix.shape)
    )
    cart_repr = polar_repr.to_cartesian()
    polar_inds = np.array([cart_repr.y + yc, cart_repr.x + xc])  #  NB: index sequence not like in IDL: img_arr[x,y]
    polar_img_arr = ndimage.map_coordinates(
        img_arr, polar_inds, order=interp_order, cval=cval
    )
    return polar_img_arr



def polar_to_cart(img_arr, cart_img_shape, xc, yc, rsun_pix, r_lims, dr, 
                  pa_lims, dpa, rot_angle=None, interp_order=1, cval=np.nan):
    """
    Convert image array from polar (`Rsun`, `deg`) to Cartesian (`Rsun`,`Rsun`)
    coordinates by means of CartesianRepresentation object and 
    CylindricalRepresentation object of astropy.coordinates.representation 
    library.

    Parameters
    ----------
    img_arr : `numpy.ndarray`
        The input array with initial image in polar coordinates.
    cart_img_shape : `array_like`
        The array, list or tuple which define a shape of the output 
        Cartesian image array.
    xc : `float`
        The first Cartesian coordinate (along `x`-axis) of the Sun center.
    yc : `float`
        The second Cartesian coordinate (along `x`-axis) of the Sun center.
    rsun_pix : `float`
        The radius of the Sun in pixels of ``img_arr``.
    r_lims : `list`, `[float, float]`
        The lower and upper limits (in Rsun) of the field of view covered by 
        the output image array. It corresponds to the range of heliocentric 
        distances covered by polar ``img_arr``.
    dr : `float`
        Sampling width (in Rsun) of the polar ``img_arr`` along the radial 
        r-axis .
    pa_lims : list`, `[float, float]`
        The lower and upper limits (in degrees) of the field of view covered by 
        the output image array. 
    dpa : `float`
        Sampling width (in degrees) of the polar ``img_arr`` along the polar 
        phi-axis .
    rot_angle : `float`, optional
        If not None, the rotation angle in degrees, applied counter clockwise. 
        Default is None.
    interp_order : `int`, optional
        The order of the spline interpolation used in function
        ``ndimage.map_coordinates``, default is 1. The order has to be in the 
        range 0-5.
    cval : `float`, optional
        The value to fill past edges of input. Default is 0.0.

    Returns
    -------
    cart_img_arr : `numpy.ndarray`
        The result of transforming the input ``img_arr`` to Cartesian 
        coordinates. 
    
    """
    
    n_xpix, n_ypix = cart_img_shape
    r1, r2 = r_lims
    pa1, pa2 = pa_lims

    if pa1 == 0.0 and pa2 == 359.0:
        ndimage_mode = 'grid-wrap'
        mask_occs = True
    else:
        ndimage_mode = 'constant'
        mask_occs = False

    x_arr = np.arange(int(n_xpix))
    y_arr = np.arange(int(n_ypix))
    x_matrix, y_matrix = np.meshgrid(x_arr, y_arr)
    cart_repr = CartesianRepresentation(
        x_matrix-xc, y_matrix-yc, np.zeros(x_matrix.shape)
    )
    cylin_repr = CylindricalRepresentation(1, 1*u.deg, 0)  # defining an object of the class CylindricalRepresentation
    polar_repr = cylin_repr.from_cartesian(cart_repr)  # converting CartesianRepresentation 'cart_repr' to the class of object 'cylin_repr': i.e. CylindricalRepresentation
    polar_repr_r = (polar_repr.rho/rsun_pix - r1)/dr
#     polar_repr_pa = polar_repr.phi.to_value('deg')
    polar_repr_pa = (polar_repr.phi.to_value('deg') - pa1)/dpa
    if rot_angle is not None:
        polar_repr_pa -= rot_angle
    polar_repr_pa[polar_repr_pa < 0.0] += 360.0/dpa
    cart_inds = np.array([polar_repr_r, polar_repr_pa])
    cart_img_arr = ndimage.map_coordinates(
        img_arr, cart_inds, mode=ndimage_mode, order=interp_order, cval=cval
    ) 
    if mask_occs:
        fov1 = r1 * rsun_pix
        fov2 = r2 * rsun_pix
        x = np.arange(0, cart_img_arr.shape[1], 1)
        y = np.arange(0, cart_img_arr.shape[0], 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        dist_cen = np.sqrt((xx-xc)**2 + (yy-yc)**2)
        cart_img_arr[dist_cen < fov1] = cval
        cart_img_arr[dist_cen > fov2] = cval    
    return cart_img_arr


def get_img_arr_polar(smap, r_lims, dr, pa_lims, dpa, rotate=True, 
                      interp_order=1, cval=np.nan):
    """
    Return polar image of the METISMap.
    
    Parameters
    ----------
    smap : `METISMap`
        The input METISMap.
    r_lims : `list`, `[float, float]`
        The lower and upper limits (in Rsun) of the field of view covered by 
        the output image array. It corresponds to the range of heliocentric 
        distances covered by polar ``img_arr``.
    dr : `float`
        Sampling width (in Rsun) of the polar ``img_arr`` along the radial 
        r-axis .
    pa_lims : list`, `[float, float]`
        The lower and upper limits (in degrees) of the field of view covered by 
        the output image array. 
    dpa : `float`
        Sampling width (in degrees) of the polar ``img_arr`` along the polar 
        phi-axis.
    rotate : `bool`, optional
        Determines whether the METISMap image is rotated to have the solar north 
        pole oriented upward. Default is True.
    interp_order : `int`, optional
        The order of the spline interpolation used in function
        ``ndimage.map_coordinates``, default is 1. The order has to be in the 
        range 0-5.
    cval : `float`, optional
        The value to fill past edges of input. Default is 0.0.

    Returns
    -------
    img_arr_polar : `numpy.ndarray`
        Array with the polar image.
    
    """
    
    if rotate:
        smap = smap.rotate()

    xc, yc = smap.world_to_pixel(
        SkyCoord(0*u.arcsec, 0*u.arcsec, frame=smap.coordinate_frame)
    )  # center of coordinate_frame: Sun center in pixels
    xc = xc.value
    yc = yc.value
    if smap.scale.axis1 != smap.scale.axis2:
        raise ValueError(
            'Error. Scales of the map (CDELT[1/2]) are different'
        )
    rsun_pix = smap.rsun_obs/smap.scale.axis1
    rsun_pix = rsun_pix.value
    
    r_arr = get_linspace_arr(r_lims[0], r_lims[1], dr)
    pa_arr = get_linspace_arr(pa_lims[0], pa_lims[1], dpa)
    
    img_arr_polar = cart_to_polar(
        smap.data, r_arr, pa_arr, xc, yc, rsun_pix, interp_order=interp_order,
        cval=cval
    )
    return img_arr_polar


def set_rsun_pa_axes(ax, img_arr, r_lims, dr, pa_lims, dpa, labelfontsize=13, 
                     labelsize=10):
    """
    Convert `x`- and `y`- axes of input ``ax`` from pixels to polar angle and 
    Rsun, respectively.
    
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The input axes.
    img_arr : `numpy.ndarray`
        The polar image array plotted in ``ax``.
    r_lims : `list`, `[float, float]`
        The lower and upper limits (in Rsun) of the field of view covered by 
        the output image array. It corresponds to the range of heliocentric 
        distances covered by polar ``img_arr``.
    dr : `float`
        Sampling width (in Rsun) of the polar ``img_arr`` along the radial 
        r-axis .
    pa_lims : list`, `[float, float]`
        The lower and upper limits (in degrees) of the field of view covered by 
        the output image array. 
    dpa : `float`
        Sampling width (in degrees) of the polar ``img_arr`` along the polar 
        phi-axis.
    labelfontsize : `float`, optional
        The fontsize of the axes labels. Default is 13.
    labelsize : `float`, optional
        The fontsize of the tick labels. Default is 10.
    
    """
    
    r1, r2 = r_lims
    pa1, pa2 = pa_lims

    ax.set_xticks([])
    secax_x = ax.secondary_xaxis(
        'bottom', 
        functions=(lambda x: pa1 + x*dpa, lambda x: (x-pa1)/dpa)
    )
    #secax_x.set_xticks(np.arange(pa1, pa2, 90))
    secax_x.set_xticks([pa1, 90, 180, 270, pa2])
    secax_x.tick_params(axis='x', which='major', labelsize=labelsize)

    ax.set_yticks([])
    secax_y = ax.secondary_yaxis(
        'left', 
        functions=(lambda y: r1 + y*dr, lambda y: (y-r1)/dr)
    )
    secax_y.set_yticks(np.linspace(r1, r2, 5))
    secax_y.tick_params(axis='y', which='major', labelsize=labelsize)

    secax_x.set_xlabel('Polar angle [deg]', fontsize=labelfontsize)
    secax_y.set_ylabel(r'$R_\odot$', fontsize=labelfontsize)


#######################
### Temp. Functions ###
#######################

# temp function to fine tune clip_intervals in plot()
def get_img_clim(img_arr, contr_cut, n_std=2, bins=1000, show_plot=False):
    """
    Calculate intensity range of an image to be provided to 
    ``matplotlib.pyplot.imshow``. The flattened input array is converted into a 
    histogram with a number of bins specified by `bins`, considering pixel 
    intensities within a range of +/- `n_std` standard deviations from the mean 
    of the pixel intensity distribution. 
    
    Parameters
    ----------
    img_arr : `numpy.ndarray`
        Input array with image.
    contr_cut : `float`
        
        Contrast cut defines a fraction of the maximum 
    
    Returns
    -------
    eval_arr : ``numpy.ndarray``
        Output array.
    
    """
    
    img_arr_mean = np.nanmean(img_arr)
    img_arr_std = np.nanstd(img_arr)
    
    if np.isnan(img_arr_mean) and np.isnan(img_arr_std):
        return np.nan, np.nan
    
    img_hist, img_bins = np.histogram(
        img_arr[img_arr!=0.0].flatten(), bins=bins,
        range=(
            img_arr_mean - n_std*img_arr_std, 
            img_arr_mean + n_std*img_arr_std
        )
    )
    img_bins_arr = np.vstack((img_bins[:-1],img_bins[1:])).transpose()
    img_hist_mask = img_hist >= contr_cut*np.nanmax(img_hist)
    img_bins_arr_cutted = img_bins_arr[img_hist_mask]

    img_vmin = np.nanmin(img_bins_arr_cutted[:,0])
    img_vmax = np.nanmax(img_bins_arr_cutted[:,1])
    
    if show_plot:
        img_hist_cutted = img_hist[img_hist_mask]
        img_bincen = (img_bins[:-1]+img_bins[1:])/2.0
        img_bincen_cutted = img_bincen[img_hist_mask]
        fig_h, ax_h = plt.subplots(2, 1)
        img_h = ax_h[0].imshow(
            img_arr, origin='lower', vmin=img_vmin, vmax=img_vmax
        )
        ax_h[1].plot(img_bincen, img_hist, drawstyle='steps-mid')
        ax_h[1].plot(
            img_bincen_cutted, img_hist_cutted, drawstyle='steps-mid', ls='--'
        )
        ax_h[1].axvline(img_vmin, color='k', ls=':')
        ax_h[1].axvline(img_vmax, color='k', ls=':')
        ax_h[1].axvline(
            np.nanmean(img_arr) - n_std*np.nanstd(img_arr), color='k', ls='--'
        )
        ax_h[1].axvline(
            np.nanmean(img_arr) + n_std*np.nanstd(img_arr), color='k', ls='--'
        )
        plt.show()

    return img_vmin, img_vmax
    


def metis_color_table(cmap_name):
    """
    Credits: V. Andretta, A. Liberatore, A. Burtovoi, G. Jerse
    NB:
     - Names from _get_cmap_name()_ should be defined in sunpy.visualization.colormaps.cm
     - They can be in turn defined by calling metis_color_table() inserted in sunpy.visualization.colormaps.color_tables
     - Current function is a prototype of function which should be inserted in sunpy.visualization.colormaps.color_tables
     - [?] Should we define different colormaps for L0 and L1?
    """
    ### Temporary imports ###
    import matplotlib
    import sunpy.visualization.colormaps as cm
    import cmcrameri

#     print(cmap_name)

    if cmap_name == 'solar orbitermetisvl-tb':
# #         aia_wave_dict = create_aia_wave_dict()
#         aia_wave_dict = cm.color_tables.create_aia_wave_dict()  # temp
#         r, g, b = aia_wave_dict[193*u.angstrom]
#         cmap = cm.color_tables._cmap_from_rgb(
#             r, g, b, 'SolO Metis VL Total Brightness'
#         )
        #cmap = matplotlib.colormaps['pink'].copy()  # ASk Vincenzo ???  # np.savetxt('py_cmap_pink.csv', np.array(cmap.colors)*255, delimiter=',')
        #cmap = cmcrameri.cm.batlow.copy()
        cmap = cmcrameri.cm.batlow.copy()
        # chk also Stokes I
        cmap.name =  'SolO Metis VL Total Brightness'
        
    elif cmap_name == 'solar orbitermetisvl-pb':
        '''
        Metis VL/pB images uses AIA color table
        '''
#         aia_wave_dict = create_aia_wave_dict()
        aia_wave_dict = cm.color_tables.create_aia_wave_dict()  # temp
        r, g, b = aia_wave_dict[304*u.angstrom]
        cmap = cm.color_tables._cmap_from_rgb(
            r, g, b, 'SolO Metis VL Polarized Brightness'
        )

    elif cmap_name == 'solar orbitermetisvl-fp':
        r = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        g = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        cmap = cm.color_tables._cmap_from_rgb(
            r, g, b, 'SolO Metis VL Fixed Polarization'
        )

    elif cmap_name == 'solar orbitermetisvl-pa':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Polarization Angle'

    elif cmap_name == 'solar orbitermetisvl-si':
        #cmap = matplotlib.colormaps['pink'].copy()  # ASk Vincenzo ???  # np.savetxt('py_cmap_pink.csv', np.array(cmap.colors)*255, delimiter=',')
        cmap = cmcrameri.cm.batlow.copy()
        cmap.name =  'SolO Metis VL Stokes I'

    elif cmap_name == 'solar orbitermetisvl-sq':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Stokes Q'

    elif cmap_name == 'solar orbitermetisvl-su':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Stokes U'

    elif cmap_name == 'solar orbitermetisvl-pq':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV PQ
        cmap.name =  'SolO Metis VL Pixel Quality'

    elif cmap_name == 'solar orbitermetisvl-ae':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV AE
        cmap.name =  'SolO Metis VL Absolute Error'

    elif cmap_name == 'solar orbitermetisvl-re':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV RE
        cmap.name =  'SolO Metis VL Relative Error'
        
    elif cmap_name == 'solar orbitermetisuv':
        cmap = matplotlib.colormaps['Blues_r'].copy()  # Blues_r, PuBu, BuGn
        cmap.name =  'SolO Metis UV'

    elif cmap_name == 'solar orbitermetisuv-pq':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL PQ
        cmap.name =  'SolO Metis UV Pixel Quality'

    elif cmap_name == 'solar orbitermetisuv-ae':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL AE
        cmap.name =  'SolO Metis UV Absolute Error'

    elif cmap_name == 'solar orbitermetisuv-re':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL RE
        cmap.name =  'SolO Metis UV Relative Error'


    cmap.set_bad(color='k')
    
    return cmap
    


#####################
### Test Function ###
#####################    

def run_test(fits_path):

    print('\n =========== run_test ===========')
    print()
    print(fits_path)
    
    ### Example usage: ###
    
    import matplotlib.pyplot as plt
    import sunpy.map
    
    smap_list = sunpy.map.Map(fits_path)
    if len(smap_list) == 3:
        metis_smap_img, metis_smap_qmat, metis_smap_emat = smap_list
        metis_smap_img2 = None
        metis_smap_img3 = None
    elif len(smap_list) == 5:
        metis_smap_img, metis_smap_img2, metis_smap_img3, metis_smap_qmat, metis_smap_emat = smap_list
    else:
        raise ValueError(
            'Error. Not expected number of header data units in the fits file: {fits_path}'.format(
                fits_path=fits_path
            )
        )

    ### Print map information ###
    
    print()
    print(metis_smap_img.meta)
    print()
    print(metis_smap_qmat)
    print()
    print(metis_smap_emat)


    ### Masking Metis Occulter ###
    
    metis_smap_img.mask_occs()
    metis_smap_qmat.mask_occs()
    metis_smap_emat.mask_occs()
    if metis_smap_img2 is not None:
        metis_smap_img2.mask_occs()
    if metis_smap_img3 is not None:
        metis_smap_img3.mask_occs()


    ### Masking Metis Bad pixels ###
    
#     metis_smap_img.mask_bad_pix(metis_smap_qmat.data)
#     if metis_smap_img2 is not None:
#         metis_smap_img2.mask_bad_pix(metis_smap_qmat.data)
#     if metis_smap_img3 is not None:
#         metis_smap_img3.mask_bad_pix(metis_smap_qmat.data)


    ### Visualize the map ###
    
    metis_smap_img.peek()
    metis_smap_qmat.peek()
    metis_smap_emat.peek()
    if metis_smap_img2 is not None:
        metis_smap_img2.peek()
    if metis_smap_img3 is not None:
        metis_smap_img3.peek()


    ### Conversion from cart to polar coordinates ###

    rmin_rsun, rmax_rsun, board_rsun = metis_smap_img.get_fov_rsun()
    r_lims =[rmin_rsun, rmax_rsun]
    dr = 0.01 
    pa_lims = [0.0, 359.0] # [90.0, 215.0] # [45.0, 300.0] # 
    dpa = 1.0  # 1.0
    metis_img_arr_polar = get_img_arr_polar(
        metis_smap_img, r_lims, dr, pa_lims, dpa
    )

    fig1, ax1 = plt.subplots(1, 1, tight_layout=True)
    metis_img_arr_polar_vlim = metis_smap_img.get_img_vlim()
    metis_img_polar = ax1.imshow(
        metis_img_arr_polar, origin='lower', vmin=metis_img_arr_polar_vlim[0], 
        vmax=metis_img_arr_polar_vlim[1], 
    )
    cbar_polar = fig1.colorbar(metis_img_polar, ax=ax1) #, fraction=0.035)
    cbar_polar.set_label('MSB', fontsize=13)
    set_rsun_pa_axes(ax1, metis_img_arr_polar, r_lims, dr, pa_lims, dpa)
    ax1.set_title('Metis image (polar) \n' + metis_smap_img.meta['filename'])
    ax1.set_aspect(
        np.shape(metis_img_arr_polar)[1]/np.shape(metis_img_arr_polar)[0]
    )
    
    ## TBD: saving polar_fits ???



    ### Conversion polar coordinates ###
    
    cart_img_shape = metis_smap_img.data.shape
    rsun_pix = metis_smap_img.rsun_obs/metis_smap_img.scale.axis1
    rsun_pix = rsun_pix.value
    metis_img_arr_cart = polar_to_cart(
        metis_img_arr_polar, cart_img_shape, cart_img_shape[0]/2, 
        cart_img_shape[1]/2, rsun_pix, r_lims, dr, pa_lims, dpa,
    )

    fig2, ax2 = plt.subplots(1, 1, tight_layout=True)
    img_extent = 1.0/rsun_pix * np.array(
        [-cart_img_shape[0]/2.0, cart_img_shape[0]/2.0,
         -cart_img_shape[1]/2.0, cart_img_shape[1]/2.0]
    )
    metis_img_cart = ax2.imshow(
        metis_img_arr_cart, origin='lower', vmin=metis_img_arr_polar_vlim[0], 
        vmax=metis_img_arr_polar_vlim[1], extent=img_extent
    )
    cbar_cart = fig2.colorbar(metis_img_cart, ax=ax2) #, fraction=0.05)
    cbar_cart.set_label('MSB', fontsize=13)
    ax2.set_title('Metis image (cart) \n' + metis_smap_img.meta['filename'])
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_xlabel(r'$R_\odot$', fontsize=13)
    ax2.set_ylabel(r'$R_\odot$', fontsize=13)
    
    ## TBD: saving cart_fits ???

    plt.show()
    
    

#     # Register the new instrument class
#     sunpy.map.sources.RegisteredMapClasses.append(METISMap)

    ### AB tests ###

    print('\ntest0:')
    print(' >', type(metis_smap_img))
    print(' >', type(metis_smap_qmat))
    print(' >', type(metis_smap_emat))

    print('\ntest1:')
    print(' >', metis_smap_img.meta['CUNIT1'], metis_smap_img.meta['CTYPE1'])
    print(' >', metis_smap_img.meta['CUNIT2'], metis_smap_img.meta['CTYPE2'])

    print('\ntest2:')
    print(' >', metis_smap_img.meta['RSUN_REF'])
    
    print('\ntest3:')
    print(' >', metis_smap_img.observatory)
    
    print('\ntest4:')
    print(' >', metis_smap_img.detector)
    
    print('\ntest5:')
    print(' >', metis_smap_img.wavelength)
    
    print('\ntest6:')
    print(' >', metis_smap_img.rsun_obs)

    print('\n ================================')


if __name__ == '__main__':


    run_test('./content/solo_L2_metis-vl-image_20241123T233047_V01.fits')
#     run_test('./content/solo_L2_metis-uv-image_20241130T180707_V01.fits')
#     run_test('./content/solo_L2_metis-vl-tb_20241130T181849_V01.fits')
#     run_test('./content/solo_L2_metis-vl-pb_20241125T212401_V01.fits')
#     run_test('./content/solo_L2_metis-vl-pol-angle_20241125T230001_V01.fits')
#     run_test('./content/solo_L2_metis-vl-stokes_20241125T202401_V01.fits')
    