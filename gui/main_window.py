import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from sunpy.map import GenericMap
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import tempfile
import plotly.express as px
import datetime
import io
from streamlit_plotly_events import plotly_events
from sunpy.coordinates import SphericalScreen

from logic.process import wow_filter, log_scale, nrgf_filter, mgn_filter, unsharp_mask_filter
from logic.plot import plot_profile, aspiics_cmap, aspiics_cmap_new, plot_NE_labels, plot_pprof, plot_rprof, plot_stars, fix_observer_if_opposite
from logic.utils import sun_center, download_map_btn, file_to_smap, aspiics_files_url, aspiics_files_api, download_web_files_btn, header_from_sunpymap, search_stars
from gui.profile_window import profile_window
from gui.fits_canvas import draw_extract_panel
import logic.rob.aspiics_misc as am

#from resources.metismap.metis_map_vAB import METISMap

# cache
@st.cache_data(show_spinner=False)
def load_map_from_file(file_bytes, filename):
    uploaded_file = io.BytesIO(file_bytes)
    uploaded_file.name = filename
    return file_to_smap(uploaded_file)

@st.cache_data(show_spinner=False)
def load_map_from_url(url):
    solar_map = sunpy.map.Map(url)
    solar_image = solar_map.data
    if not np.issubdtype(solar_image.dtype, np.floating):
        solar_image = solar_image.astype(float)
    solar_map = sunpy.map.Map(solar_image, solar_map.meta)
    return solar_map

@st.cache_data(
    show_spinner=False,
    hash_funcs={sunpy.util.MetaDict: lambda m: m.item_hash()}
)
def apply_filter_cached(data, meta, scale_option,
                        unsharp_radius=5, unsharp_amount=2,
                        wow_bilateral=1, wow_gamma=4, wow_h=0.99):
    map_plot = sunpy.map.Map(data, meta)
    if scale_option == "Log":
        map_plot = log_scale(map_plot)
    elif scale_option == "MGN":
        map_plot = mgn_filter(map_plot)
    elif scale_option == "WOW filter":
        map_plot = wow_filter(map_plot, bilateral=wow_bilateral, gamma=wow_gamma, h=wow_h)
    elif scale_option == "Unsharp masking":
        map_plot = unsharp_mask_filter(map_plot, radius=unsharp_radius, amount=unsharp_amount)
    elif scale_option == "NRGF filter":
        map_plot = nrgf_filter(map_plot)
    return map_plot


tabname = 'main'

def SolarMapViewer():
    if "current_map" not in st.session_state:
        st.session_state.current_map = None
    if "original_map" not in st.session_state:
        st.session_state.original_map = None
    if "demod_maps" not in st.session_state:
        st.session_state.demod_maps = {}
    if "show_header" not in st.session_state:
        st.session_state.show_header = False
    if "show_right_panel" not in st.session_state:
        st.session_state.show_right_panel = False
    if "show_ne" not in st.session_state:
        st.session_state.show_ne = True
    if "show_files" not in st.session_state:
        st.session_state.show_files = False
    if "show_rprof" not in st.session_state:
        st.session_state.show_rprof = False
    if "show_pprof" not in st.session_state:
        st.session_state.show_pprof = False
    if "show_shiftc" not in st.session_state:
        st.session_state.show_shiftc = False
    if "show_grid" not in st.session_state:
        st.session_state.show_grid = True
    if "files_url" not in st.session_state:
        st.session_state.files_url = None
    if "intplot" not in st.session_state:
        st.session_state.intplot = False
    if "rotate_map" not in st.session_state:
        st.session_state.rotate_map = False
    if "xy_diff" not in st.session_state:
        st.session_state.xy_diff = None
    if 'stars' not in st.session_state:
        st.session_state.stars = None
    st.session_state.setdefault("unsharp_radius", 5)
    st.session_state.setdefault("unsharp_amount", 2.0)
    st.session_state.setdefault("wow_bilateral", 1)
    st.session_state.setdefault("wow_gamma", 4.0)
    st.session_state.setdefault("wow_h", 0.99)
    if "canvas_mode" not in st.session_state:
        st.session_state.canvas_mode = False
    st.session_state.setdefault("overlay_map", None)
    st.session_state.setdefault("overlay_enabled", False)

    if st.session_state.show_right_panel:
        col_left, col_main, _, col_right = st.columns([1, 10, 1, 8])
    else:
        col_left, col_main, col_right = st.columns([1, 3, 1])

    if st.session_state.original_map and not(st.session_state.demod_maps):
        smap = st.session_state.original_map
        if st.session_state.rotate_map:
            rotated_im, rotated_head = am.rotate_center1header(smap.data, header_from_sunpymap(smap.meta))
            smap = sunpy.map.Map(rotated_im, rotated_head)
        if st.session_state.xy_diff:
            smap.meta['crpix1'] += st.session_state.xy_diff[0]
            smap.meta['crpix2'] += st.session_state.xy_diff[1]
            st.session_state.xy_diff = None
        st.session_state.current_map = smap
    #elif st.session_state.original_map:
    #    st.session_state.current_map = st.session_state.original_map
    

    #col_left, col_main, _, col_right = st.columns([1, 10, 1, 8])

    with col_main:
        # nothing loaded
        if st.session_state.current_map is None and not st.session_state.demod_maps:
            st.info("No map loaded...")
            uploaded_file = st.file_uploader("Upload .fits file", type=["fits", "fts"], key="fits_uploader")
            if uploaded_file is not None:
                try:
                    #solar_map = file_to_smap(uploaded_file)
                    solar_map = load_map_from_file(uploaded_file.getvalue(), uploaded_file.name)
                    #st.session_state.current_map = solar_map
                    st.session_state.original_map = solar_map
                    st.success(f"Loaded file: {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            st.divider()
            show_webloader()
            return
        else:
            tab_map, tab_profile = st.tabs(['Map', 'Profiles'])
        
        with tab_profile:
            profile_window()

        with tab_map:
            # demod_maps 
            if st.session_state.demod_maps:
                col1, spacer, col2 = st.columns([1, 2, 1])
                map_names = list(st.session_state.demod_maps.keys())
                with col1:
                    selected_map = st.selectbox("Select demodulated map", map_names)
                if selected_map:
                    #st.session_state.current_map = st.session_state.demod_maps[selected_map]
                    smap = st.session_state.demod_maps[selected_map]
                    if st.session_state.rotate_map:
                        rotated_im, rotated_head = am.rotate_center1header(smap.data, header_from_sunpymap(smap.meta))
                        smap = sunpy.map.Map(rotated_im, rotated_head)
                    if st.session_state.xy_diff:
                        smap.meta['crpix1'] += st.session_state.xy_diff[0]
                        smap.meta['crpix2'] += st.session_state.xy_diff[1]
                        st.session_state.xy_diff = None
                        st.write('changed suncenter')
                    st.session_state.current_map = smap
                with col2:
                    download_map_btn(st.session_state.current_map)

            # current map
            #if st.session_state.current_map is not None:
            #    map_plot = sunpy.map.Map(st.session_state.current_map.data, st.session_state.current_map.meta)
            #    image_plot = map_plot.data

            with st.expander("Load overlay map", expanded=False):
                overlay_file = st.file_uploader(
                    "Upload overlay FITS", type=["fits", "fts"],
                    key="overlay_uploader", label_visibility="collapsed"
                )
                if overlay_file is not None:
                    st.session_state.overlay_map = load_map_from_file(
                        overlay_file.getvalue(), overlay_file.name
                    )
                    st.session_state.overlay_enabled = True
                    st.success(f"Loaded: {overlay_file.name}")

            #if st.session_state.overlay_map is not None:
            #    st.caption(f"Overlay: {st.session_state.overlay_map.meta.get('TELESCOP', '?')} — "
            #            f"{st.session_state.overlay_map.meta.get('DATE-OBS', '?')}")
                
            col1, col2 = st.columns([3,1])
            
            with col1:
                scale_option = st.radio("Image filters:",
                    ["Linear", 
                     "Log",
                      #"MGN", 
                      "Unsharp masking", 
                      "WOW filter", 
                      #"NRGF filter (Slow)"
                    ],
                    horizontal=True,
                    label_visibility='collapsed'  
                )

                # parameters
                if scale_option == "Unsharp masking":
                    with st.expander("Unsharp masking parameters", expanded=True):
                        with st.form("unsharp_params_form"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.number_input("Radius", value=5, min_value=1, max_value=20, key="unsharp_radius")
                            with c2:
                                st.number_input("Amount", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="unsharp_amount")
                            st.form_submit_button("Apply")

                if scale_option == "WOW filter":
                    with st.expander("WOW filter parameters", expanded=True):
                        with st.form("wow_params_form"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.number_input("Bilateral", value=1, min_value=0, max_value=5, key="wow_bilateral")
                            with c2:
                                st.number_input("Gamma", value=4.0, min_value=1.0, max_value=10.0, step=0.5, key="wow_gamma")
                            with c3:
                                st.number_input("h", value=0.99, min_value=0.1, max_value=1.0, step=0.01, key="wow_h")
                            st.form_submit_button("Apply")

            with col2:
                st.space(2)



            #col1, col2, col3 = st.columns([1,1,1])
            #with col1:
            #    interactive_plot = st.checkbox('Interactive plot', key='chk_intplot', on_change=toggle_intplot)
            #    rotate_map = st.checkbox('Rotate map', key='rotate_map_ckb', on_change=toggle_rotate)
            #    show_grid = st.checkbox('Show grid', value=True, key='show_grid_ckb', on_change=toggle_grid)
            #with col2:
            #    show_header = st.checkbox('Show header', key='show_header_ckb', on_change=toggle_header)
            #    show_radprofiles = st.checkbox('Show radial profiles', key='sh_rprof_ckb', on_change=toggle_rprof)
            #with col3:
            #    show_NE_labels = st.checkbox('Show N-E labels', key='show_NE_ckb', on_change=toggle_NE)
            #    show_polprofiles = st.checkbox('Show polar profiles', key='sh_pprof_ckb', on_change=toggle_pprof)
            
            #with col1:
            st.write("**Display settings:**")
            map_options = st.segmented_control(
                "Map Display Options",
                options=["Interactive", "Canvas", "Rotate", "Grid", "Header", 
                            "Radial Profiles", "Polar Profiles", "N-E Labels"],
                default=["Grid", "N-E Labels"],
                selection_mode="multi",
                key="map_ctrl",
                on_change=toggle_plot_options,
                label_visibility='collapsed'
            )

            # Gruppo 2: Calibrazione (Azioni/Algoritmi)
            st.write("**Analysis Tools:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.toggle("Star Search", key="star_search_ckb", on_change=toggle_star_src)
            with col_b:
                st.toggle("Shift Solar Center", key="shiftc_ckb", on_change=toggle_shiftc)

            #with col2:
            #    # Opzioni di visualizzazione profili e header
            #    view_options = st.segmented_control(
            #        "Elementi UI",
            #        options=["Header", "Radial Profiles", "Polar Profiles"],
            #        selection_mode="multi",
            #        key="view_ctrl",
            #        #on_change=toggle_view_settings
            #    )
            #
            #with col3:
            #    # Altre label o impostazioni secondarie
            #    label_options = st.segmented_control(
            #        "Labels",
            #        options=["N-E Labels"],
            #        selection_mode="multi",
            #        key="label_ctrl",
            #        on_change=toggle_NE
            #    )
                
            ##st.space(10)
            #col1, col2, col3 = st.columns([1,1,1])
            #with col1:
            #    star_src = st.checkbox('Search stars', key='star_src', on_change=toggle_shiftc)
            #with col3:
            #    shift_c = st.toggle('Shift solar center', key='shiftc_ckb', on_change=toggle_shiftc)
            
            # map container
            map_container = st.container()

            #if scale_option == "Log":
            #    map_plot = log_scale(map_plot) 
            #if scale_option == "MGN":
            #    map_plot = mgn_filter(map_plot) 
            #if scale_option == "WOW filter":
            #    map_plot = wow_filter(map_plot)
            #if scale_option == "Unsharp masking":
            #    map_plot = unsharp_mask_filter(map_plot)
            #if scale_option == "NRGF filter":
            #    map_plot = nrgf_filter(map_plot)
            map_plot = apply_filter_cached(
                st.session_state.current_map.data,
                st.session_state.current_map.meta,
                scale_option,
                wow_bilateral=st.session_state.wow_bilateral,
                wow_gamma=st.session_state.wow_gamma,
                wow_h=st.session_state.wow_h,
                unsharp_radius=st.session_state.unsharp_radius,
                unsharp_amount=st.session_state.unsharp_amount,
            )
            image_plot = map_plot.data   
            #map_plot = sunpy.map.Map(image_plot, st.session_state.current_map.meta)

            # map histogram
            flat_data = image_plot[np.isfinite(image_plot)].flatten()
            if flat_data.size == 0:
                st.info("No valid data")
                st.stop()

            limit_min, limit_max = np.percentile(flat_data, [0.05, 99.95])
            flat_data_clip = flat_data[(flat_data >= limit_min) & (flat_data <= limit_max)]

            col1, col2, col3 = st.columns([1, 7, 1])
            with col2:
                fig_hist, ax_hist = plt.subplots(figsize=(6, 1))
                ax_hist.hist(flat_data_clip, bins=256, color='blue', alpha=0.7, log=True)
                ax_hist.set_yticks([])
                ax_hist.set_ylabel("")                     
                st.pyplot(fig_hist)

            vmin_default, vmax_default = np.percentile(flat_data_clip, [1.0, 99.0])
            step = (limit_max - limit_min) / 1000 if limit_max != limit_min else 1e-8
            vmin, vmax = st.slider(
                "Select vmin/vmax",
                min_value=float(limit_min),
                max_value=float(limit_max),
                value=(float(vmin_default), float(vmax_default)),
                step=step,
                format="%.2e"
            )

            col1, col2, col3 = st.columns([1, 7, 1])
            with col2:
                c1, space, c2 = st.columns(3)
                vmin_manual = c1.number_input("vmin", value=vmin, step=step, format="%.2e")
                vmax_manual = c2.number_input("vmax", value=vmax, step=step, format="%.2e")   
            if vmin_manual != vmin or vmax_manual != vmax:
                vmin, vmax = vmin_manual, vmax_manual

            # plot map
            with map_container: 
                ## OVERLAY ##
                if st.session_state.overlay_enabled and st.session_state.overlay_map is not None:
                    overlay_map = st.session_state.overlay_map

                    overlay_map = fix_observer_if_opposite(overlay_map, map_plot)

                    overlay_map = apply_filter_cached(
                        overlay_map.data,
                        overlay_map.meta,
                        scale_option,
                        wow_bilateral=st.session_state.wow_bilateral,
                        wow_gamma=st.session_state.wow_gamma,
                        wow_h=st.session_state.wow_h,
                        unsharp_radius=st.session_state.unsharp_radius,
                        unsharp_amount=st.session_state.unsharp_amount,
                    )

                    overlay_data = overlay_map.data.copy()
                    overlay_data[overlay_data == 0] = np.nan
                    overlay_map = sunpy.map.Map(overlay_data, overlay_map.meta)

                    def _fov_area(m):
                        width = (m.dimensions[0] * m.scale[0]).to_value(u.arcsec)
                        height = (m.dimensions[1] * m.scale[1]).to_value(u.arcsec)
                        return width * height
                    
                    if _fov_area(overlay_map) >= _fov_area(map_plot):
                        with SphericalScreen(map_plot.observer_coordinate):
                            reprojected = map_plot.reproject_to(overlay_map.wcs)
                        base_map = overlay_map
                        other_map = reprojected
                        other_meta = map_plot.meta
                    else:
                        with SphericalScreen(overlay_map.observer_coordinate):
                            reprojected = overlay_map.reproject_to(map_plot.wcs)
                        base_map = map_plot
                        other_map = reprojected
                        other_meta = overlay_map.meta

                    fig, ax = plt.subplots(
                        figsize=(8, 6),
                        subplot_kw=dict(projection=base_map.wcs)
                    )
                    ax.set_facecolor('black')
                    
                    cmap_base = aspiics_cmap_new(base_map).copy()
                    cmap_base.set_bad(alpha=0)  # NaN → trasparente
                    cmap_other = aspiics_cmap_new(other_map).copy()
                    cmap_other.set_bad(alpha=0)

                    im = base_map.plot(
                        axes=ax, alpha=1.0, autoalign=True, zorder=1,
                        cmap=cmap_base,
                    )
                    other_map.plot(
                        axes=ax, alpha=1.0, autoalign=True, zorder=0,
                        cmap=cmap_other,
                    )

                    title = f"{base_map.meta.get("INSTRUME", "?")} + {other_meta.get("INSTRUME", "?")}, {base_map.meta.get("DATE-OBS", "")}"
                    ax.set_title(title)
                    ax.grid(st.session_state.show_grid, ls='--', lw=0.4)
                    if st.session_state.show_ne:
                        plot_NE_labels(ax, base_map, draw_limb=True, limb_c='gray', limb_s='--')
                    cbar = plt.colorbar(im, ax=ax, label=f"Intensity [{map_plot.meta.get('bunit', '')}]")
                    st.pyplot(fig, width='content')

                ## CANVAS MODE ##
                elif st.session_state.canvas_mode:
                    draw_extract_panel(map_plot, vmin, vmax)

                ## STANDARD PLOT ##
                elif not(st.session_state.intplot):
                    fig_map, ax_map = plt.subplots(figsize=(6,6), subplot_kw={'projection': st.session_state.current_map.wcs})
                    im = map_plot.plot(axes=ax_map, clim=(vmin, vmax), cmap=aspiics_cmap_new(map_plot))
                    #im = ax_map.imshow(map_plot.data,origin='lower',vmin=vmin,vmax=vmax,cmap=aspiics_cmap_new(map_plot))
                    ax_map.grid(st.session_state.show_grid, ls='--', lw=0.4) 
                    #ax_map.grid(True, ls='--', lw=0.4) 
                    #st.write(map_plot)
                    #ax_map.set_xlabel("Solar X [arcsec]")
                    #ax_map.set_ylabel("Solar Y [arcsec]")
                    if st.session_state.show_ne:
                        plot_NE_labels(ax_map, map_plot, draw_limb=True, limb_c='gray', limb_s='--')
                        #map_plot.draw_limb(color='gray', linestyle='--')
                    if st.session_state.show_rprof:
                        plot_rprof(ax_map, map_plot, st.session_state.rad_profiles)
                    if st.session_state.show_pprof:
                        plot_pprof(ax_map, map_plot, st.session_state.pol_profiles)
                    if st.session_state.stars is not None:
                        plot_stars(ax_map, st.session_state.stars)
                    cbar = plt.colorbar(im, ax=ax_map, label=f"Intensity [{map_plot.meta.get('bunit', '')}]")
                    st.pyplot(fig_map, width='content')
                
                ## INTERACTIVE PLOT MODE ##
                else:
                    data = np.array(map_plot.data, dtype=np.float32)
                    factor=1
                    data = data.reshape(
                        data.shape[0]//factor, factor,
                        data.shape[1]//factor, factor
                    ).mean(axis=(1,3))
                    fig = px.imshow(
                        data,
                        color_continuous_scale='gray',
                        zmin=vmin, zmax=vmax,
                        origin="lower",
                        labels={"color": f"Intensity [{map_plot.meta.get('bunit', '')}]"},
                        #title=f"Interactive map — {map_plot.meta.get('wavelnth', '')} Å"
                    )

                    #circle = fig.update_layout(
                    #    dragmode='drawrect')
                    #st.write(circle)
                    #fig.show(config={'modeBarButtonsToAdd':['drawline',
                    #                                        'drawopenpath',
                    #                                        'drawclosedpath',
                    #                                        'drawcircle',
                    #                                        'drawrect',
                    #                                        'eraseshape'
                    #                                    ]})

                    
                    fig.update_layout(
                        width=1200, height=1200,
                        xaxis_title="X [pixels]",
                        yaxis_title="Y [pixels]",
                        coloraxis_showscale=False
                    )
                    
                    points = st.plotly_chart(fig)
                
            with st.container(horizontal=True):
                download_map_btn(map_plot, label="Download current map")
                st.space('stretch')
                if st.button('Reset Image', key='btn_reset_main_image'):
                    st.session_state.original_map = None
                    st.session_state.current_map = None
                    st.session_state.demod_maps = {}
                    st.session_state.overlay_map = None
                    st.session_state.overlay_enabled = False
                    st.session_state.stars = None
                    st.session_state.xy_diff = None
                    st.session_state.files_url = None
                    st.session_state.show_header = False
                    st.session_state.show_right_panel = False
                    st.session_state.show_rprof = False
                    st.session_state.show_pprof = False
                    st.session_state.show_shiftc = False
                    
                    for k in ["unsharp_radius", "unsharp_amount", "wow_bilateral", "wow_gamma", "wow_h",
                            "perp_width", "hist_bins", "spline_k", "spline_s", "spline_samples",
                            "spline_uniform", "spline_nonce", "map_ctrl",
                            "star_search_ckb", "shiftc_ckb"]:
                        st.session_state.pop(k, None)

                    st.rerun()
                    

    with col_right:
        st.subheader("")
        st.space('large')
        st.space('large')
        
        if st.session_state.star_search_ckb:
            star_search_panel(map_plot)
            st.divider()
            

        if st.session_state.show_shiftc:
            shift_center_panel(map_plot)
            st.divider()


        if st.session_state.show_header:
            st.subheader('Header')
            header = st.session_state.current_map.meta
            st.json(header)

@st.fragment
def star_search_panel(map_plot):
    st.header('Star Search')
    with st.form("star_search_form"):
        col1, col2 = st.columns(2)
        with col1:
            catalog_display = st.selectbox(
                'Star Catalog:', 
                options=['Simbad', 'Gaia'],
                index=0
            )
            starcatalog = catalog_display.lower()                
        with col2:
            fmagn = st.number_input('Max Magnitude:', value=8.0)

        with st.container(horizontal=True):
            st.space('stretch')
            submitted = st.form_submit_button("Search Stars")
            st.space('stretch')
            
            if submitted:
                stars = search_stars(map_plot, catalog_name=starcatalog, fmagn=fmagn)
                st.session_state.stars = stars
                #st.rerun(scope="fragment")
                st.rerun()
    if st.session_state.stars is not None:
        if not st.session_state.stars.empty:
            st.write(f"Stars found: {len(st.session_state.stars)}")
            st.dataframe(st.session_state.stars)
        else:
            st.warning("No stars found with the selected criteria.")

@st.fragment
def shift_center_panel(map_plot):
    sunx, suny = sun_center(map_plot)
    st.header('Sun Center')

    with st.form("shift_center_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_sunx = st.number_input('X [px]', value=sunx)
        with col2:
            new_suny = st.number_input('Y [px]', value=suny)

        with st.container(horizontal=True):
            st.space('stretch')
            submitted = st.form_submit_button("Submit")
            st.space('stretch')
            
            if submitted:
                x_diff = new_sunx - sunx
                y_diff = new_suny - suny
                st.session_state.xy_diff = [x_diff, y_diff]
                st.rerun()


def update_right_panel_visibility():
    header_active = "Header" in st.session_state.map_ctrl
    shift_active = st.session_state.get('show_shiftc', False)
    star_active = st.session_state.get('star_search_ckb', False)
    
    # Risultato finale: basta che uno sia True
    st.session_state.show_right_panel = any([
        header_active, 
        shift_active, 
        star_active, 
    ])

def toggle_header():
    show = st.session_state.show_header_ckb
    st.session_state.show_header = show
    if not(st.session_state.show_shiftc):
        st.session_state.show_right_panel = show

def toggle_NE():
    st.session_state.show_ne = not st.session_state.show_ne

def toggle_rprof():
    st.session_state.show_rprof = not st.session_state.show_rprof

def toggle_pprof():
    st.session_state.show_pprof = not st.session_state.show_pprof

def toggle_intplot():
    st.session_state.intplot = not st.session_state.intplot

def toggle_rotate():
    st.session_state.rotate_map = not st.session_state.rotate_map

def toggle_shiftc():
    st.session_state.show_shiftc = not st.session_state.show_shiftc
    update_right_panel_visibility()

def toggle_star_src():
    update_right_panel_visibility()

def toggle_grid():
    st.session_state.show_grid = not st.session_state.show_grid

def toggle_plot_options():
    selected = st.session_state.map_ctrl
    
    st.session_state.intplot = "Interactive" in selected
    st.session_state.canvas_mode = "Canvas" in selected
    if st.session_state.intplot and st.session_state.canvas_mode:
        st.session_state.intplot = False
    st.session_state.rotate_map = "Rotate" in selected
    st.session_state.show_grid = "Grid" in selected
    st.session_state.show_header = "Header" in selected
    st.session_state.show_rprof = "Radial Profiles" in selected
    st.session_state.show_pprof = "Polar Profiles" in selected
    st.session_state.show_ne = "N-E Labels" in selected
    # per il right panel
    update_right_panel_visibility()


def show_webloader():
    st.subheader('Upload from web')
    st.write('')

    level = st.radio("Level", ["L1", "L2", "L3"], horizontal=True, key=f'level_{tabname}', index=2)
    
    with st.form("webloader_form"):

        if level != 'L3':
            col, col1, col2, col3 = st.columns([1,1,1,1])
            with col:
                st.write('Filter')
            with col1:
                flt_wb = st.checkbox('Wideband', key=f'wb_{tabname}')
                flt_p2 = st.checkbox('Polarizer 0°', key=f'p2_{tabname}')
            with col2:
                flt_fe = st.checkbox('Fe XIV', key=f'fe_{tabname}')
                flt_p1 = st.checkbox('Polarizer 60°', key=f'p1_{tabname}')
            with col3:
                flt_he = st.checkbox('He I D3', key=f'he_{tabname}')
                flt_p3 = st.checkbox('Polarizer 120°', key=f'p3_{tabname}')

            checkboxes = {
                "Wide": flt_wb,
                "Fe": flt_fe,
                "He": flt_he,
                "Polarizer%2060": flt_p1,
                "Polarizer%200": flt_p2,
                "Polarizer%20120": flt_p3
            }
        else:
            col, col1, col2, col3 = st.columns([1,1,1,1])
            with col:
                st.write('Type')
            with col1:
                flt_bt = st.checkbox('Total brightness', key=f'bt_{tabname}')
                flt_gr = st.checkbox('Green line', key=f'p2_{tabname}')
            with col2:
                flt_pb = st.checkbox('Polarisation brightness', key=f'pb_{tabname}')
                flt_he = st.checkbox('He I D3 line', key=f'p1_{tabname}')
            with col3:
                flt_pa = st.checkbox('Polarisation angle', key=f'he_{tabname}')

            checkboxes = {
                "Total": flt_bt,
                "Polarisation brightness": flt_pb,
                "Polarisation angle": flt_pa,
                "Green": flt_gr,
                "He": flt_he,
            }

        orbit_id = st.text_input("Orbit ID", "", key=f'oid_{tabname}', help="Check the orbit programs on the [ASPIICS website](https://www.sidc.be/proba-3/commanding/)")
        cycle_id = st.text_input("Cycle ID (8 digits)", "", key=f'cid_{tabname}')

        col1, col2 = st.columns([1,1])
        with col1:
            date_start = st.date_input("Start date", value=datetime.date(2025, 5, 1), format="DD.MM.YYYY", key=f'date_start_{tabname}')
            date_end = st.date_input("End date", value=datetime.date(2026, 9, 1), format="DD.MM.YYYY", key=f'date_end_{tabname}')
        with col2:
            time_start = st.time_input("Time", datetime.time(0,0), step=600, key=f'time_start_{tabname}')
            time_end = st.time_input("Time", datetime.time(23,59), step=600, key=f'time_end_{tabname}')

        limit = st.text_input("Limit query results:", "10", key=f'limit_{tabname}')

        submitted = st.form_submit_button("Show files")
        if submitted:
            flt_list = [key for key, checked in checkboxes.items() if checked]
            if not flt_list:
                flt_list = list(checkboxes.keys())
            dt_start = datetime.datetime.combine(date_start, time_start)
            dt_end = datetime.datetime.combine(date_end, time_end)
            files_url = aspiics_files_api(flt_list, level, orbit_id, cycle_id, dt_start, dt_end, limit)
            st.session_state.files_url = files_url

    if st.session_state.files_url:
        st.write('Files found:')
        selected_files = []
        for f in st.session_state.files_url:
            filename = f.split('/')[-1]
            if st.checkbox(filename):
                selected_files.append(f)

        st.write("Selected:", selected_files)

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button('Upload map', key=f'upload_map_{tabname}'):
                if len(selected_files) == 1:
                    upl_map = load_map_from_url(selected_files[0])
                    st.session_state.original_map = sunpy.map.Map(upl_map.data.astype('float'), upl_map.meta)
                    st.success(f"Loaded web file")
                    st.rerun()
                else:
                    st.error('Select only one file')
        with col2:
            download_web_files_btn(selected_files, label='Download selected files')

    st.divider()
    st.subheader('Upload using filename')
    st.write('')

    upl_filename = st.text_input("Filename:", "", key=f'ulfn_{tabname}', help="Must be the exact 'filename.fits'")

    if st.button('Upload map', key=f'upload_map2_{tabname}'):
        baseDataURL = "https://p3sc.oma.be/datarepfiles"
        parts = upl_filename.split('_')
        fileLevel = parts[2].upper()
        if 'v03' in upl_filename:
            file_version = 'v03'
        else:
            file_version = 'v2'
        file_url = f"{baseDataURL}/{fileLevel}/{file_version}/{upl_filename}"
        st.write(f'Link: {file_url}')
        upl_map = load_map_from_url(file_url)
        st.session_state.original_map = sunpy.map.Map(upl_map.data.astype('float'), upl_map.meta)
        st.success(f"Loaded web file")
        st.rerun()
        
################ not used ################

def show_sidebar():
    st.markdown("""
    <style>
    .stSidebar div.stButton > button {
        all: unset;                 /* rimuove tutto lo stile di default */
        color: inherit;             /* usa il colore del testo normale */
        font: inherit;              /* usa font e dimensione di st.write() */
        cursor: pointer;            /* mostra il cursore a mano quando ci passi sopra */
        text-align: left;           /* allinea il testo a sinistra */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("")

    with st.sidebar.expander("File"):
        if "show_open" not in st.session_state:
            st.session_state.show_open = False

        if st.button("Open", key="open"):
            st.session_state.show_open = not st.session_state.show_open

        if st.session_state.show_open:
            uploaded_file = st.file_uploader("", type=["fits", "fts"], key="fits_uploader")
            if uploaded_file is not None:
                try:
                    solar_map = file_to_smap(uploaded_file)
                    st.session_state.current_map = solar_map
                    st.success(f"File caricato: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Errore caricamento file: {e}")

        if st.button("Load from web", key="load"):
            #solar_map = sunpy.map.Map('https://p3sc.oma.be/datarepfiles/L3/v2/aspiics_l3_pb_1540C0280001_20250909T134312.fits')
            #st.session_state.current_map = solar_map
            #st.success(f"Loaded web file")
            #st.rerun()
            st.write('do it from the main tab...')
        

    with st.sidebar.expander('Image'):
        if st.button("Show header", key="show_header_btn"):
            st.session_state.show_header = not st.session_state.show_header
            st.session_state.show_right_panel = not st.session_state.show_right_panel
            print('show header')
        if st.button("Show N-E axis", key="show_NE"):
            st.session_state.show_ne = not st.session_state.show_ne
        #if st.button("Rotate image", key="rotate"):
        #    st.write('work in progress...')

    #with st.sidebar.expander('Process'):
    #    if st.button("Remove dark+bias", key="rem_darkbias"):
    #        st.write('work in progress...')
    #    if st.button("Flat Field correction", key="FF_corr"):
    #        st.write('work in progress...')
