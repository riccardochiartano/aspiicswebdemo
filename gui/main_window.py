import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import tempfile
import plotly.express as px
import datetime
from streamlit_plotly_events import plotly_events

from logic.process import wow_filter, log_scale, nrgf_filter, mgn_filter, unsharp_mask_filter
from logic.plot import plot_profile, aspiics_cmap, aspiics_cmap_new, plot_NE_labels, plot_pprof, plot_rprof
from logic.utils import sun_center, download_map_btn, file_to_smap, aspiics_files_url, download_web_files_btn, header_from_sunpymap
from gui.profile_window import profile_window
import logic.rob.aspiics_misc as am


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
        st.session_state.show_ne = False
    if "show_files" not in st.session_state:
        st.session_state.show_files = False
    if "show_rprof" not in st.session_state:
        st.session_state.show_rprof = False
    if "show_pprof" not in st.session_state:
        st.session_state.show_pprof = False
    if "show_shiftc" not in st.session_state:
        st.session_state.show_shiftc = False
    if "files_url" not in st.session_state:
        st.session_state.files_url = None
    if "intplot" not in st.session_state:
        st.session_state.intplot = False
    if "rotate_map" not in st.session_state:
        st.session_state.rotate_map = False
    if "xy_diff" not in st.session_state:
        st.session_state.xy_diff = None
 
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
                    solar_map = file_to_smap(uploaded_file)
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
            if st.session_state.current_map is not None:
                map_plot = sunpy.map.Map(st.session_state.current_map.data, st.session_state.current_map.meta)
                image_plot = map_plot.data

            col1, col2 = st.columns([3,1])
            
            with col1:
                scale_option = st.radio("Image filters:",
                    ["Linear", 
                     "Log",
                      #"MGN", 
                      #"WOW filter", 
                      #"Unsharp masking", 
                      #"NRGF filter (Slow)",
                      ],
                    horizontal=True,
                    label_visibility='collapsed'  
                )
            with col2:
                st.space(21)
                shift_c = st.checkbox('Shift solar center', key='shiftc_ckb', on_change=toggle_shiftc)

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                interactive_plot = st.checkbox('Interactive plot', key='chk_intplot', on_change=toggle_intplot)
                rotate_map = st.checkbox('Rotate map', key='rotate_map_ckb', on_change=toggle_rotate)
            with col2:
                show_header = st.checkbox('Show header', key='show_header_ckb', on_change=toggle_header)
                show_radprofiles = st.checkbox('Show radial profiles', key='sh_rprof_ckb', on_change=toggle_rprof)
            with col3:
                show_NE_labels = st.checkbox('Show N-E labels', key='show_NE_ckb', on_change=toggle_NE)
                show_polprofiles = st.checkbox('Show polar profiles', key='sh_pprof_ckb', on_change=toggle_pprof)

            # map container
            map_container = st.container()

            if scale_option == "Log":
                map_plot = log_scale(map_plot) 
            if scale_option == "MGN":
                map_plot = mgn_filter(map_plot) 
            if scale_option == "WOW filter":
                map_plot = wow_filter(map_plot)
            if scale_option == "Unsharp masking":
                map_plot = unsharp_mask_filter(map_plot)
            if scale_option == "NRGF filter":
                map_plot = nrgf_filter(map_plot)
            image_plot = map_plot.data   
            #map_plot = sunpy.map.Map(image_plot, st.session_state.current_map.meta)

            # map histogram
            flat_data = image_plot[np.isfinite(image_plot)].flatten()
            if flat_data.size == 0:
                st.info("No valid data")
                return

            p_low, p_high = np.percentile(flat_data, [0.1, 99.9])
            #flat_data_clip = flat_data[(flat_data >= p_low) & (flat_data <= p_high)]
            flat_data_clip = flat_data

            col1, col2, col3 = st.columns([1,7,1])
            with col2:
                fig_hist, ax_hist = plt.subplots(figsize=(6,1))
                ax_hist.hist(flat_data, bins=256, color='blue', alpha=0.7)
                ax_hist.set_yticks([])
                ax_hist.set_ylabel("") 
                st.pyplot(fig_hist)

            # vmin/max map
            vmin_default, vmax_default = np.percentile(flat_data_clip, [1,99.9])
            step = (vmax_default - vmin_default) / 1000 if vmax_default != vmin_default else 1e-8
            vmin, vmax = st.slider(
                "Select vmin/vmax",
                min_value=float(np.nanmin(flat_data_clip)),
                max_value=float(np.nanmax(flat_data_clip)),
                value=(float(vmin_default), float(vmax_default)),
                step=step,
                format="%.2e"
            )

            # plot map
            with map_container:
                if not(st.session_state.intplot):
                    fig_map, ax_map = plt.subplots(figsize=(6,6), subplot_kw={'projection': st.session_state.current_map.wcs})
                    im = map_plot.plot(axes=ax_map, clim=(vmin, vmax), cmap=aspiics_cmap_new(map_plot))
                    #im = ax_map.imshow(map_plot.data,origin='lower',vmin=vmin,vmax=vmax,cmap=aspiics_cmap_new(map_plot))
                    ax_map.grid(True, ls='--', lw=0.4) 
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
                    cbar = plt.colorbar(im, ax=ax_map, label=f"Intensity [{map_plot.meta.get('bunit', '')}]")
                    st.pyplot(fig_map, width='content')
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
                    st.rerun()
                
                    

    with col_right:
        st.subheader("")

        if st.session_state.show_shiftc:
            sunx, suny = sun_center(map_plot)
            st.space('large')
            st.space('large')
            st.header('Sun center')
            #with st.container(horizontal=True):
            #    new_sunx = st.number_input('X [px]', value=sunx)
            #    st.space('stretch')
            #    new_suny = st.number_input('Y [px]', value=suny)
            #st.space('medium')
            #with st.container(horizontal=True):
            #    st.space('stretch')
            #    if st.button('Submit'):
            #        x_diff = new_sunx - sunx
            #        y_diff = new_suny - suny
            #        st.session_state.current_map.meta['crpix1'] += x_diff 
            #        st.session_state.current_map.meta['crpix2'] += y_diff 
            #        st.rerun()
            #    st.space('stretch')

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
                        #st.session_state.current_map.meta['crpix1'] += x_diff 
                        #st.session_state.current_map.meta['crpix2'] += y_diff 
                        st.rerun()


        if st.session_state.show_header:
            st.subheader('Header')
            header = st.session_state.current_map.meta
            st.json(header)

        
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
    if not(st.session_state.show_header):
        st.session_state.show_right_panel = not st.session_state.show_right_panel

def show_webloader():
    st.subheader('Upload from web')
    st.write('')
    
    level = st.radio("Level", ["L1", "L2", "L3"], horizontal=True, key=f'level_{tabname}')

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
            "wb": flt_wb,
            "fe": flt_fe,
            "he": flt_he,
            "p1": flt_p1,
            "p2": flt_p2,
            "p3": flt_p3
        }
    else:
        col, col1, col2, col3 = st.columns([1,1,1,1])
        with col:
            st.write('Type')
        with col1:
            flt_bt = st.checkbox('Total brightness', key=f'bt_{tabname}')    
        #    flt_p2 = st.checkbox('Polarizer 0°', key=f'p2_{tabname}')    
        with col2:
            flt_pb = st.checkbox('Polarized brightness', key=f'pb_{tabname}')
        #    flt_p1 = st.checkbox('Polarizer 60°', key=f'p1_{tabname}')    
        #with col3:
        #    flt_he = st.checkbox('He I D3', key=f'he_{tabname}')    
        #    flt_p3 = st.checkbox('Polarizer 120°', key=f'p3_{tabname}')    

        checkboxes = {
            "bt": flt_bt,
            "pb": flt_pb,
            #"he": flt_he,
            #"p1": flt_p1,
            #"p2": flt_p2,
            #"p3": flt_p3
        }

    flt_list = [key for key, checked in checkboxes.items() if checked]


    cycle_id = st.text_input("Cycle ID (8 digits)", "1560607E", key=f'cid_{tabname}')
    #seq_num = st.text_input("Sequence number", "")
    #acq_num = st.text_input("Acquisition number", "")
    #exp_num = st.text_input("Exposure number", "")

    col1, col2 = st.columns([1,1])
    with col1:
        date_start = st.date_input("Start date", value=datetime.date(2025, 5, 1), format="DD.MM.YYYY", key=f'date_start_{tabname}')
        date_end = st.date_input("End date", value=datetime.date(2026, 9, 1), format="DD.MM.YYYY", key=f'date_end_{tabname}')
        #date_start = st.date_input("Start date", value=None, format="DD.MM.YYYY")
        #date_end = st.date_input("End date", value=None, format="DD.MM.YYYY")
    with col2:    
        #time_start = st.time_input("Time", value=None, key='time_start', step=60)
        #time_end = st.time_input("Time", value=None, key='time_end', step=60)
        time_start = st.time_input("Time", datetime.time(0,0), step=600, key=f'time_start_{tabname}')
        time_end = st.time_input("Time", datetime.time(23,59), step=600, key=f'time_end_{tabname}')
    
    if st.button("Show files", key=f'shw_files_{tabname}'):
    #    st.session_state.show_files = not st.session_state.show_files
    #
    #if st.session_state.show_files:
        dt_start = datetime.datetime.combine(date_start, time_start)
        dt_end = datetime.datetime.combine(date_end, time_end)
            
        files_url = aspiics_files_url(flt_list, level, cycle_id, dt_start, dt_end)
        if files_url == []:
            st.error('No files with these keys...')
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
                    #st.session_state.current_map = sunpy.map.Map(selected_files[0])
                    upl_map = sunpy.map.Map(selected_files[0])
                    st.session_state.original_map = sunpy.map.Map(upl_map.data.astype('float'), upl_map.meta)
                    st.success(f"Loaded web file")
                    st.rerun()
                else:
                    st.error('Select only one file')
        with col2:
            download_web_files_btn(selected_files, label='Download selected files')
                    