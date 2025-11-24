import streamlit as st
from gui.main_window import SolarMapViewer, show_sidebar
from gui.profile_window import profile_window
from gui.demod_window import demodulation_window
from gui.deviation_window import deviation_window
from gui.continuum_sub_window import continuum_sub_window
from gui.merge_window import merge_window
from gui.e_density_window import e_density_window
from gui.calib_window import calib_window
from gui.movie_window import movie_window
from gui.comparison_window import comparison_window

st.set_page_config(
    page_title="ASPIICS Web",
    page_icon="🌞",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stSlider"] {
        width: 600px;  
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("ASPIICS web")

    # sidebar
    #show_sidebar()

    tab_main, tab_calib, tab_merge, tab_demod, tab_edens, tab_subc, tab_movies, tab_comparison = st.tabs([
        "Main",
        #"Plot profile",
        "Calibration",
        "Merging",
        "Demodulation",
        "Electron Density",
        "Continuum Subtraction",
        "Movies",
        "Comparison"
        #"Deviation from RF"
    ])

    #for key, val in sorted(st.session_state.items()):
    #    st.write(f"{key} → {val}")

    with tab_demod:
        demodulation_window()

    # main window
    with tab_main:
            SolarMapViewer() 
        #tab_main_map, tab_prof = st.tabs(['Main', 'Profiles'])
        #with tab_main_map:
        #with tab_prof:
        #    profile_window()

    with tab_calib:
        calib_window()

    with tab_merge:
        merge_window()

    with tab_edens:
        e_density_window()

    with tab_subc:
        continuum_sub_window()

    with tab_movies:
        movie_window()

    with tab_comparison:
        comparison_window()
    
    #with tab9:
    #    deviation_window()

 

if __name__ == "__main__":
    main()
