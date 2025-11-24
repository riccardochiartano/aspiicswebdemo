import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import tempfile
import os
import pandas as pd
from scipy.interpolate import interp1d, griddata
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.special import gamma
from matplotlib import colors
import plotly.express as px
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio


from logic.utils import sun_center


base_dir = Path(__file__).resolve().parent.parent

tabname = 'movie'

def movie_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Movies")

        uploaded_movie_files = st.file_uploader("Upload files for the movie creation:", 
                                       type=["fits", "fts"], 
                                       accept_multiple_files=True)
        
        if uploaded_movie_files:
            paths = []
            names = []
            for uf in uploaded_movie_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                    tmp.write(uf.read())
                    paths.append(tmp.name)
                names.append(uf.name)

            st.session_state[f"movie_files_path"] = paths   # real path
            st.session_state[f"movie_files_name"] = names   # real name

        movie_type = st.radio("Movie type:",
            ["Normal", "Base difference", "Running difference"],
            horizontal=True,
            key="movie_type"
        )

        rotation = st.checkbox('Rotate with north upwards')
        
        if st.button('Create movie'):

            if st.session_state[f"movie_files_path"]:
                create_movie(st.session_state[f"movie_files_path"], movie_type, rotation)            
            else:
                st.error('Load files first.')


def create_movie(filenames, movie_type, rotation):
    m_seq = sunpy.map.Map(filenames, sequence=True)
    m_seq = sunpy.map.Map([sunpy.map.Map(m.data.astype(float), m.meta) for m in m_seq], sequence=True)
    if rotation:
        m_seq = sunpy.map.Map([m.rotate() for m in m_seq], sequence=True)
    
    #telesc = f'{m_seq[0].meta['instrume']}_{m_seq[0].meta['detector']}'

    if movie_type == 'Normal':
        m_seq_plot = m_seq
    elif movie_type == "Base difference":
        m_seq_plot = sunpy.map.Map([m - m_seq[0].quantity for m in m_seq[1:]], sequence=True)
    elif movie_type == "Running difference":
        m_seq_plot = sunpy.map.Map(
            [m - prev_m.quantity for m, prev_m in zip(m_seq[1:], m_seq[:-1])],
            sequence=True
        )

    ##### old with FuncAnimation #####
    #ani = create_animation_fast(m_seq_plot)
    #tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    #output_path = tmp.name
    #ani.save(output_path, writer='ffmpeg', fps=4)
    #st.video(output_path)

    frames = create_frames(m_seq_plot)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    output_path = tmp.name
    iio.imwrite(output_path, frames, fps=4, codec='vp9', quality=7)
    st.video(output_path)

    #imgs = [m.data for m in m_seq_plot]
    #imgs = np.array(imgs)
    #fig = px.imshow(imgs, animation_frame=0)
    #st.plotly_chart(fig)
    return

def create_frames(m_seq):
    rsun_px = int((m_seq[0].rsun_obs/m_seq[0].scale[0]).value)

    frames = []
    s_center = sun_center(m_seq[0])
    flat_data = m_seq[0].data[np.isfinite(m_seq[0].data)].flatten()
    vmin, vmax = np.percentile(flat_data, [0.01,99.997])
    max_abs = np.percentile(np.abs(flat_data), 99.99)
    vmin, vmax = -max_abs, max_abs
    st.write(vmin, vmax)
    for i, m in enumerate(m_seq):
        frame = map_to_rgb(m, vmin=vmin, vmax=vmax)
        frame = draw_limb(frame, center=s_center, r_sun=rsun_px, width=3, color="#ffffff")
        frame = draw_grid(frame, center=s_center, r_sun=rsun_px, width=1, color='#ffffff')
        title = create_title(m)                  
        frame = add_text_to_frame(frame, title)  
        frames.append(frame)

    return frames
    

def create_title(map):
    header = map.meta
    #dateobs = header['date-obs']
    #timeobs = header['time-obs']
    #title = f'{header['instrume']}-{header['detector']} {dateobs} {timeobs[:5]}'
    title = f'{map.observatory}-{map.instrument} {map.detector}     {str(map.date)[:-7]}'
    return title

def create_animation_fast(m_seq):
    h, w = m_seq[0].data.shape

    dpi = 300
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(projection=m_seq[0])

    m_seq[0].draw_grid(axes=ax)
    m_seq[0].draw_limb(axes=ax)

    for coord in [0, 1]:
        ax.coords[coord].set_ticklabel_visible(False)
        ax.coords[coord].set_ticks_visible(False)

    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    norm = colors.Normalize(vmin=-1500, vmax=2000)
    im = ax.imshow(m_seq[0].data, origin='lower', cmap='Greys_r', norm=norm)

    text = fig.text(0.03, 0.02, create_title(m_seq[0]), color='white')

    def update(i):
        im.set_data(m_seq[i].data)
        text.set_text(create_title(m_seq[i]))
        return im, text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(m_seq),
        interval=500,
        blit=False
    )

    return ani


def map_to_rgb(m, vmin=-1500, vmax=2000, add_text=None):
    # Ottieni i dati come array
    data = m.data.copy()
    
    # Normalizza e applica cmap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    smap = plt.colormaps[m.cmap.name]
    img_rgb = (smap(norm(data))[:, :, :3] * 255).astype(np.uint8)
    
    return img_rgb

def draw_limb(rgb_array, color=(255,255,255), center=(512,512), r_sun=(100), width=2):
    img = Image.fromarray(rgb_array)
    draw = ImageDraw.Draw(img)
    
    # bbox del cerchio
    bbox = [center[0]-r_sun, center[1]-r_sun, center[0]+r_sun, center[1]+r_sun]
    draw.ellipse(bbox, outline=color, width=width)
    
    return np.array(img)

def draw_grid(rgb_array, center=(512,512), r_sun=(100), n_lat=5, n_lon=8, color=(255,255,255), width=1):
    img = Image.fromarray(rgb_array)
    draw = ImageDraw.Draw(img)

    cx, cy = center

    # --- latitudini: ellissi concentriche
    for i in range(1, n_lat):
        f = i / n_lat
        #ry = r_sun * (1 - 0.99 * f)  
        #bbox = [cx - r_sun, cy - ry, cx + r_sun, cy + ry]
        rx = r_sun * (1 - 0.99 * f)  
        bbox = [cx - rx, cy - r_sun, cx + rx, cy + r_sun]
        draw.ellipse(bbox, outline=color, width=width)
    draw.line([(cx, cy-r_sun), (cx, cy+r_sun)], fill=color, width=width)

    # --- longitudini: linee radiali
    for i in range(n_lon):
        angle = np.pi * i / n_lon -np.pi/2
        x_start = cx - r_sun * np.cos(angle)
        y_start = cy + r_sun * np.sin(angle)
        x_end = cx + r_sun * np.cos(angle)
        y_end = cy + r_sun * np.sin(angle)
        draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=width)

    return np.array(img)

def add_text_to_frame(frame, text, fontsize=40):
    # frame: numpy array 1024x1024x3 uint8
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    
    # font di default (puoi cambiare se vuoi)
    font = ImageFont.load_default(size=fontsize)
    
    # posiziona in basso a sinistra, simile a fig.text(0.03,0.02)
    x = int(0.03 * frame.shape[1])
    y = int(0.98 * frame.shape[0]) - 30  # leggero offset dal bordo
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return np.array(pil_img)



