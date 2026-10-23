import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev
from skimage.measure import profile_line

from logic.plot import aspiics_cmap_new


def _canvas_line_endpoints(obj, sx, sy):
    cx, cy = obj["left"], obj["top"]
    x1 = (cx + obj["x1"]) * sx
    y1 = (cy + obj["y1"]) * sy
    x2 = (cx + obj["x2"]) * sx
    y2 = (cy + obj["y2"]) * sy
    return x1, y1, x2, y2


def _canvas_rect_bounds(obj, sx, sy, data_shape):
    w = obj["width"] * obj.get("scaleX", 1)
    h = obj["height"] * obj.get("scaleY", 1)
    x_start, y_start = obj["left"], obj["top"]
    x_end, y_end = x_start + w, y_start + h
    x_a = int(np.clip(min(x_start, x_end) * sx, 0, data_shape[1]))
    x_b = int(np.clip(max(x_start, x_end) * sx, 0, data_shape[1]))
    y_a = int(np.clip(min(y_start, y_end) * sy, 0, data_shape[0]))
    y_b = int(np.clip(max(y_start, y_end) * sy, 0, data_shape[0]))
    return x_a, x_b, y_a, y_b


def _canvas_spline_points(objects, sx, sy):
    pts = []
    for o in objects:
        if o.get("type") == "circle":
            pts.append((o["left"] * sx, o["top"] * sy))
    return np.array(pts) if pts else np.empty((0, 2))


def _sample_spline(points, n_samples, smooth, k):
    x, y = points[:, 0], points[:, 1]
    tck, _ = splprep([x, y], s=smooth, k=k)
    u_dense = np.linspace(0, 1, max(n_samples * 20, 1000))
    x_dense, y_dense = splev(u_dense, tck)
    x_dense = np.asarray(x_dense)
    y_dense = np.asarray(y_dense)
    seg = np.hypot(np.diff(x_dense), np.diff(y_dense))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0, arc[-1], n_samples)
    u_new = np.interp(targets, arc, u_dense)
    xs, ys = splev(u_new, tck)
    return np.asarray(xs), np.asarray(ys), targets

def _make_canvas_background(data, vmin, vmax, cmap="gray", max_canvas=700):
    # Normalizza in [0, 1] e gestisci i NaN
    disp = np.nan_to_num(data, nan=vmin)
    disp = np.clip(disp, vmin, vmax)
    if vmax == vmin:
        norm = np.zeros_like(disp)
    else:
        norm = (disp - vmin) / (vmax - vmin)

    cmap_obj = cm.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgba = cmap_obj(norm)                      # shape (H, W, 4)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).convert("RGB")

    scale = min(1.0, max_canvas / max(img.width, img.height))
    canvas_w, canvas_h = int(img.width * scale), int(img.height * scale)
    canvas_bg = img.resize((canvas_w, canvas_h))
    sx = data.shape[1] / canvas_w
    sy = data.shape[0] / canvas_h
    return canvas_bg, canvas_w, canvas_h, sx, sy

@st.fragment
def draw_extract_panel(map_plot, vmin, vmax, results_container):
    results_container.empty()
    
    #data = map_plot.data
    # we need to flip, canvas is upside down respect to fits/sunpy
    data = np.flipud(map_plot.data)
    cmap = aspiics_cmap_new(map_plot)
    canvas_bg, canvas_w, canvas_h, sx, sy = _make_canvas_background(data, vmin, vmax, cmap=cmap)
    st.write("**######################################### BETA #########################################**")
    shape = st.radio(
        "Shape", ["Line", "Rectangle", "Spline"],
        horizontal=True, key="draw_shape",
    )
    if shape == "Line":
        analysis_options = ["Intensity profile", "Perpendicular-averaged profile"]
    elif shape == "Rectangle":
        analysis_options = ["Statistics + histogram"]
    else:
        analysis_options = ["Intensity profile along spline"]

    analysis = st.radio(
        "Analysis", analysis_options, horizontal=True, key="draw_analysis",
    )

    # Parametri per tipo
    if shape == "Line" and analysis == "Perpendicular-averaged profile":
        st.session_state.setdefault("perp_width", 5)
        st.slider("Perpendicular width (px)", 1, 21, step=2, key="perp_width")

    if shape == "Rectangle":
        st.session_state.setdefault("hist_bins", 50)
        st.slider("Histogram bins", 10, 200, key="hist_bins")

    if shape == "Spline":
        st.session_state.setdefault("spline_k", 3)
        st.session_state.setdefault("spline_s", 0.0)
        st.session_state.setdefault("spline_samples", 500)
        st.session_state.setdefault("spline_uniform", False)
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Spline degree", 1, 4, key="spline_k")
        with c2:
            st.number_input("Smoothing s", value=0.0, format="%.3f", key="spline_s")
        st.slider("Samples along spline", 50, 2000, key="spline_samples")
        st.checkbox("Use arc-length uniform sampling", key="spline_uniform")

    # Canvas key (nonce per reset spline)
    if shape == "Spline":
        if st.button("Clear points", key="clear_spline"):
            st.session_state["spline_nonce"] = st.session_state.get("spline_nonce", 0) + 1
        canvas_key = f"canvas_point_{st.session_state.get('spline_nonce', 0)}"
    else:
        canvas_key = f"canvas_{shape.lower()}"

    drawing_mode = {"Line": "line", "Rectangle": "rect", "Spline": "point"}[shape]

    canvas_result = st_canvas(
        background_image=canvas_bg,
        drawing_mode=drawing_mode,
        stroke_width=2,
        stroke_color="#FF0000",
        fill_color="rgba(255,0,0,0.7)" if drawing_mode == "point" else "rgba(255,0,0,0.15)",
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        key=canvas_key,
    )

    objects = (canvas_result.json_data or {}).get("objects", [])
    if not objects:
        st.info(f"Draw a {shape.lower()} on the image.")
        return

    last_obj = objects[-1]

    # ---------------- LINE ----------------
    if shape == "Line" and last_obj.get("type") == "line":
        x1, y1, x2, y2 = _canvas_line_endpoints(last_obj, sx, sy)
        length = float(np.hypot(x2 - x1, y2 - y1))
        n_pts = max(int(np.ceil(length)) + 1, 2)

        if analysis == "Intensity profile":
            t = np.linspace(0, 1, n_pts)
            xs = x1 + (x2 - x1) * t
            ys = y1 + (y2 - y1) * t
            profile = map_coordinates(data, np.array([ys, xs]), order=1, mode="constant", cval=np.nan)
            distances = np.hypot(xs - x1, ys - y1)
            ylabel, title = "Intensity", "Line Profile"
        else:
            linewidth = st.session_state.perp_width
            profile = profile_line(
                data, src=(y1, x1), dst=(y2, x2),
                linewidth=linewidth, order=1, mode="constant", cval=np.nan,
            )
            distances = np.linspace(0, length, len(profile))
            ylabel = f"Mean intensity (width {linewidth} px)"
            title = "Perpendicular-averaged Profile"

        with results_container:
            with st.container():
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(distances, profile, color="crimson", lw=1.2)
                ax.set_xlabel("Distance along line (pixels)")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(alpha=0.3)
            
                st.subheader("Line Profile")
                st.pyplot(fig, width='stretch')
                st.caption(
                    f"({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}) · "
                    f"{len(profile)} samples · length ≈ {length:.1f} px"
                )
                st.metric("Mean", f"{np.nanmean(profile):.4g}")
                st.metric("Max", f"{np.nanmax(profile):.4g}")
        csv = "distance_px,intensity\n" + "\n".join(
            f"{d:.6f},{v:.6f}" for d, v in zip(distances, profile)
        )
        st.download_button("Download profile (CSV)", csv, "line_profile.csv", "text/csv")

    # ---------------- RECTANGLE ----------------
    elif shape == "Rectangle" and last_obj.get("type") == "rect":
        x_a, x_b, y_a, y_b = _canvas_rect_bounds(last_obj, sx, sy, data.shape)
        if x_b <= x_a or y_b <= y_a:
            st.warning("Selection is empty.")
            return

        vals = data[y_a:y_b, x_a:x_b].ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            st.warning("No finite values in the selected region.")
            return
        with results_container:
            with st.container():
                st.caption(
                    f"ROI x ∈ [{x_a}, {x_b}], y ∈ [{y_a}, {y_b}] · "
                    f"{x_b - x_a} × {y_b - y_a} px · {vals.size} valid px"
                )

                c1, c2 = st.columns(2)
                c1.metric("Mean", f"{vals.mean():.4g}")
                c1.metric("Median", f"{np.median(vals):.4g}")
                c2.metric("Std", f"{vals.std():.4g}")
                c2.metric("Min/Max", f"{vals.min():.3g} / {vals.max():.3g}")
                st.metric("Sum", f"{vals.sum():.4g}")

                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(vals, bins=st.session_state.hist_bins, color="steelblue",
                        edgecolor="black", alpha=0.85)
                ax.set_xlabel("Intensity")
                ax.set_ylabel("Count")
                ax.set_title("Histogram of selection")
                ax.grid(alpha=0.3)
                st.pyplot(fig, width='stretch')

        csv_stats = (
            "stat,value\n"
            f"mean,{vals.mean():.6f}\n"
            f"median,{np.median(vals):.6f}\n"
            f"std,{vals.std():.6f}\n"
            f"min,{vals.min():.6f}\n"
            f"max,{vals.max():.6f}\n"
            f"sum,{vals.sum():.6f}\n"
            f"npix,{vals.size}\n"
        )
        st.download_button("Download ROI stats (CSV)", csv_stats, "roi_stats.csv", "text/csv")

    # ---------------- SPLINE ----------------
    elif shape == "Spline":
        pts = _canvas_spline_points(objects, sx, sy)
        if len(pts) < 2:
            st.info(f"Place at least 2 points (you have {len(pts)}).")
            return

        k_eff = min(st.session_state.spline_k, len(pts) - 1)
        try:
            if st.session_state.spline_uniform:
                xs, ys, dist = _sample_spline(
                    pts, st.session_state.spline_samples,
                    st.session_state.spline_s, k_eff,
                )
            else:
                x, y = pts[:, 0], pts[:, 1]
                tck, _ = splprep([x, y], s=st.session_state.spline_s, k=k_eff)
                u_new = np.linspace(0, 1, st.session_state.spline_samples)
                xs, ys = splev(u_new, tck)
                xs, ys = np.asarray(xs), np.asarray(ys)
                seg = np.hypot(np.diff(xs), np.diff(ys))
                dist = np.concatenate([[0.0], np.cumsum(seg)])
        except Exception as e:
            st.error(f"Spline fit failed: {e}")
            return

        xs = np.clip(xs, 0, data.shape[1] - 1)
        ys = np.clip(ys, 0, data.shape[0] - 1)
        profile = map_coordinates(data, np.array([ys, xs]), order=1, mode="constant", cval=np.nan)
            
        with results_container:
            with st.container():
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(dist, profile, color="darkviolet", lw=1.3)
                ax.set_xlabel("Arc length along spline (pixels)")
                ax.set_ylabel("Intensity")
                ax.set_title("Intensity profile along spline")
                ax.grid(alpha=0.3)
                st.pyplot(fig, width='stretch')

                st.caption(
                    f"{len(pts)} control points · degree {k_eff} · "
                    f"s={st.session_state.spline_s} · "
                    f"{st.session_state.spline_samples} samples · length ≈ {dist[-1]:.1f} px"
                )
                st.metric("Mean", f"{np.nanmean(profile):.4g}")
                st.metric("Max", f"{np.nanmax(profile):.4g}")

        csv = "arc_length_px,intensity\n" + "\n".join(
            f"{d:.6f},{v:.6f}" for d, v in zip(dist, profile)
        )
        st.download_button("Download spline profile (CSV)", csv, "spline_profile.csv", "text/csv")

    else:
        st.info(f"Draw a {shape.lower()} on the image to see the result.")