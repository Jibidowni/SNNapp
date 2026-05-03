import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import torch
from pathlib import Path
import snntorch as snn
import snntorch.spikegen as spikegen
import network_mlp as mlp
import tonic
import numpy as np
import time
from PIL import Image

st.set_page_config(
    page_title="LIF Neuron Explorer",
    page_icon="⚡",
    layout="wide",
)


# -----------------------------
# Navigation state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page_name: str):
    st.session_state.page = page_name


# -----------------------------
# Cached model loading
# -----------------------------
@st.cache_resource
def load_nmnist_mlp_model(weights_path: str, device_type: str = "auto"):
    """Load MLP model with caching. Called once and cached across reruns."""
    # Determine device
    if device_type == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    
    try:
        model = mlp.load_model(weights_path, device=device)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {weights_path}: {e}")
        return None

# Core model
# -----------------------------
# Reference implementation kept for clarity.
# def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
#     tau_mem = R * C
#     spk = (mem > threshold).float()
#     mem = mem + (time_step / tau_mem) * (-mem + cur * R) - spk * threshold
#     return mem, spk


def build_current(num_steps, amplitude, start_step=5):
    cur = torch.cat(
        (
            torch.zeros(start_step, 1),
            torch.ones(max(0, num_steps - start_step), 1) * amplitude,
        ),
        0,
    )
    return cur[:num_steps]


def build_spike_train(num_steps, mode, spike_amplitude, spike_frequency, spike_probability, dt, start_step=5):
    spk_in = torch.zeros(num_steps, 1)
    start_step = min(start_step, max(0, num_steps - 1))

    if mode == "Regular":
        if spike_frequency > 0:
            step_interval = max(1, int(round(1.0 / (spike_frequency * dt))))
            spk_in[start_step::step_interval] = 1.0

    elif mode == "Random":
        prob_tensor = torch.ones((num_steps, 1)) * spike_probability
        prob_tensor[:start_step] = 0.0
        spk_in = spikegen.rate_conv(prob_tensor).float()

    cur_in = spk_in * spike_amplitude
    return spk_in, cur_in


def simulate_lapicque(num_steps, cur_in, threshold, dt, R, C, reset_mechanism):
    lif = snn.Lapicque(
        beta=False,
        R=R,
        C=C,
        time_step=dt,
        threshold=threshold,
        reset_mechanism=reset_mechanism,
    )

    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []

    for step in range(num_steps):
        spk, mem = lif(cur_in[step], mem)
        mem_rec.append(mem.clone())
        spk_rec.append(spk.clone())

    mem_rec = torch.stack(mem_rec).squeeze(-1).squeeze(-1)
    spk_rec = torch.stack(spk_rec).squeeze(-1).squeeze(-1)
    return mem_rec, spk_rec


def build_trace_charts(cur_in, mem_rec, spk_rec, thr_line=1.0, spk_input=None, input_mode="Constant current"):
    cur_vals = torch.flatten(cur_in.detach().cpu()).tolist()
    mem_vals = torch.flatten(mem_rec.detach().cpu()).tolist()
    spk_vals = torch.flatten(spk_rec.detach().cpu()).tolist()
    steps = list(range(len(mem_vals)))

    if spk_input is not None and input_mode == "Spike train":
        input_vals = torch.flatten(spk_input.detach().cpu()).tolist()
        input_label = "Input spike"
        input_title = "Input spike train"
    else:
        input_vals = cur_vals
        input_label = "Current"
        input_title = "Input current"

    input_df = pd.DataFrame(
        {
            "Step": steps,
            input_label: input_vals,
        }
    )

    mem_df = pd.DataFrame(
        {
            "Step": steps,
            "Membrane potential": mem_vals,
            "Threshold": [thr_line] * len(steps),
        }
    )

    spk_df = pd.DataFrame(
        {
            "Step": steps,
            "Spike": spk_vals,
        }
    )

    base_axis = alt.Axis(labelColor="#D7E0EC", titleColor="#D7E0EC", gridColor="rgba(215,224,236,0.14)")
    x_encoding = alt.X("Step:Q", axis=base_axis, scale=alt.Scale(domain=[0, max(1, len(steps) - 1)]))

    if spk_input is not None and input_mode == "Spike train":
        input_chart = (
            alt.Chart(input_df)
            .mark_bar(width=2)
            .encode(
                x=x_encoding,
                y=alt.Y(f"{input_label}:Q", axis=base_axis, title=input_label),
                tooltip=["Step", input_label],
            )
            .properties(title=input_title, height=230)
        )
    else:
        input_chart = (
            alt.Chart(input_df)
            .mark_line(strokeWidth=2.5, interpolate="step-after")
            .encode(
                x=x_encoding,
                y=alt.Y(f"{input_label}:Q", axis=base_axis, title=input_label),
                tooltip=["Step", input_label],
            )
            .properties(title=input_title, height=230)
        )

    mem_long = mem_df.melt("Step", var_name="Trace", value_name="Value")
    mem_chart = (
        alt.Chart(mem_long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=x_encoding,
            y=alt.Y("Value:Q", axis=base_axis, title="Membrane"),
            color=alt.Color(
                "Trace:N",
                scale=alt.Scale(
                    domain=["Membrane potential", "Threshold"],
                    range=["#7DD3FC", "#FCA5A5"],
                ),
                legend=alt.Legend(labelColor="#D7E0EC", titleColor="#D7E0EC", orient="top"),
            ),
            tooltip=["Step", "Trace", "Value"],
        )
        .properties(title="Membrane trace", height=320)
    )

    spike_chart = (
        alt.Chart(spk_df)
        .mark_bar(width=2)
        .encode(
            x=x_encoding,
            y=alt.Y("Spike:Q", axis=base_axis, scale=alt.Scale(domain=[0, 1.05]), title="Spike"),
            tooltip=["Step", "Spike"],
        )
        .properties(title="Output spike train", height=230)
    )

    return input_chart, mem_chart, spike_chart


def build_neuron_charts(mem_trace, spk_trace, layer_name, neuron_index, threshold=1.0):
    steps = list(range(len(mem_trace)))
    mem_vals = torch.flatten(mem_trace.detach().cpu()).tolist()
    spk_vals = torch.flatten(spk_trace.detach().cpu()).tolist()
    
    trace_df = pd.DataFrame(
        {
            "Step": steps,
            "Membrane": mem_vals,
            "Spike": spk_vals,
            "Threshold": [threshold] * len(steps),
        }
    )

    base_axis = alt.Axis(labelColor="#D7E0EC", titleColor="#D7E0EC", gridColor="rgba(215,224,236,0.14)")
    x_encoding = alt.X("Step:Q", axis=base_axis, scale=alt.Scale(domain=[0, max(1, len(steps) - 1)]))

    mem_long = trace_df[["Step", "Membrane", "Threshold"]].melt("Step", var_name="Trace", value_name="Value")
    mem_chart = (
        alt.Chart(mem_long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=x_encoding,
            y=alt.Y("Value:Q", axis=base_axis, title="Membrane"),    #("Value:Q", axis=base_axis, title=f"{layer_name} neuron {neuron_index} membrane"),
            color=alt.Color(
                "Trace:N",
                scale=alt.Scale(
                    domain=["Membrane", "Threshold"],
                    range=["#7DD3FC", "#FCA5A5"],
                ),
                legend=alt.Legend(labelColor="#D7E0EC", titleColor="#D7E0EC", orient="top"),
            ),
            tooltip=["Step", "Trace", "Value"],
        )
        .properties(height=288, title=f"{layer_name} neuron {neuron_index} membrane")
    )

    spike_chart = (
        alt.Chart(trace_df)
        .mark_bar(width=2)
        .encode(
            x=x_encoding,
            y=alt.Y("Spike:Q", axis=base_axis, scale=alt.Scale(domain=[0, 1.05]), title="Spike"),
            tooltip=["Step", "Spike"],
        )
        .properties(height=222, title=f"{layer_name} neuron {neuron_index} spikes")
    )

    return mem_chart, spike_chart


def plot_dvs_frames(sample, max_frames=None):
    """Display DVS frames with interactive animation using tonic.
    
    sample: tensor with shape [time, batch, channels, h, w]
    Tonic's plot_animation handles polarity coloring automatically.
    """
    time_steps = sample.shape[0]
    if max_frames is not None:
        time_steps = min(time_steps, max_frames)
    
    # Extract frames: [time, batch, channels, h, w] -> [time, channels, h, w]
    frames = sample[:time_steps, 0].detach().cpu().numpy()  # [time, channels, h, w]
    
    # Create animation using tonic (handles polarity coloring automatically)
    try:
        # Set figure size via matplotlib params before calling tonic
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams["figure.dpi"] = 60
        
        ani = tonic.utils.plot_animation(frames)
        html_str = ani.to_jshtml()
        components.html(html_str, height=400, scrolling=False)
        
        # Reset params to default to not affect other plots
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        plt.rcParams["figure.dpi"] = plt.rcParamsDefault["figure.dpi"]
        
        #st.write(f"Animation: {time_steps} frames, {frames.shape[2]}×{frames.shape[3]} pixels")
    except Exception as e:
        st.error(f"Failed to create animation: {e}")
        st.info(f"Frames shape: {frames.shape}")

def upscale_nearest(img, scale=10):
    pil_img = Image.fromarray(img)
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    return pil_img.resize(new_size, Image.Resampling.NEAREST)

# -----------------------------
# Minimal styling
# -----------------------------
# The visual theme should come from `.streamlit/config.toml`.
# This CSS only handles the centered welcome screen and button shape.
st.markdown(
    """
<style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .welcome-spacer-top {
        height: 22vh;
    }

    .welcome-title {
        text-align: center;
        font-size: clamp(3rem, 7vw, 6.4rem);
        font-weight: 760;
        letter-spacing: -0.065em;
        line-height: 0.95;
        margin: 0 0 2.2rem 0;
    }

    div.stButton > button {
        height: 3.4rem;
        border-radius: 18px;
        font-size: 1rem;
        font-weight: 500;
        border: 1px solid rgba(226, 232, 240, 0.16);
        background: rgba(30, 41, 59, 0.72);
    }

    div.stButton > button:hover {
        border-color: rgba(56, 189, 248, 0.55);
        background: rgba(30, 41, 59, 0.95);
    }   

    .tiny-muted {
        opacity: 0.75;
        font-size: 0.92rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Home page
# -----------------------------
if st.session_state.page == "home":
    #st.markdown('<div class="welcome-spacer-top"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 36vh;"></div>', unsafe_allow_html=True)
    left_space, content, right_space = st.columns([1.0, 1.6, 1.0])

    with content:
        st.markdown(
            """
            <div style="
                text-align: center;
                font-size: 1.5rem;
                font-weight: 500;
                opacity: 0.72;
                margin-bottom: 1.1rem;
            ">
                Choose a module
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("LIF Neuron Simulator", use_container_width=True):
                go_to("lif")
                st.session_state.show_loading = True
                st.rerun()
        with col2:
            if st.button("Spiking Network Explorer", use_container_width=True):
                go_to("network")
                st.session_state.show_loading = True
                st.rerun()

    st.stop()


# -----------------------------
# Network placeholder page
# -----------------------------
if st.session_state.page == "network":
    if st.button("← Back"):
        go_to("home")
        st.rerun()

    st.header("Spiking Neuronal Network Explorer")
    st.caption("Explore how a trained SNN classifies event-based N-MNIST samples.")

    with st.expander("About DVS and N-MNIST"):

        st.markdown("""
        **Dynamic Vision Sensors (DVS)** record pixel-wise changes in brightness instead of full Images.  
        Each Event contains a pixel position, timestamp, and polarity.
        **N-MNIST** is an event-based version of MNIST. The digits are represented as temporal event streams rather than static images.
        In this Explorer, a trained Spiking Neural Network processes these event frames over time. You can inspect the input animation, the prediction, and the membrane/spike activity of individual neurons.
        """)
        st.caption("Use the animation controls to move through time. Use the Inspector to select a Layer and Neuron.")


    # --- Model and State Initialization ---
    weights_path = "./weights/nmnist_mlp64_16T.pt"  # Hardcoded for now
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Initialize session state variables
    if "model" not in st.session_state:
        st.session_state.model = None
    if "last_recordings" not in st.session_state:
        st.session_state.last_recordings = None
    if "last_target" not in st.session_state:
        st.session_state.last_target = None
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_sample" not in st.session_state:
        st.session_state.last_sample = None
    if "current_class" not in st.session_state:
        st.session_state.current_class = None
    if "class_samples" not in st.session_state:
        st.session_state.class_samples = {}

    # --- Auto-load model and samples on first run ---
    if st.session_state.model is None:
        with st.spinner("Loading MLP model..."):
            st.session_state.model = load_nmnist_mlp_model(weights_path, device_type=str(device.type))
            if st.session_state.model is not None:
                st.session_state.model_loaded = True

    if not st.session_state.class_samples:
        try:
            cache_path = Path("./assets/class_samples.pt")
            if cache_path.exists():
                class_samples_raw = torch.load(cache_path)
                st.session_state.class_samples = {cls: (sample, cls, 0) for cls, sample in class_samples_raw.items()}
            else:
                st.warning("Cache not found. Run: python extract_class_samples.py")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load samples: {e}")
            st.stop()

    if st.session_state.current_class is None and st.session_state.class_samples:
        st.session_state.current_class = 0
        sample, target, _ = st.session_state.class_samples[0]
        st.session_state.last_sample = sample
        st.session_state.last_target = target

    # --- Auto-run inference on first sample load ---
    if st.session_state.last_sample is not None and st.session_state.last_recordings is None:
        with st.spinner("Running inference..."):
            pred, recordings = mlp.infer_sample(st.session_state.model, st.session_state.last_sample, device=device, record=True)
            st.session_state.last_recordings = recordings
            st.session_state.last_pred = pred

    # --- Main Layout: Two columns ---
    left_pane, right_pane = st.columns([1, 1.5], gap="medium")

    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "auto_frame" not in st.session_state:
        st.session_state.auto_frame = 0

    # --- Left Pane: DVS Animation and Controls ---
    with left_pane:
        st.subheader("Input")

        if st.session_state.last_sample is not None:

            num_steps = st.session_state.last_sample.shape[0]

            if "is_playing" not in st.session_state:
                st.session_state.is_playing = False
            if "auto_frame" not in st.session_state:
                st.session_state.auto_frame = 0
            if "slider_frame" not in st.session_state:
                st.session_state.slider_frame = 0
            if "playback_speed" not in st.session_state:
                st.session_state.playback_speed = 1
  

            st.session_state.auto_frame = min(st.session_state.auto_frame, num_steps - 1)
            st.session_state.slider_frame = min(st.session_state.slider_frame, num_steps - 1)

            ctrl_cols = st.columns([1, 1, 1, 1])

            with ctrl_cols[0]:
                if st.button(
                    "⏸ Pause" if st.session_state.is_playing else "▶ Play",
                    use_container_width=True,
                    key="play_btn",
                ):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()

            with ctrl_cols[1]:
                if st.button("⏮ Reset", use_container_width=True, key="reset_btn"):
                    st.session_state.auto_frame = 0
                    st.session_state.slider_frame = 0
                    st.session_state.is_playing = False
                    st.rerun()

            with ctrl_cols[2]:
                if st.button("- Slow", use_container_width=True, key="slow_btn"):
                    st.session_state.playback_speed = max(
                        1, st.session_state.playback_speed - 1
                    )
                    st.rerun()

            with ctrl_cols[3]:
                if st.button("+ Fast", use_container_width=True, key="fast_btn"):
                    st.session_state.playback_speed = min(
                        5, st.session_state.playback_speed + 1
                    )
                    st.rerun()

            # Synchronize slider BEFORE rendering it
            if st.session_state.is_playing:
                st.session_state.slider_frame = st.session_state.auto_frame

            slider_value = st.slider(
                "Timestep",
                0,
                num_steps - 1,
                key="slider_frame",
                disabled=st.session_state.is_playing,
            )

            if not st.session_state.is_playing:
                st.session_state.auto_frame = slider_value

            st.write(f"**Class: {st.session_state.current_class}**")
            frame_idx = st.session_state.auto_frame
            frame = st.session_state.last_sample[frame_idx, 0].detach().cpu().numpy()

            frame_viz = np.zeros((frame.shape[1], frame.shape[2], 3), dtype=np.uint8)

            on = frame[0]
            off = frame[1]

            on = on / max(on.max(), 1e-6)
            off = off / max(off.max(), 1e-6)

            frame_on = (np.power(on, 0.5) * 255).clip(0, 255).astype(np.uint8)
            frame_off = (np.power(off, 0.5) * 255).clip(0, 255).astype(np.uint8)

            frame_viz[:, :, 0] = frame_on
            frame_viz[:, :, 2] = frame_off

            frame_img = upscale_nearest(frame_viz, scale=10)

            st.image(frame_img,
                    caption=(f"Frame {frame_idx} / {num_steps - 1} " 
                             f"| Speed: {st.session_state.playback_speed}x"),
            )
            
            # Auto-advance only ONCE, after rendering the image
            if st.session_state.is_playing:
                time.sleep(0.08)
                st.session_state.auto_frame = (
                    st.session_state.auto_frame + st.session_state.playback_speed
                ) % num_steps
                st.rerun()

            spacer, button_col, right_spacer = st.columns([0.4, 3, 0.6])
            with button_col:
                prev_col, next_col = st.columns([1, 1])

                # Previous sample
                if prev_col.button("Previous Sample", use_container_width=True):
                    if st.session_state.class_samples and st.session_state.current_class is not None:
                        prev_cls = (st.session_state.current_class - 1) % 10
                        attempts = 0
                        while prev_cls not in st.session_state.class_samples and attempts < 10:
                            prev_cls = (prev_cls - 1) % 10
                            attempts += 1
                        if prev_cls in st.session_state.class_samples:
                            st.session_state.current_class = prev_cls
                            sample, target, _ = st.session_state.class_samples[prev_cls]
                            st.session_state.last_sample = sample
                            st.session_state.last_target = target
                            st.session_state.last_pred = None
                            st.session_state.last_recordings = None
                            st.session_state.auto_frame = 0
                            st.session_state.reset_slider_frame = True
                            st.rerun()

                # Next sample
                if next_col.button("Next Sample", use_container_width=True):
                    if st.session_state.class_samples and st.session_state.current_class is not None:
                        next_cls = (st.session_state.current_class + 1) % 10
                        attempts = 0
                        while next_cls not in st.session_state.class_samples and attempts < 10:
                            next_cls = (next_cls + 1) % 10
                            attempts += 1
                        if next_cls in st.session_state.class_samples:
                            st.session_state.current_class = next_cls
                            sample, target, _ = st.session_state.class_samples[next_cls]
                            st.session_state.last_sample = sample
                            st.session_state.last_target = target
                            st.session_state.last_pred = None
                            st.session_state.last_recordings = None
                            st.session_state.auto_frame = 0
                            st.session_state.reset_slider_frame = True
                            st.rerun()
        else:
            st.info("Samples not loaded.")
        

    # --- Right Pane: Network Inspection and Plots ---
    with right_pane:
        st.subheader("Network Inspector")
        
        recordings = st.session_state.get("last_recordings")
        disable_selectors = recordings is None
        
        if recordings:
            st.write(f"**Pred:** {st.session_state.last_pred} | **Target:** {st.session_state.last_target}")
            layer_keys = list(recordings.keys())
        else:
            st.write("**Pred:** - | **Target:** -")
            layer_keys = ["Hidden1", "Hidden2", "Output"]

        layer = st.selectbox("Select layer", layer_keys, disabled=disable_selectors, key="layer_select")
        
        if recordings:
            layer_rec = recordings[layer]
            mem = layer_rec["mem"]
            num_neurons = mem.shape[2]
        else:
            num_neurons = 500

        neuron_index = st.slider("Neuron index", 0, max(0, num_neurons - 1), 0, disabled=disable_selectors, key="neuron_select")

        if recordings:
            spk = layer_rec["spk"]
            # Slice data up to the current timestep
            mem_trace = mem[: st.session_state.auto_frame + 1, 0, neuron_index]
            spk_trace = spk[: st.session_state.auto_frame + 1, 0, neuron_index]
            
            mem_chart, spike_chart = build_neuron_charts(mem_trace, spk_trace, layer, neuron_index)
            
            st.altair_chart(mem_chart, use_container_width=True)
            st.altair_chart(spike_chart, use_container_width=True)
        else:
            st.altair_chart(alt.Chart().mark_line().properties(height=260, title="Membrane trace"), use_container_width=True)
            st.altair_chart(alt.Chart().mark_bar().properties(height=180, title="Spike train"), use_container_width=True)
            #st.info("Run inference to see neuron activity.")

    st.stop()


# --- LIF page ---
if st.session_state.page == "lif":
    if st.button("← Back"):
        go_to("home")
        st.rerun()

    st.markdown("""
    # :material/electric_bolt: LIF Neuron Simulator

    Simulate the RC-Circuit based leaky integrate-and-fire neuron.
    """)

    with st.expander("About the model"):
        st.markdown(
            """
            In 1907, **Louis Lapicque** described neural excitation through an electrical lens — an idea that later became one of the foundations of integrate-and-fire neuron models.

            The **leaky integrate-and-fire model (LIF)** describes the neuronal membrane as an RC circuit: input current charges the membrane, leakage drives the potential back down, and a spike is emitted once the threshold is crossed.
            """
        )
        st.markdown("**Membrane time constant:**")
        st.latex(r"\tau = RC")
        st.markdown("**ODE:**")
        st.latex(r"\tau \frac{dU(t)}{dt} = -U(t) + R I_{\mathrm{in}}(t)")
        st.markdown("**Discrete Euler update:**")
        st.latex(r"U[t+1] = U[t] + \frac{dt}{\tau}\left(-U[t] + R I_{\mathrm{in}}[t]\right) - S[t] \cdot \mathrm{reset}")

        st.markdown(
            """
            The key parameters you can control in this simulator are:

            - **Input type**: constant current or spike train.
            - **Amplitude**: strength of the input current or input spikes.
            - **Spike frequency/probability**: Frequency or probability of input spikes.
            - **Number of steps** and **dt**: simulation length and temporal resolution. 
            - **R** and **C**: membrane resistance and capacitance.
            - **Threshold**: membrane threshold at which the neuron fires.
            - **Reset mechanism**: subtract the threshold or reset the membrane to zero after a spike.
            """
        )


    # Sidebar controls
    with st.sidebar:
        st.header("LIF Parameters")

        input_mode = st.selectbox("Input type", ["Constant current", "Spike train"])

        amplitude = 0.3
        start_step = 5
        spike_mode = "Regular"
        spike_amplitude = 0.5
        spike_frequency = 20.0
        spike_probability = 0.40

        if input_mode == "Constant current":
            amplitude = st.slider("Input current", 0.0, 1.5, 0.3, 0.01)
        else:
            spike_mode = st.segmented_control("Spike train type", ["Regular", "Random"], default="Regular")
            spike_amplitude = st.slider("Spike amplitude", 0.0, 2.0, 0.5, 0.01)
            if spike_mode == "Regular":
                spike_frequency = st.slider("Spike frequency (Hz)", 1.0, 500.0, 20.0, 1.0)
            else:
                spike_probability = st.slider("Spike probability", 0.0, 1.0, 0.40, 0.01)

        st.divider()

        num_steps = st.slider("Number of steps", 50, 500, 200, 10)
        if input_mode == "Constant current":
            start_step = min(start_step, num_steps - 1)

        dt = st.slider("dt (s)", 1e-4, 5e-3, 1e-3, 1e-4, format="%.4f")
        R = st.slider("R (Ω)", 0.5, 15.0, 5.1, 0.1)
        C = st.slider("C (F)", 1e-3, 20e-3, 5e-3, 1e-3, format="%.3f")
        threshold = st.slider("Threshold", 0.2, 2.0, 1.0, 0.01)
        reset_mechanism = st.selectbox("Reset mechanism", ["subtract", "zero"])

    # Simulation
    spk_input = None
    if input_mode == "Spike train":
        spk_input, cur_in = build_spike_train(
            num_steps=num_steps,
            mode=spike_mode,
            spike_amplitude=spike_amplitude,
            spike_frequency=spike_frequency,
            spike_probability=spike_probability,
            dt=dt,
            start_step=5,
        )
    else:
        cur_in = build_current(num_steps=num_steps, amplitude=amplitude, start_step=5)

    mem_rec, spk_rec = simulate_lapicque(
        num_steps=num_steps,
        cur_in=cur_in,
        threshold=threshold,
        dt=dt,
        R=R,
        C=C,
        reset_mechanism=reset_mechanism,
    )

    spike_count = int(spk_rec.sum().item())
    peak_mem = float(mem_rec.max().item())
    tau_mem = R * C
    sim_time = num_steps * dt
    firing_rate = spike_count / sim_time if sim_time > 0 else 0.0

    # Layout
    trace_cell = st.container(border=True)
    with trace_cell:
        st.subheader("Simulation traces")
        input_chart, mem_chart, spike_chart = build_trace_charts(
            cur_in=cur_in,
            mem_rec=mem_rec,
            spk_rec=spk_rec,
            thr_line=threshold,
            spk_input=spk_input,
            input_mode=input_mode,
        )
        st.altair_chart(input_chart, use_container_width=True)
        st.altair_chart(mem_chart, use_container_width=True)
        st.altair_chart(spike_chart, use_container_width=True)
    
    st.stop()

