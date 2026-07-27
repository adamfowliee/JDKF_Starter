import math
import shutil
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize


def _ensure_ffmpeg():
    """Find ffmpeg directly or via imageio-ffmpeg, and register it with Matplotlib."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(
                "ffmpeg could not be found. Install ffmpeg or `pip install imageio-ffmpeg`."
            ) from exc
    mpl.rcParams["animation.ffmpeg_path"] = exe
    return exe


def _schedule_weights(t, n_betas, hold_seconds, fade_seconds):
    """Return opacity weights for beta paths at time t (seconds).

    Sequence:
      beta_0 holds -> cross-fade beta_0 to beta_1 -> beta_1 holds -> ...
    The final beta simply holds until the end.
    """
    weights = np.zeros(n_betas, dtype=float)

    if n_betas == 1:
        weights[0] = 1.0
        return weights

    # First path starts fully visible.
    cursor = 0.0
    for i in range(n_betas):
        # Hold beta i.
        if cursor <= t < cursor + hold_seconds:
            weights[i] = 1.0
            return weights
        cursor += hold_seconds

        # Cross-fade from beta i to beta i+1, unless i is final beta.
        if i < n_betas - 1:
            if cursor <= t < cursor + fade_seconds:
                u = (t - cursor) / fade_seconds
                # Smoothstep gives a gentler fade than a linear ramp.
                u = u * u * (3.0 - 2.0 * u)
                weights[i] = 1.0 - u
                weights[i + 1] = u
                return weights
            cursor += fade_seconds

    weights[-1] = 1.0
    return weights


def export_rotating_beta_animation(
    *,
    Psi,
    rho,
    V,
    A_dist_sym,
    graph_path_fn,
    s,
    e,
    betas=(0.0, 0.5, 0.75, 1.5, 4.0),
    output_path="beta_paths_rotation.mp4",
    # Timing
    fps=30,
    hold_seconds=2.0,
    fade_seconds=1.0,
    final_hold_seconds=1.5,
    # Camera
    camera_elev=22.0,
    camera_azim_start=-60.0,
    rotation_deg_per_sec=24.0,
    rotation_coords=None,
    zoom=1.0,
    n_rotations,
    # Figure / appearance
    figsize=(16, 9),
    dpi=120,
    cloud_size=12,
    cloud_alpha=0.45,
    path_linewidth=4.5,
    path_marker_size=4.0,
    linear_linewidth=2.4,
    show_linear_path=True,
    background=True,
    title=None,
):
    """Export a rotating 3D MP4 with beta graph paths cross-fading.

    Assumes:
      Psi.shape == (N, 3) or has at least 3 columns
      V.shape == (N,)
      graph_path_fn(A_dist_sym, beta, V, s, e) returns node indices

    Camera controls:
      rotation_coords=None rotates around the centre of the data cloud.
      rotation_coords=(psi1, psi2) recentres the 3D axes so the camera
      rotates around a vertical line through that (psi1, psi2) location.

    Appearance:
      background=False removes the 3D panes and grid while keeping axes,
      labels, data, and a white video background.
    """
    _ensure_ffmpeg()

    Psi = np.asarray(Psi)
    V = np.asarray(V)
    rho = np.asarray(rho)
    betas = list(betas)

    if Psi.ndim != 2 or Psi.shape[1] < 3:
        raise ValueError("Psi must have shape (N, 3) or at least three columns.")
    if len(V) != len(Psi):
        raise ValueError("V must have one value for each row of Psi.")
    if len(rho) != len(Psi):
        raise ValueError("rho must have one value for each row of Psi.")
    if len(betas) == 0:
        raise ValueError("Provide at least one beta value.")

    if rotation_coords is not None:
        if len(rotation_coords) != 2:
            raise ValueError("rotation_coords must be None or a 2-tuple (psi1, psi2).")
        rotation_coords = tuple(float(v) for v in rotation_coords)

    # Compute all paths once. This is much faster than recomputing inside animation frames.
    beta_paths = {}
    for beta in betas:
        node_idx = np.asarray(graph_path_fn(A_dist_sym, beta, V, s, e), dtype=int)
        if node_idx.ndim != 1 or len(node_idx) < 2:
            raise ValueError(f"Path for beta={beta} did not return a valid index sequence.")
        beta_paths[beta] = Psi[node_idx, :3]

    linear_path = np.linspace(Psi[s, :3], Psi[e, :3], 80)

    # Duration: one hold for each beta, one cross-fade between neighbours, plus final hold.
    total_seconds = (
        len(betas) * hold_seconds
        + max(0, len(betas) - 1) * fade_seconds
        + final_hold_seconds
    )
    n_frames = max(2, int(round(total_seconds * fps)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ---- Static cloud ----
    density_colour = V
    norm = Normalize(vmin=np.nanmin(density_colour), vmax=np.nanmax(density_colour))
    cloud = ax.scatter(
        Psi[:, 0], Psi[:, 1], Psi[:, 2],
        c=density_colour,
        cmap="viridis",
        norm=norm,
        s=cloud_size,
        alpha=cloud_alpha,
        depthshade=False,
        linewidths=0,
    )

    cbar = fig.colorbar(cloud, ax=ax, pad=0.03, shrink=0.72)
    cbar.set_label(r"Density penalty ($\log(\hat\rho)$)")

    if show_linear_path:
        ax.plot(
            linear_path[:, 0], linear_path[:, 1], linear_path[:, 2],
            linestyle="--", linewidth=linear_linewidth, color="red",
            alpha=0.95, label="Linear path",
        )

    # Start and end points remain visible throughout.
    ax.scatter(
        [Psi[s, 0]], [Psi[s, 1]], [Psi[s, 2]],
        s=100, c="lime", edgecolors="black", linewidths=1.5,
        depthshade=False, label="Start", zorder=20,
    )
    ax.scatter(
        [Psi[e, 0]], [Psi[e, 1]], [Psi[e, 2]],
        s=100, c="red", edgecolors="black", linewidths=1.5,
        depthshade=False, label="End", zorder=20,
    )

    # One line + marker artist for each beta. Alpha controls the cross-fade.
    colours = plt.cm.plasma(np.linspace(0.12, 0.88, len(set(betas))))
    period = 2 * (len(colours) - 1)
    indices = np.abs((np.arange(len(betas)) + len(colours) - 1) % period - (len(colours) - 1))
    colours_rep = colours[indices]
    beta_artists = []
    for colour, beta in zip(colours_rep, betas):
        P = beta_paths[beta]
        line, = ax.plot(
            P[:, 0], P[:, 1], P[:, 2],
            linewidth=path_linewidth,
            marker="o",
            markersize=path_marker_size,
            color=colour,
            alpha=0.0,
            label=fr"Graph $\beta={beta:g}$",
            zorder=10,
        )
        beta_artists.append(line)

    # Axes / framing
    ax.set_xlabel(r"$\psi_1$")
    ax.set_ylabel(r"$\psi_2$")
    ax.set_zlabel(r"$\psi_3$")
    if title:
        ax.set_title(title, pad=18)

    # Fixed equal-ish visual scale, preventing the scene from 'breathing' during animation.
    # Matplotlib rotates the camera around the centre of the 3D axes box.
    # By centring the x/y limits on rotation_coords, that centre becomes the
    # vertical rotation axis through the requested (psi1, psi2) location.
    mins = np.nanmin(Psi[:, :3], axis=0)
    maxs = np.nanmax(Psi[:, :3], axis=0)
    data_centres = 0.5 * (mins + maxs)

    if rotation_coords is None:
        centre_x, centre_y = data_centres[0], data_centres[1]
    else:
        centre_x, centre_y = rotation_coords
    centre_z = data_centres[2]

    required_half_ranges = [
        np.nanmax(np.abs(Psi[:, 0] - centre_x)),
        np.nanmax(np.abs(Psi[:, 1] - centre_y)),
        np.nanmax(np.abs(Psi[:, 2] - centre_z)),
    ]
    half_range = float(np.nanmax(required_half_ranges))
    if not np.isfinite(half_range) or half_range == 0:
        half_range = 1.0
    half_range *= 1.05

    ax.set_xlim((centre_x - half_range)/zoom, (centre_x + half_range)/zoom)
    ax.set_ylim((centre_y - half_range)/zoom, (centre_y + half_range)/zoom)
    ax.set_zlim((centre_z - half_range)/zoom, (centre_z + half_range)/zoom)

    ax.set_box_aspect((1, 1, 0.6))

    if not background:
        # Remove the grey 3D panes and grid without hiding the axes themselves.
        ax.set_facecolor("none")
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        fig.patch.set_facecolor("white")

    # Small beta label in the axes, separate from the legend.
    beta_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=17, fontweight="bold",
    )

    # Compact legend: linear/start/end only. Beta value is shown by beta_text.
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, label in enumerate(labels) if not label.startswith("Graph")]
    if keep:
        ax.legend(
            [handles[i] for i in keep], [labels[i] for i in keep],
            loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
            frameon=False,
        )

    plt.tight_layout()

    active_sequence_duration = (
        len(betas) * hold_seconds + max(0, len(betas) - 1) * fade_seconds
    )

    def update(frame):
        t = frame / fps

        # Camera rotates continuously around the vertical psi_3 axis.
        azim = camera_azim_start + rotation_deg_per_sec * t
        azim = camera_azim_start + frame * 360 * n_rotations / n_frames
        ax.view_init(elev=camera_elev, azim=azim)

        # During the final hold, freeze on the final beta.
        schedule_t = min(t, max(0.0, active_sequence_duration - 1e-9))
        weights = _schedule_weights(schedule_t, len(betas), hold_seconds, fade_seconds)

        for artist, alpha in zip(beta_artists, weights):
            artist.set_alpha(float(alpha))

        # Label the dominant path. During a cross-fade, show the transition explicitly.
        visible = np.flatnonzero(weights > 0.02)
        if len(visible) == 1:
            beta_text.set_text(fr"$\beta={betas[visible[0]]:g}$")
        elif len(visible) >= 2:
            i, j = visible[0], visible[-1]
            beta_text.set_text(
                fr"$\beta={betas[i]:g}\;\rightarrow\;{betas[j]:g}$"
            )
        else:
            beta_text.set_text("")

        return [*beta_artists, beta_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,  # 3D axes do not work reliably with blitting.
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

    return output_path