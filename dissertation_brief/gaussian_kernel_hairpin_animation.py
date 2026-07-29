"""
Gaussian-kernel intuition animation on a folded 1D manifold.

The animation is designed for diffusion-map presentations:
1. A front view (looking down z) makes the manifold appear almost like a straight line.
2. The camera rotates to reveal a hairpin fold in the x-z plane.
3. The Gaussian bandwidth changes smoothly. As sqrt(epsilon) grows, affinities
   extend farther and eventually connect to the neighbouring fold.

Main entry point:
    export_gaussian_hairpin_animation(...)

The hairpin is sampled uniformly in arc length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _ease01(u: float) -> float:
    """Cosine ease-in/ease-out on [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * u)


def make_hairpin(
    n_points: int = 140,
    branch_length: float = 4.0,
    fold_gap: float = 0.85,
    y_wiggle: float = 0.0,
) -> np.ndarray:
    """Create a smooth hairpin curve sampled uniformly in arc length."""
    if n_points < 10:
        raise ValueError("n_points should be at least 10.")
    if branch_length <= 0 or fold_gap <= 0:
        raise ValueError("branch_length and fold_gap must be positive.")

    L = float(branch_length)
    r = float(fold_gap) / 2.0
    turn_length = np.pi * r
    total_length = 2.0 * L + turn_length

    s = np.linspace(0.0, total_length, n_points)
    x = np.empty_like(s)
    z = np.empty_like(s)

    lower = s <= L
    turn = (s > L) & (s <= L + turn_length)
    upper = s > L + turn_length

    x[lower] = -L / 2.0 + s[lower]
    z[lower] = 0.0

    u = s[turn] - L
    theta = -np.pi / 2.0 + u / r
    x[turn] = L / 2.0 + r * np.cos(theta)
    z[turn] = r + r * np.sin(theta)

    u = s[upper] - (L + turn_length)
    x[upper] = L / 2.0 - u
    z[upper] = fold_gap

    if y_wiggle == 0:
        y = np.zeros_like(s)
    else:
        y = float(y_wiggle) * np.sin(2.0 * np.pi * s / total_length)

    return np.column_stack([x, y, z])


def choose_lower_branch_focal_node(
    points: np.ndarray,
    branch_length: float,
    lower_fraction: float = 0.43,
) -> int:
    """Choose a focal node on the lower branch."""
    lower_fraction = float(np.clip(lower_fraction, 0.0, 1.0))
    target_x = -branch_length / 2.0 + lower_fraction * branch_length
    lower_candidates = np.flatnonzero(np.isclose(points[:, 2], 0.0, atol=1e-10))
    if len(lower_candidates) == 0:
        raise RuntimeError("Could not identify lower-branch nodes.")
    return int(lower_candidates[np.argmin(np.abs(points[lower_candidates, 0] - target_x))])


def gaussian_affinity(points: np.ndarray, focal_idx: int, epsilon: float) -> np.ndarray:
    """W_ij = exp(-||x_i-x_j||^2 / epsilon) from one focal node."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    d2 = np.sum((points - points[focal_idx]) ** 2, axis=1)
    return np.exp(-d2 / float(epsilon))


def export_gaussian_hairpin_animation(
    output_path: str | Path = "gaussian_hairpin.mp4",
    *,
    # Geometry
    n_points: int = 140,
    branch_length: float = 4.0,
    fold_gap: float = 0.85,
    y_wiggle: float = 0.0,
    focal_fraction: float = 0.43,
    # Bandwidth: specify visually via e-fold radius sqrt(epsilon)
    radius_small: Optional[float] = None,
    radius_large: Optional[float] = None,
    # Timing
    hold_front_seconds: float = 1.8,
    rotate_seconds: float = 3.2,
    hold_side_seconds: float = 0.8,
    expand_seconds: float = 4.5,
    hold_large_seconds: float = 1.0,
    shrink_seconds: float = 0.0,
    final_hold_seconds: float = 0.8,
    # Camera
    front_elev: float = 90.0,
    front_azim: float = -90.0,
    side_elev: float = 8.0,
    side_azim: float = -90.0,
    # Affinity visuals
    affinity_line_threshold: float = 0.06,
    max_connection_lines: int = 45,
    line_width_min: float = 0.4,
    line_width_max: float = 3.2,
    line_alpha_max: float = 0.78,
    node_size: float = 28.0,
    focal_size: float = 115.0,
    affinity_node_size_boost: float = 40.0,
    # Epsilon sphere
    show_epsilon_ball: bool = True,
    sphere_alpha: float = 0.10,
    sphere_resolution: Tuple[int, int] = (24, 24),
    # Figure / output
    figsize: Tuple[float, float] = (10.5, 6.2),
    dpi: int = 120,
    fps: int = 24,
    bitrate: int = 3000,
    show_axes: bool = True,
    show_grid: bool = True,
    show_colorbar: bool = True,
    show_equation: bool = True,
    show_bandwidth_text: bool = True,
    show_view_text: bool = True,
    title: str = "Gaussian affinity on a folded manifold",
    orthographic: bool = True,
    clean_axes: bool = True,
    repeat: bool = False,
    return_objects: bool = False,
):
    """
    Export an MP4 explaining Gaussian affinities on a folded manifold.

    The displayed epsilon-ball radius is sqrt(epsilon), because
        exp(-||x_i-x_j||^2 / epsilon) = exp(-1)
    when ||x_i-x_j|| = sqrt(epsilon).

    Default scene:
      front hold -> smooth camera rotation -> side hold ->
      smooth bandwidth expansion -> optional smooth shrink -> final hold.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pts = make_hairpin(n_points, branch_length, fold_gap, y_wiggle)
    focal_idx = choose_lower_branch_focal_node(pts, branch_length, focal_fraction)
    focal = pts[focal_idx]

    if radius_small is None:
        radius_small = 0.38 * fold_gap
    if radius_large is None:
        radius_large = 1.25 * fold_gap
    # if radius_small <= 0 or radius_large <= 0:
    #     raise ValueError("radius_small and radius_large must be positive.")
    # if radius_large <= radius_small:
    #     raise ValueError("radius_large must be greater than radius_small.")

    eps_small = float(radius_small) ** 2
    eps_large = float(radius_large) ** 2

    durations = np.array([
        hold_front_seconds, rotate_seconds, hold_side_seconds,
        expand_seconds, hold_large_seconds, shrink_seconds,
        final_hold_seconds,
    ], dtype=float)
    if np.any(durations < 0):
        raise ValueError("Animation durations must be non-negative.")

    boundaries = np.cumsum(durations)
    total_seconds = float(boundaries[-1])
    n_frames = max(2, int(round(total_seconds * fps)))

    pad = max(radius_large * 1.2, 0.25)
    x_min, x_max = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    y_span = max(radius_large * 3, 0.55)
    z_min = min(pts[:, 2].min(), focal[2] - radius_large) - 0.5
    z_max = max(pts[:, 2].max(), focal[2] + radius_large) + 0.15

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    centreline, = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.25, alpha=0.30)

    W0 = gaussian_affinity(pts, focal_idx, eps_small)
    node_colors = cmap(norm(W0))
    node_colors[:, 3] = 0.22 + 0.78 * W0
    sizes = node_size + affinity_node_size_boost * W0
    nodes = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=sizes, facecolors=node_colors, depthshade=False)

    focal_artist = ax.scatter(
        [focal[0]], [focal[1]], [focal[2]], s=focal_size, marker="o",
        depthshade=False, label=r"focal node $x_i$"
    )

    Wmax = gaussian_affinity(pts, focal_idx, eps_large)
    candidates = np.flatnonzero((np.arange(len(pts)) != focal_idx) & (Wmax >= affinity_line_threshold))
    if len(candidates) > max_connection_lines:
        candidates = candidates[np.argsort(Wmax[candidates])[-max_connection_lines:]]

    connection_lines = []
    connection_color = centreline.get_color()
    for j in candidates:
        line, = ax.plot(
            [focal[0], pts[j, 0]], [focal[1], pts[j, 1]], [focal[2], pts[j, 2]],
            linewidth=line_width_min, alpha=0.0, color=connection_color,
        )
        connection_lines.append((j, line))

    equation_text = None
    bandwidth_text = None
    view_text = None
    if show_equation:
        equation_text = ax.text2D(
            0.02, 0.965,
            r"$W_{ij}=\exp\!\left(-\|x_i-x_j\|^2/\varepsilon\right)$",
            transform=ax.transAxes, va="top", fontsize=14
        )
    if show_bandwidth_text:
        bandwidth_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes, va="top", fontsize=12)
    if show_view_text:
        view_text = ax.text2D(0.98, 0.965, "", transform=ax.transAxes, va="top", ha="right", fontsize=11)

    ax.set_title(title, pad=12)
    if orthographic:
        ax.set_proj_type("ortho")
    if show_axes:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
    else:
        ax.set_axis_off()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_span, y_span)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, 2 * y_span, z_max - z_min))

    if show_grid:
        ax.xaxis.pane.set_alpha(0.12)
        ax.yaxis.pane.set_alpha(0.12)
        ax.zaxis.pane.set_alpha(0.12)
        ax.grid(True)
    else:
        ax.grid(False)

    if clean_axes and show_axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.0)

    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.08, fraction=0.03)
        cb.set_label(r"Affinity $W_{ij}$")

    nu, nv = sphere_resolution
    u = np.linspace(0.0, 2.0 * np.pi, int(nu))
    v = np.linspace(0.0, np.pi, int(nv))
    sphere_unit_x = np.outer(np.ones_like(u), np.cos(v))
    sphere_unit_y = np.outer(np.cos(u), np.sin(v))
    sphere_unit_z = np.outer(np.sin(u), np.sin(v))
    sphere_holder = [None]; sphere_grid_holder = [None]

    def scene_state(t: float):
        b0, b1, b2, b3, b4, b5, b6 = boundaries
        if t < b0:
            elev, azim, radius = front_elev, front_azim, radius_small
        elif t < b1:
            u0 = _ease01((t - b0) / max(rotate_seconds, 1e-12))
            elev = front_elev + u0 * (side_elev - front_elev)
            azim = front_azim + u0 * (side_azim - front_azim)
            radius = radius_small
        elif t < b2:
            elev, azim, radius = side_elev, side_azim, radius_small
        elif t < b3:
            u0 = _ease01((t - b2) / max(expand_seconds, 1e-12))
            elev, azim = side_elev, side_azim
            radius = radius_small + u0 * (radius_large - radius_small)
        elif t < b4:
            elev, azim, radius = side_elev, side_azim, radius_large
        elif shrink_seconds > 0 and t < b5:
            u0 = _ease01((t - b4) / max(shrink_seconds, 1e-12))
            elev, azim = side_elev, side_azim
            radius = radius_large + u0 * (radius_small - radius_large)
        else:
            elev, azim = side_elev, side_azim
            radius = radius_small if shrink_seconds > 0 else radius_large
        return float(elev), float(azim), float(radius)

    def update(frame: int):
        t = min(frame / fps, total_seconds - 1.0 / fps)
        elev, azim, radius = scene_state(t)
        epsilon = radius ** 2
        W = gaussian_affinity(pts, focal_idx, epsilon)

        ax.view_init(elev=elev, azim=azim)

        rgba = cmap(norm(W))
        # rgba[:, 3] = 0.18 + 0.82 * W
        rgba[:, 3] = 1.0
        rgba[focal_idx, 3] = 0.0
        nodes.set_facecolors(rgba)
        nodes.set_edgecolors(rgba)
        nodes.set_sizes(node_size + affinity_node_size_boost * W)

        for j, line in connection_lines:
            w = float(W[j])
            if w < affinity_line_threshold:
                line.set_alpha(0.0)
                line.set_linewidth(line_width_min)
            else:
                scaled = (w - affinity_line_threshold) / max(1.0 - affinity_line_threshold, 1e-12)
                scaled = float(np.clip(scaled, 0.0, 1.0))
                line.set_alpha(line_alpha_max * scaled)
                line.set_linewidth(line_width_min + (line_width_max - line_width_min) * scaled)

        if show_epsilon_ball:
            if sphere_holder[0] is not None:
                sphere_holder[0].remove()

            xs = focal[0] + radius * sphere_unit_x
            ys = focal[1] + radius * sphere_unit_y
            zs = focal[2] + radius * sphere_unit_z
            sphere_holder[0] = ax.plot_surface(
                xs, ys, zs, alpha=sphere_alpha, linewidth=0,
                antialiased=True, shade=False, color="green"
            )

            if sphere_grid_holder[0] is not None:
                sphere_grid_holder[0].remove()
            sphere_grid_holder[0] = ax.plot_wireframe(
                xs, ys, zs, rstride=2, cstride=2, 
                linewidth=0.5, alpha=0.15
            )

        if bandwidth_text is not None:
            bandwidth_text.set_text(
                rf"$\sqrt{{\varepsilon}}={radius:.2f}$" + "\n" + rf"$\varepsilon={epsilon:.2f}$"
            )
        if view_text is not None:
            if t < boundaries[0]:
                view_text.set_text(r"$x$-$y$ projection")
            elif t < boundaries[1]:
                view_text.set_text("$x$-$y$-$z$ view")
            else:
                view_text.set_text(r"$x$-$z$ view")

        if show_axes:
            ax.set_xlabel(r"$x$")
            if t < boundaries[0]:
                ax.set_ylabel(r"$y$"); ax.set_zlabel("")
            elif t < boundaries[1]:
                ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")
            else:
                ax.set_ylabel(""); ax.set_zlabel(r"$z$")

        artists = [nodes, focal_artist, centreline]
        artists.extend(line for _, line in connection_lines)
        if sphere_holder[0] is not None:
            artists.append(sphere_holder[0])
        if sphere_grid_holder[0] is not None:
            artists.append(sphere_grid_holder[0])
        if equation_text is not None:
            artists.append(equation_text)
        if bandwidth_text is not None:
            artists.append(bandwidth_text)
        if view_text is not None:
            artists.append(view_text)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps,
        blit=False, repeat=repeat
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, metadata={"title": title})
    ani.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

    result = {
        "output_path": str(output_path),
        "points": pts,
        "focal_idx": focal_idx,
        "focal_point": focal,
        "epsilon_small": eps_small,
        "epsilon_large": eps_large,
        "radius_small": radius_small,
        "radius_large": radius_large,
        "total_seconds": total_seconds,
        "n_frames": n_frames,
    }
    if return_objects:
        result["animation"] = ani
    return result


if __name__ == "__main__":
    info = export_gaussian_hairpin_animation(
        output_path="gaussian_hairpin_demo.mp4",
        n_points=140,
        branch_length=4.0,
        fold_gap=0.85,
        focal_fraction=0.43,
        radius_small=0.30,
        radius_large=1.05,
        hold_front_seconds=1.5,
        rotate_seconds=2.8,
        hold_side_seconds=0.6,
        expand_seconds=3.8,
        hold_large_seconds=0.8,
        final_hold_seconds=0.6,
        fps=24,
        dpi=110,
    )
    print(info)