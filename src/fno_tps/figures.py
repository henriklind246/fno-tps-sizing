"""Rendering layer for the four-figure result set.

Reads the artifacts written by :mod:`fno_tps.figure_data` and renders them.
Importing this module pulls in neither matplotlib nor the solver stack, and
rendering never re-runs the finite-volume solver or the surrogate, so restyling
a figure costs nothing beyond the redraw.

Encoding invariant across every figure: **linestyle encodes the solver**
(solid = FV, dashed = FNO) and colour encodes whatever that panel varies.

Palette slots come from the bundled data-visualization design system and were
checked with its validator against a white print surface: the categorical sets
below pass the lightness band, chroma floor, all-pairs CVD separation,
normal-vision floor, and 3:1 contrast. The two sequential ramps are single-hue
and monotone in lightness, blue for temperature rise and orange for the second
sequential context (absolute error).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FV_COLOR = "#2a78d6"
FNO_COLOR = "#eb6834"
THICKNESS_COLORS = ("#2a78d6", "#eb6834", "#4a3aa7")
MATERIAL_X_COLOR = "#2a78d6"
MATERIAL_Y_COLOR = "#a16c00"
CAPACITY_COLOR = "#4a3aa7"
RADIATION_COLOR = "#4a3aa7"
NET_FLUX_COLOR = "#2a78d6"
LIMIT_COLOR = "#d03b3b"
FALSE_FEASIBLE_COLOR = "#d03b3b"
FALSE_INFEASIBLE_COLOR = "#2a78d6"
AGREEMENT_COLOR = "#898781"

DELTA_T_RAMP = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)
ERROR_RAMP = (
    "#fdd6c8", "#f5ae95", "#ea845e", "#de5412", "#b43f01", "#8a2e00", "#611e00",
)

#: Below this bond-to-stack fraction the bond is unreadable at true scale and
#: figure 1 gains a depth-zoom row.
BOND_ZOOM_FRACTION = 0.08
#: Bilinear display refinement for Figure 1. This smooths cell-centred fields
#: for reading but does not add FV/FNO information or change reported errors.
FIELD_DISPLAY_REFINEMENT = 8
#: Power-law colour normalization for Figure 1 temperature fields. Values
#: below one expand low-to-mid-temperature contrast while preserving a shared
#: physical scale and the exact numeric temperature labels on the colourbar.
TEMPERATURE_COLOR_GAMMA = 0.60
FIGURE_NAMES = {
    1: "fig1_field_comparison",
    2: "fig2_critical_histories",
    3: "fig3_sizing_curves",
    4: "fig4_margin_scatter",
}


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Rendering result figures requires matplotlib. Install it with:\n"
            "  pip install -e '.[plot]'"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": (0, (3, 3)),
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    })
    return matplotlib, plt


def _ramp(name: str, colors: Sequence[str]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def _save(fig, output_dir: str | Path, name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = destination / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def _millimetres(value: float) -> str:
    return f"{value * 1e3:g} mm"


def _interpolate_cell_field(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    field: np.ndarray,
    factor: int = FIELD_DISPLAY_REFINEMENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly refine a cell-centred field for display only.

    Refining each original cell separately preserves the nonuniform material
    grid in ``x``. Values are held constant from the outermost cell centres to
    the physical boundaries rather than extrapolated beyond the solved domain.
    """
    x_faces = np.asarray(x_faces, dtype=np.float64)
    y_faces = np.asarray(y_faces, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    if factor < 1:
        raise ValueError("Display refinement factor must be at least one.")
    expected = (len(x_faces) - 1, len(y_faces) - 1)
    if values.shape != expected:
        raise ValueError(
            f"Field shape {values.shape} does not match face grid {expected}."
        )
    if np.any(np.diff(x_faces) <= 0.0) or np.any(np.diff(y_faces) <= 0.0):
        raise ValueError("Field face coordinates must be strictly increasing.")

    def subdivide(faces: np.ndarray) -> np.ndarray:
        pieces = [
            np.linspace(left, right, factor, endpoint=False)
            for left, right in zip(faces[:-1], faces[1:])
        ]
        return np.concatenate((*pieces, faces[-1:]))

    refined_x_faces = subdivide(x_faces)
    refined_y_faces = subdivide(y_faces)
    x_centres = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centres = 0.5 * (y_faces[:-1] + y_faces[1:])
    refined_x_centres = 0.5 * (
        refined_x_faces[:-1] + refined_x_faces[1:]
    )
    refined_y_centres = 0.5 * (
        refined_y_faces[:-1] + refined_y_faces[1:]
    )

    along_y = np.stack([
        np.interp(
            refined_y_centres,
            y_centres,
            row,
            left=float(row[0]),
            right=float(row[-1]),
        )
        for row in values
    ])
    refined = np.stack([
        np.interp(
            refined_x_centres,
            x_centres,
            along_y[:, column],
            left=float(along_y[0, column]),
            right=float(along_y[-1, column]),
        )
        for column in range(along_y.shape[1])
    ], axis=1)
    return refined_x_faces, refined_y_faces, refined


def _headroom(ax, limit: float, series: Sequence[np.ndarray]) -> None:
    """Keep the limit line clear of the axes edge so labels and legends fit."""
    values = np.concatenate([np.asarray(values).ravel() for values in series])
    low = float(min(values.min(), limit))
    high = float(max(values.max(), limit))
    span = (high - low) or max(abs(high), 1.0)
    ax.set_ylim(low - 0.05 * span, high + 0.16 * span)


def _limit_line(ax, value: float, label: str) -> None:
    ax.axhline(
        value,
        color=LIMIT_COLOR,
        linestyle=(0, (6, 3, 1, 3)),
        linewidth=1.4,
        zorder=1.5,
    )
    ax.annotate(
        label,
        xy=(0.995, value),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK_SECONDARY,
    )


def _reserve(fig, bottom: float = 0.0, top: float = 0.0) -> None:
    """Shrink the layout area so figure-level text and legends get their own strips."""
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, bottom, 1.0, 1.0 - bottom - top))


def _footnote(
    fig,
    text: str | None,
    height: float = 0.055,
    top: float = 0.0,
) -> None:
    """Reserve a strip at the bottom of the figure and write a caveat into it."""
    if not text:
        if top:
            _reserve(fig, top=top)
        return
    _reserve(fig, bottom=height, top=top)
    fig.text(
        0.006,
        0.5 * height,
        text,
        ha="left",
        va="center",
        fontsize=7,
        color=INK_MUTED,
        wrap=True,
    )


def _physics_note(manifest: dict[str, Any]) -> str | None:
    physics = manifest.get("physics")
    if not physics:
        return None
    pressure = float(physics["pressure_Pa"]) / 1.0e3
    if physics["temperature_dependent"]:
        material = (
            r"$k_x(T)$, $k_y(T)$, $\rho c_p(T)$, and enthalpy"
        )
    else:
        material = "effective constant properties"
    radiation = (
        f"coupled RCG re-radiation (ε={physics['emissivity']:.2f}, "
        f"T∞={physics['radiation_sink_temperature_K']:g} K)"
        if physics["reradiation_enabled"]
        else "no hot-face re-radiation"
    )
    return (
        f"FV reference: {material} at fixed p={pressure:.2f} kPa "
        f"(not a transient pressure solve); {radiation}."
    )


# --------------------------------------------------------------------------- #
# Figure 1 — FV / FNO / |error| thermal fields
# --------------------------------------------------------------------------- #


def render_figure1(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    _, plt = _pyplot()
    from matplotlib.colors import Normalize, PowerNorm
    from matplotlib.patches import Rectangle

    info = manifest["figure1"]
    if info is None:
        raise ValueError("The artifact has no figure 1 section; rebuild it with -f 1.")
    # Depths span 0.25 mm (bond) to 26 mm (stack); millimetres keep both readable.
    x_faces = arrays["f1_x_faces"] * 1e3
    y_faces = arrays["f1_y_faces"] * 1e3
    fv, fno = arrays["f1_fv"], arrays["f1_fno"]
    plot_x_faces, plot_y_faces, plot_fv = _interpolate_cell_field(
        x_faces,
        y_faces,
        fv,
    )
    _, _, plot_fno = _interpolate_cell_field(
        x_faces,
        y_faces,
        fno,
    )
    plot_error = np.abs(plot_fno - plot_fv)
    vmin, vmax = info["delta_t_limits"]
    error_max = max(info["error_max_K"], 1e-9)
    tps_bond = info["x_tps_bond"] * 1e3
    bond_backing = info["x_bond_backing"] * 1e3
    bond_mm = info["bond_thickness"] * 1e3

    delta_cmap = _ramp("delta_t", DELTA_T_RAMP)
    error_cmap = _ramp("abs_error", ERROR_RAMP)
    delta_norm = PowerNorm(
        gamma=TEMPERATURE_COLOR_GAMMA,
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )
    error_norm = Normalize(vmin=0.0, vmax=error_max, clip=True)

    stack = info["d_tps"] + info["bond_thickness"] + info["backing_thickness"]
    zoom = info["bond_thickness"] / stack < BOND_ZOOM_FRACTION
    zoom_low = max(float(x_faces[0]), tps_bond - 4.0 * bond_mm)
    zoom_high = min(
        float(x_faces[-1]),
        bond_backing + 0.35 * info["backing_thickness"] * 1e3,
    )

    field_rows = 2 if zoom else 1
    has_property_strip = all(
        key in arrays
        for key in (
            "f1_property_temperature",
            "f1_property_k_x",
            "f1_property_k_y",
            "f1_property_rho_c",
            "f1_property_alpha_x",
            "f1_property_alpha_y",
        )
    )
    rows = field_rows + int(has_property_strip)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(
            10.5,
            (8.7 if zoom else 6.6) if has_property_strip
            else (6.4 if zoom else 3.9),
        ),
        sharex=False,
        squeeze=False,
        gridspec_kw={
            "height_ratios": (
                ([1.0] * field_rows + [0.72])
                if has_property_strip
                else [1.0] * field_rows
            )
        },
    )

    panels = (
        ("FV  ΔT", plot_fv, delta_cmap, delta_norm),
        ("FNO  ΔT", plot_fno, delta_cmap, delta_norm),
        (
            "|ΔT$_{FNO}$ − ΔT$_{FV}$|",
            plot_error,
            error_cmap,
            error_norm,
        ),
    )
    meshes = []
    for column, (title, field, cmap, norm) in enumerate(panels):
        for row in range(field_rows):
            ax = axes[row][column]
            mesh = ax.pcolormesh(
                plot_y_faces,
                plot_x_faces,
                field,
                cmap=cmap,
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            if row == 0:
                meshes.append(mesh)
                ax.set_title(title, pad=6)
            ax.grid(False)
            ax.invert_yaxis()
            for depth in (tps_bond, bond_backing):
                ax.axhline(depth, color=INK_PRIMARY, linewidth=0.9, alpha=0.55)
            for marker, key in (("^", "bond_marker"), ("o", "structural_marker")):
                ax.plot(
                    info[key]["y"] * 1e3,
                    info[key]["x"] * 1e3,
                    marker=marker,
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor=INK_PRIMARY,
                    markeredgewidth=1.4,
                    linestyle="none",
                    zorder=4,
                )
            if zoom and row == 1:
                ax.set_ylim(zoom_high, zoom_low)
            if column == 0:
                ax.set_ylabel(
                    "depth x (mm)" if row == 0 else "depth x (mm), bond zoom"
                )
            if row == field_rows - 1:
                ax.set_xlabel("lateral y (mm)")
        if zoom:
            axes[0][column].add_patch(
                Rectangle(
                    (float(y_faces[0]), zoom_low),
                    float(y_faces[-1] - y_faces[0]),
                    zoom_high - zoom_low,
                    fill=False,
                    edgecolor=INK_MUTED,
                    linewidth=0.8,
                    linestyle=(0, (2, 2)),
                    zorder=5,
                )
            )

    delta_axes = [
        axes[row][column]
        for row in range(field_rows)
        for column in (0, 1)
    ]
    bar = fig.colorbar(meshes[0], ax=delta_axes, location="bottom", shrink=0.7, pad=0.04)
    bar.set_label(
        "temperature rise ΔT (K; enhanced color contrast)",
        color=INK_SECONDARY,
    )
    bar.outline.set_edgecolor(AXIS)
    error_axes = [axes[row][2] for row in range(field_rows)]
    error_bar = fig.colorbar(
        meshes[2], ax=error_axes, location="bottom", shrink=0.7, pad=0.04
    )
    error_bar.set_label("absolute error (K)", color=INK_SECONDARY)
    error_bar.outline.set_edgecolor(AXIS)

    if has_property_strip:
        property_axes = axes[field_rows]
        temperature = arrays["f1_property_temperature"]
        realized_low, realized_high = info["realized_tps_temperature_range_K"]
        realized_low = max(realized_low, float(temperature[0]))
        realized_high = min(realized_high, float(temperature[-1]))
        for ax in property_axes:
            ax.axvspan(
                realized_low,
                realized_high,
                color=GRID,
                alpha=0.6,
                linewidth=0.0,
                label="realized TPS-cell range",
                zorder=0,
            )
            ax.set_xlim(float(temperature[0]), float(temperature[-1]))
            ax.set_xlabel("temperature (K)")

        property_axes[0].plot(
            temperature,
            arrays["f1_property_k_x"],
            color=MATERIAL_X_COLOR,
            label=r"through-thickness $k_x$",
        )
        property_axes[0].plot(
            temperature,
            arrays["f1_property_k_y"],
            color=MATERIAL_Y_COLOR,
            label=r"in-plane $k_y$",
        )
        property_axes[0].set_title("directional conductivity", pad=6)
        property_axes[0].set_ylabel("k (W m⁻¹ K⁻¹)")
        property_axes[0].legend(loc="best")

        property_axes[1].plot(
            temperature,
            arrays["f1_property_rho_c"] / 1.0e6,
            color=CAPACITY_COLOR,
            label=r"$\rho c_p$",
        )
        property_axes[1].set_title("volumetric heat capacity", pad=6)
        property_axes[1].set_ylabel(
            r"$\rho c_p$ (MJ m$^{-3}$ K$^{-1}$)"
        )
        property_axes[1].legend(loc="best")

        property_axes[2].plot(
            temperature,
            arrays["f1_property_alpha_x"] * 1.0e6,
            color=MATERIAL_X_COLOR,
            label=r"$\alpha_x = k_x / (\rho c_p)$",
        )
        property_axes[2].plot(
            temperature,
            arrays["f1_property_alpha_y"] * 1.0e6,
            color=MATERIAL_Y_COLOR,
            label=r"$\alpha_y = k_y / (\rho c_p)$",
        )
        property_axes[2].set_title("directional thermal diffusivity", pad=6)
        property_axes[2].set_ylabel("α (mm² s⁻¹)")
        property_axes[2].legend(loc="best")

    handles = [
        plt.Line2D([], [], color=INK_PRIMARY, linewidth=0.9, alpha=0.55,
                   label="TPS–bond and bond–backing interfaces"),
        plt.Line2D([], [], marker="^", linestyle="none", markersize=7,
                   markerfacecolor="none", markeredgecolor=INK_PRIMARY,
                   label="max bond temperature"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                   markerfacecolor="none", markeredgecolor=INK_PRIMARY,
                   label="max structural-interface temperature"),
    ]
    # Title strip on top, then the shared key, then the axes; the footnote gets
    # its own strip at the bottom so nothing lands on the colorbars.
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.962),
    )
    fig.suptitle(
        f"Held-out case {info['case_id']} · d$_{{TPS}}$ = "
        f"{_millimetres(info['d_tps'])} · critical time t = {info['t_critical']:g} s",
        color=INK_PRIMARY,
        fontsize=11,
        x=0.006,
        y=0.986,
        ha="left",
    )
    figure_note = (
        f"Panels 1–2 share one ΔT scale. Field RMSE {info['field_rmse_K']:.2f} K, "
        f"max |error| {info['error_max_K']:.2f} K; bond-peak error "
        f"{info['bond_peak_error_K']:.2f} K, structural-peak error "
        f"{info['structural_peak_error_K']:.2f} K. Case chosen by "
        f"{info['selection_rule']}. Fields are bilinearly interpolated from "
        f"{fv.shape[0]}×{fv.shape[1]} solver cells to "
        f"{plot_fv.shape[0]}×{plot_fv.shape[1]} display cells; interpolation "
        "adds no physical resolution. Temperature color uses a shared "
        f"power-law scale (γ={TEMPERATURE_COLOR_GAMMA:.2f}) to emphasize "
        "low-to-mid-range differences; colorbar values remain ΔT in kelvin."
    )
    physics_note = _physics_note(manifest)
    if physics_note:
        figure_note = f"{figure_note} {physics_note}"
    _footnote(fig, figure_note, height=0.072, top=0.075)
    return _save(fig, output_dir, FIGURE_NAMES[1])


# --------------------------------------------------------------------------- #
# Figure 2 — critical temperature histories
# --------------------------------------------------------------------------- #


def render_figure2(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    _, plt = _pyplot()

    info = manifest["figure2"]
    if info is None:
        raise ValueError("The artifact has no figure 2 section; rebuild it with -f 2.")
    limits = manifest["limits"]
    times = arrays["f2_times"]
    thicknesses = list(info["thicknesses"])
    roles = list(info["roles"])

    has_diagnostics = all(
        key in arrays
        for key in (
            "f2_incident_flux",
            "f2_reradiated_flux",
            "f2_net_flux",
            "f2_solver_times",
            "f2_nonlinear_iterations",
            "f2_nonlinear_residuals",
            "f2_nonlinear_backtracks",
        )
    )
    fig, axes = plt.subplots(
        2 if has_diagnostics else 1,
        2,
        figsize=(10.0, 7.4 if has_diagnostics else 4.2),
        squeeze=False,
    )
    temperature_axes = axes[0]
    panels = (
        (
            temperature_axes[0],
            "maximum bond temperature",
            arrays["f2_bond_fv"],
            arrays["f2_bond_fno"],
            limits["bond_temperature"],
            f"bond limit {limits['bond_temperature']:g} K",
        ),
        (
            temperature_axes[1],
            "maximum structural-interface temperature",
            arrays["f2_struct_fv"],
            arrays["f2_struct_fno"],
            limits["structural_interface_temperature"],
            f"structural limit {limits['structural_interface_temperature']:g} K",
        ),
    )
    for ax, title, fv, fno, limit, label in panels:
        for index, thickness in enumerate(thicknesses):
            color = THICKNESS_COLORS[index % len(THICKNESS_COLORS)]
            emphasis = roles[index] == "selected"
            width = 2.4 if emphasis else 1.6
            ax.plot(times, fv[index], color=color, linewidth=width, zorder=3)
            ax.plot(
                times,
                fno[index],
                color=color,
                linewidth=width,
                linestyle=(0, (5, 2.5)),
                zorder=3,
            )
        _headroom(ax, limit, (fv, fno))
        _limit_line(ax, limit, label)
        ax.set_title(title, pad=6)
        ax.set_xlabel("time t (s)")
        ax.set_xlim(float(times[0]), float(times[-1]))
    temperature_axes[0].set_ylabel("temperature (K)")

    thickness_handles = [
        plt.Line2D(
            [], [],
            color=THICKNESS_COLORS[index % len(THICKNESS_COLORS)],
            linewidth=2.4 if roles[index] == "selected" else 1.6,
            label=f"d$_{{TPS}}$ = {_millimetres(thickness)} ({roles[index]})",
        )
        for index, thickness in enumerate(thicknesses)
    ]
    solver_handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.6, label="FV"),
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.6,
                   linestyle=(0, (5, 2.5)), label="FNO"),
    ]
    temperature_axes[0].legend(handles=thickness_handles, loc="best")
    temperature_axes[1].legend(handles=solver_handles, loc="best")

    diagnostic_note = None
    if has_diagnostics:
        diagnostic_index = int(info["diagnostic_thickness_index"])
        diagnostic_thickness = float(info["diagnostic_thickness"])
        flux_ax = axes[1][0]
        flux_ax.plot(
            times,
            arrays["f2_incident_flux"][diagnostic_index] / 1.0e3,
            color=FNO_COLOR,
            label="incident q″",
        )
        flux_ax.plot(
            times,
            arrays["f2_reradiated_flux"][diagnostic_index] / 1.0e3,
            color=RADIATION_COLOR,
            label="outward re-radiation q″ᵣₐd",
        )
        flux_ax.plot(
            times,
            arrays["f2_net_flux"][diagnostic_index] / 1.0e3,
            color=NET_FLUX_COLOR,
            label="net into TPS q″ − q″ᵣₐd",
        )
        flux_ax.axhline(0.0, color=AXIS, linewidth=0.9)
        flux_ax.set_xlim(float(times[0]), float(times[-1]))
        flux_ax.set_xlabel("time t (s)")
        flux_ax.set_ylabel("hot-face flux (kW m⁻²)")
        flux_ax.set_title(
            "boundary energy balance at the FV hot spot",
            pad=6,
        )
        flux_ax.legend(loc="best")

        solver_ax = axes[1][1]
        solver_times = arrays["f2_solver_times"]
        iteration_counts = arrays["f2_nonlinear_iterations"][diagnostic_index]
        backtracks = arrays["f2_nonlinear_backtracks"][diagnostic_index]
        residuals = arrays["f2_nonlinear_residuals"][diagnostic_index]
        solver_ax.step(
            solver_times,
            iteration_counts,
            where="post",
            color=FV_COLOR,
            linewidth=1.5,
            label="Newton iterations",
        )
        if np.any(backtracks):
            solver_ax.step(
                solver_times,
                backtracks,
                where="post",
                color=INK_MUTED,
                linewidth=1.1,
                label="line-search backtracks",
            )
        solver_ax.set_xlim(float(times[0]), float(times[-1]))
        solver_ax.set_ylim(bottom=0.0)
        solver_ax.set_xlabel("internal timestep t (s)")
        solver_ax.set_ylabel("count per timestep")
        solver_ax.set_title("global nonlinear-solver convergence", pad=6)

        positive = residuals > 0.0
        residual_ax = solver_ax.twinx()
        if np.any(positive):
            residual_ax.semilogy(
                solver_times[positive],
                residuals[positive],
                color=FNO_COLOR,
                linewidth=1.2,
                alpha=0.85,
                label="final temperature residual",
            )
            residual_ax.set_ylabel("final residual norm (K)")
            residual_ax.spines["right"].set_visible(True)
            residual_ax.spines["right"].set_color(AXIS)
        else:
            solver_ax.text(
                0.5,
                0.5,
                "linear step path",
                transform=solver_ax.transAxes,
                ha="center",
                va="center",
                color=INK_MUTED,
            )
        handles, labels = solver_ax.get_legend_handles_labels()
        residual_handles, residual_labels = residual_ax.get_legend_handles_labels()
        solver_ax.legend(
            handles + residual_handles,
            labels + residual_labels,
            loc="best",
        )

        peak = info["peaks"][diagnostic_index]
        diagnostic_note = (
            f"Diagnostic trace: dTPS={_millimetres(diagnostic_thickness)}, "
            f"fixed y*={info['diagnostic_hot_spot_y_m'] * 1e3:.1f} mm; "
            f"{100.0 * peak['reradiated_energy_fraction']:.1f}% of incident "
            f"energy re-radiated, max {peak['max_nonlinear_iterations']} nonlinear "
            f"iterations/step, {peak['total_nonlinear_backtracks']} total backtracks."
        )
    fig.suptitle(
        f"Scenario {info['scenario_id']} · {info['representation']} surrogate",
        color=INK_PRIMARY,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    notes = [
        value
        for value in (
            info.get("reason"),
            diagnostic_note,
            _physics_note(manifest),
        )
        if value
    ]
    _footnote(fig, " ".join(notes) or None, height=0.075 if notes else 0.0)
    return _save(fig, output_dir, FIGURE_NAMES[2])


# --------------------------------------------------------------------------- #
# Figure 3 — thickness / temperature sizing curves
# --------------------------------------------------------------------------- #


def render_figure3(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    _, plt = _pyplot()

    info = manifest["figure3"]
    if info is None:
        raise ValueError("The artifact has no figure 3 section; rebuild it with -f 3.")
    limits = manifest["limits"]
    thickness = arrays["f3_thickness"] * 1e3
    violation = arrays["f3_surrogate_validity_violation"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True)
    panels = (
        (
            axes[0],
            "maximum bond temperature",
            "fv_bond",
            "fno_bond",
            limits["bond_temperature"],
            f"bond limit {limits['bond_temperature']:g} K",
        ),
        (
            axes[1],
            "maximum structural-interface temperature",
            "fv_struct",
            "fno_struct",
            limits["structural_interface_temperature"],
            f"structural limit {limits['structural_interface_temperature']:g} K",
        ),
    )
    for ax, title, fv_key, fno_key, limit, label in panels:
        for key, color, linestyle in (
            (fv_key, FV_COLOR, "solid"),
            (fno_key, FNO_COLOR, (0, (5, 2.5))),
        ):
            ax.fill_between(
                thickness,
                arrays[f"f3_{key}_lo"],
                arrays[f"f3_{key}_hi"],
                color=color,
                alpha=0.13,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                thickness,
                arrays[f"f3_{key}_worst"],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                marker="o",
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                zorder=3,
            )
        if violation.any():
            ax.plot(
                thickness[violation],
                arrays[f"f3_{fno_key}_worst"][violation],
                linestyle="none",
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=FNO_COLOR,
                markeredgewidth=1.6,
                zorder=4,
            )
        _headroom(
            ax,
            limit,
            [arrays[f"f3_{key}_{statistic}"]
             for key in (fv_key, fno_key)
             for statistic in ("lo", "hi")],
        )
        _limit_line(ax, limit, label)
        selections = [
            (value, color, name)
            for value, color, name in (
                (info["fv_selected_thickness"], FV_COLOR, "FV"),
                (info["fno_selected_thickness"], FNO_COLOR, "FNO"),
            )
            if value is not None
        ]
        # A shared selection is the expected outcome; one label, not two on top
        # of each other.
        coincident = len(selections) == 2 and np.isclose(
            selections[0][0], selections[1][0], rtol=0.0, atol=1e-12
        )
        for index, (selected, color, name) in enumerate(selections):
            ax.axvline(
                selected * 1e3,
                color=color,
                linewidth=1.1,
                linestyle=(0, (2, 2)),
                alpha=0.8,
                zorder=2,
            )
            if coincident and index == 1:
                continue
            # Label into the headroom above the limit, not across the curves.
            right_half = selected * 1e3 > 0.6 * float(thickness.max())
            ax.annotate(
                (
                    f"min feasible, FV and FNO\n{_millimetres(selected)}"
                    if coincident
                    else f"min {name}-feasible\n{_millimetres(selected)}"
                ),
                xy=(selected * 1e3, 0.98 - (0.0 if coincident else 0.11 * index)),
                xycoords=("data", "axes fraction"),
                xytext=(-5 if right_half else 5, 0),
                textcoords="offset points",
                ha="right" if right_half else "left",
                va="top",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
        ax.set_title(title, pad=6)
        ax.set_xlabel("TPS thickness d$_{TPS}$ (mm)")
    axes[0].set_ylabel("worst-case peak temperature (K)")

    handles = [
        plt.Line2D([], [], color=FV_COLOR, linewidth=2.2, marker="o",
                   label="FV worst case over scenarios"),
        plt.Line2D([], [], color=FNO_COLOR, linewidth=2.2, marker="o",
                   linestyle=(0, (5, 2.5)), label="FNO worst case over scenarios"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=8, alpha=0.35,
                   label="range across scenarios"),
    ]
    if violation.any():
        handles.append(
            plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
                       markerfacecolor="none", markeredgecolor=FNO_COLOR,
                       markeredgewidth=1.6,
                       label="surrogate outside validity domain")
        )
    axes[0].legend(handles=handles, loc="best")

    missing = [
        name
        for name, value in (
            ("FNO", info["fno_selected_thickness"]),
            ("FV", info["fv_selected_thickness"]),
        )
        if value is None
    ]
    notes = []
    if missing:
        notes.append(
            f"No feasible design in the candidate grid for {', '.join(missing)}; "
            "the corresponding minimum-feasible marker is omitted."
        )
    physics_note = _physics_note(manifest)
    if physics_note:
        notes.append(physics_note)
    fig.suptitle(
        f"Sizing curves · {info['representation']} surrogate versus FV "
        f"({max(info['scenario_count'])} scenarios)",
        color=INK_PRIMARY,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _footnote(fig, " ".join(notes) or None, height=0.065 if notes else 0.0)
    return _save(fig, output_dir, FIGURE_NAMES[3])


# --------------------------------------------------------------------------- #
# Figure 4 — decision-margin ablation
# --------------------------------------------------------------------------- #


def render_figure4(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    _, plt = _pyplot()

    info = manifest["figure4"]
    if info is None:
        raise ValueError("The artifact has no figure 4 section; rebuild it with -f 4.")
    panels = info["panels"]
    if not panels:
        raise ValueError("Figure 4 needs at least one sizing run.")

    values = np.concatenate([
        np.concatenate([
            arrays[f"f4_{panel['representation']}_m_fv"],
            arrays[f"f4_{panel['representation']}_m_fno"],
        ])
        for panel in panels
    ])
    span = float(np.max(values) - np.min(values)) or 1.0
    low = float(np.min(values)) - 0.06 * span
    high = float(np.max(values)) + 0.06 * span
    # The quadrants only exist where the zero lines are actually on screen.
    # Drawing corner labels when every cell sits on one side would assert
    # false-feasible and infeasible regions that contain no data.
    straddles_zero = low < 0.0 < high

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.7 * len(panels) + 0.6, 4.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, panel in enumerate(panels):
        ax = axes[0][index]
        representation = panel["representation"]
        m_fv = arrays[f"f4_{representation}_m_fv"]
        m_fno = arrays[f"f4_{representation}_m_fno"]
        false_feasible = arrays[f"f4_{representation}_false_feasible"]
        false_infeasible = arrays[f"f4_{representation}_false_infeasible"]
        agree = ~(false_feasible | false_infeasible)

        ax.plot([low, high], [low, high], color=INK_MUTED, linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1)
        if straddles_zero:
            ax.axhline(0.0, color=AXIS, linewidth=1.2, zorder=1)
            ax.axvline(0.0, color=AXIS, linewidth=1.2, zorder=1)
        ax.scatter(m_fv[agree], m_fno[agree], s=26, c=AGREEMENT_COLOR,
                   edgecolors=SURFACE, linewidths=0.6, alpha=0.85, zorder=2)
        ax.scatter(m_fv[false_infeasible], m_fno[false_infeasible], s=70, marker="^",
                   c=FALSE_INFEASIBLE_COLOR, edgecolors=SURFACE, linewidths=0.8,
                   zorder=3)
        ax.scatter(m_fv[false_feasible], m_fno[false_feasible], s=80, marker="v",
                   c=FALSE_FEASIBLE_COLOR, edgecolors=SURFACE, linewidths=0.8,
                   zorder=4)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, axis="y")
        ax.set_title(
            f"{representation}\nfalse feasible {panel['false_feasible']} · "
            f"false infeasible {panel['false_infeasible']} · "
            f"margin MAE {panel['margin_mae_K']:.2f} K",
            pad=6,
            fontsize=9,
        )
        ax.set_xlabel("FV controlling margin m$_{FV}$ (K)")
        if index == 0:
            ax.set_ylabel("predicted controlling margin m$_{FNO}$ (K)")
            if straddles_zero:
                for x, y, ha, va, text in (
                    (0.97, 0.97, "right", "top", "correctly feasible"),
                    (0.03, 0.03, "left", "bottom", "correctly infeasible"),
                    (0.03, 0.97, "left", "top", "false feasible"),
                    (0.97, 0.03, "right", "bottom", "false infeasible"),
                ):
                    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                            fontsize=7.5, color=INK_MUTED)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=AGREEMENT_COLOR, markeredgecolor=SURFACE,
                   label="sign agrees with FV"),
        plt.Line2D([], [], marker="v", linestyle="none", markersize=8,
                   markerfacecolor=FALSE_FEASIBLE_COLOR, markeredgecolor=SURFACE,
                   label="false feasible (unsafe)"),
        plt.Line2D([], [], marker="^", linestyle="none", markersize=8,
                   markerfacecolor=FALSE_INFEASIBLE_COLOR, markeredgecolor=SURFACE,
                   label="false infeasible (conservative)"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                   label="m$_{FNO}$ = m$_{FV}$"),
    ]
    axes[0][-1].legend(handles=handles, loc="lower right")
    fig.suptitle(
        "Sizing-margin agreement by forcing representation",
        color=INK_PRIMARY,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    notes = []
    if not straddles_zero:
        # Decide from the margins themselves, not the padded axis limits.
        side = "feasible" if float(np.min(values)) >= 0.0 else "infeasible"
        notes.append(
            f"Every cell is {side} under both solvers, so the feasibility "
            "boundary m = 0 lies outside the plotted range and the quadrant "
            "lines are omitted."
        )
    if info["missing_representations"]:
        notes.append(
            "Missing representation(s): "
            f"{', '.join(info['missing_representations'])}; the ablation is incomplete."
        )
    physics_note = _physics_note(manifest)
    if physics_note:
        notes.append(physics_note)
    _footnote(fig, " ".join(notes) or None, height=0.065 if notes else 0.0)
    return _save(fig, output_dir, FIGURE_NAMES[4])


_RENDERERS = {
    1: render_figure1,
    2: render_figure2,
    3: render_figure3,
    4: render_figure4,
}


def render_figures(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
    figures: Iterable[int] | None = None,
) -> dict[int, list[str]]:
    requested = (
        list(manifest["generated_figures"])
        if figures is None
        else sorted({int(value) for value in figures})
    )
    written: dict[int, list[str]] = {}
    for number in requested:
        if number not in _RENDERERS:
            raise ValueError(f"Unknown figure number {number}; expected 1, 2, 3, or 4.")
        written[number] = [str(path) for path in _RENDERERS[number](
            manifest, arrays, output_dir
        )]
    return written


def render_from_directory(
    directory: str | Path,
    output_dir: str | Path | None = None,
    figures: Iterable[int] | None = None,
) -> dict[int, list[str]]:
    # Imported lazily so that rendering a manifest already in hand never drags in
    # the solver and torch stack behind `figure_data`.
    from fno_tps.figure_data import load_figure_data

    manifest, arrays = load_figure_data(directory)
    return render_figures(
        manifest,
        arrays,
        directory if output_dir is None else output_dir,
        figures,
    )
