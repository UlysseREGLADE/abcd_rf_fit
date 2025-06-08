import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .resonators import resonator_dict
from .utils import dB, deg, get_prefix


def plot(
    freq,
    signal,
    fit=None,
    fig=None,
    params=None,
    fit_params=None,
    plot_not_corrected=True,
    font_size=15,
    plot_circle=True,
    center_freq=False,
    only_f_and_kappa=False,
    precision=2,
    alpha_fit=1.0,
    style="Normal",
    title=None,
):
    if fit_params is not None and fit_params.edelay is not None:
        corrected_signal = signal * np.exp(-2j * np.pi * freq * fit_params.edelay)
        if fit is not None:
            corrected_fit = fit * np.exp(-2j * np.pi * freq * fit_params.edelay)
    else:
        corrected_signal = None

    y_axis_str = r"S_{11}"
    if fit_params is not None and fit_params.resonator_func == resonator_dict["t"]:
        y_axis_str = r"S_{21}"

    if center_freq:
        freq = freq - fit_params.f_0

    if style == "Normal":
        size = None
        zorder = 1
        facecolors = "C0"
    if style == "Leghtas":
        size = 10
        zorder = -10
        facecolors = "none"

    mpl.rcParams.update({"font.size": font_size})

    freq_disp, freq_prefix = get_prefix(freq)
    if params is not None:
        params_label = params.str(
            latex=True,
            separator="\n",
            precision=precision,
            only_f_and_kappa=only_f_and_kappa,
        )
    else:
        params_label = None
    if fit_params is not None:
        fit_params_label = fit_params.str(
            latex=True,
            separator="\n",
            precision=precision,
            only_f_and_kappa=only_f_and_kappa,
        )
    else:
        fit_params_label = None

    fig = fig or plt.figure(figsize=(18, 6))

    grid = GridSpec(2, 2, fig, wspace=0.2, hspace=0.3, width_ratios=[1.5, 1], left=0.3)

    if plot_circle:
        ax = fig.add_subplot(grid[:, 1])

        if corrected_signal is None:
            # ax.plot(np.real(signal), np.imag(signal), ".C0")
            ax.scatter(
                np.real(signal),
                np.imag(signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=alpha_fit,
            )
            if fit is not None:
                ax.plot(np.real(fit), np.imag(fit), "-C1", zorder=zorder)
        else:
            if plot_not_corrected:
                # ax.plot(np.real(signal), np.imag(signal), ".C0", alpha=0.15)
                ax.scatter(
                    np.real(signal),
                    np.imag(signal),
                    s=size,
                    facecolors=facecolors,
                    edgecolors="C0",
                    alpha=0.15 * alpha_fit,
                )
            # ax.plot(np.real(corrected_signal), np.imag(corrected_signal), ".C0")
            ax.scatter(
                np.real(corrected_signal),
                np.imag(corrected_signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=alpha_fit,
            )
            if fit is not None:
                if plot_not_corrected:
                    ax.plot(
                        np.real(fit), np.imag(fit), "-C1", alpha=0.15, zorder=zorder
                    )
                ax.plot(
                    np.real(corrected_fit), np.imag(corrected_fit), "-C1", zorder=zorder
                )

        ax.plot(0, 0, "+C3")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")

        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[0, 0]) if plot_circle else fig.add_subplot(grid[0, :])

    if title is not None:
        ax.set_title(title)

    # ax.plot(freq_disp, dB(signal), ".C0")
    ax.scatter(
        freq_disp,
        dB(signal),
        s=size,
        facecolors=facecolors,
        edgecolors="C0",
        alpha=alpha_fit,
    )
    if fit is not None:
        ax.plot(freq_disp, dB(fit), "-C1", label=fit_params_label, zorder=zorder)

    if fit_params_label is not None:
        ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylabel(rf"$|{y_axis_str}|$ [dB]")

    ax = fig.add_subplot(grid[1, 0]) if plot_circle else fig.add_subplot(grid[1, :])

    if corrected_signal is None:
        # ax.plot(freq_disp, deg(signal), ".C0", label=params_label)
        ax.scatter(
            freq_disp,
            deg(signal),
            s=size,
            facecolors=facecolors,
            edgecolors="C0",
            alpha=alpha_fit,
            label=params_label,
        )
        if fit is not None:
            ax.plot(freq_disp, deg(fit), "-C1", zorder=zorder)
    else:
        if plot_not_corrected:
            # ax.plot(freq_disp, deg(signal), ".C0", alpha=0.15)
            ax.scatter(
                freq_disp,
                deg(signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=0.15 * alpha_fit,
            )
        # ax.plot(freq_disp, deg(corrected_signal), ".C0", label=params_label)
        ax.scatter(
            freq_disp,
            deg(corrected_signal),
            s=size,
            facecolors=facecolors,
            edgecolors="C0",
            alpha=alpha_fit,
            label=params_label,
        )
        if fit is not None:
            if plot_not_corrected:
                ax.plot(freq_disp, deg(fit), "-C1", alpha=0.15, zorder=zorder)
            ax.plot(freq_disp, deg(corrected_fit), "-C1", zorder=zorder)

        angle_min, angle_max = (
            np.min(deg(corrected_signal)),
            np.max(deg(corrected_signal)),
        )
        angle_center, angle_span = (
            0.5 * (angle_min + angle_max),
            0.5 * (angle_max - angle_min),
        )
        ax.set_ylim(
            angle_center - 1.1 * angle_span,
            angle_center + 1.1 * angle_span,
        )

    if params_label is not None:
        ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylabel(rf"$\arg({y_axis_str})$ [deg]")
    ax.set_xlabel(f"f [{freq_prefix}Hz]")
