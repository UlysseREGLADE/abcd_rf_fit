import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .utils import dB, deg, get_prefix

# from .resonators import resonator_dict

cm = 1 / 2.54  # centimeters in inches
column_width = 12.4 * cm


def get_ax_ratio(ax):
    bbox = ax.get_window_extent().transformed(
        ax.get_figure().dpi_scale_trans.inverted()
    )
    screen_width, screen_height = bbox.width, bbox.height

    x_lim = np.abs(np.diff(ax.get_xlim())[0])
    y_lim = np.abs(np.diff(ax.get_ylim())[0])

    return (screen_height / screen_width) * (x_lim / y_lim)


def grid_spec_inches(
    fig,
    width_ratios=(1,),
    height_ratios=(1,),
    left=1.2 * cm,
    right=1.2 * cm,
    bottom=1.2 * cm,
    top=0.0,
    wspace=0.2 * cm,
    hspace=0.2 * cm,
):

    if type(width_ratios) in (int, float):
        width_ratios = [width_ratios]
    if type(height_ratios) in (int, float):
        height_ratios = [height_ratios]

    fig_width = fig.get_size_inches()[0]
    fig_height = fig.get_size_inches()[1]

    avg_width = (fig_width - right - left - (len(width_ratios) - 1) * wspace) / len(
        width_ratios
    )
    wspace = wspace / avg_width
    left = left / fig_width
    right = 1 - (right / fig_width)

    avg_height = (fig_height - top - bottom - (len(height_ratios) - 1) * hspace) / len(
        height_ratios
    )
    hspace = hspace / avg_height
    bottom = bottom / fig_height
    top = 1 - (top / fig_height)

    return fig.add_gridspec(
        len(height_ratios),
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )


def format_fig(fig, ignored_axes=None, cbar_axes=None):
    if ignored_axes is None:
        ignored_axes = []
    if cbar_axes is None:
        cbar_axes = []

    for ax in fig.axes:
        if ax not in ignored_axes:
            if ax not in cbar_axes:
                ax.tick_params(
                    direction="in",
                    which="both",
                    color=[0, 0, 0, 0.5],
                    colors=[0, 0, 0, 0.75],
                )
                ax.yaxis.set_ticks_position("both")
                ax.xaxis.set_ticks_position("both")
            else:
                ax.tick_params(direction="in", color=[0, 0, 0, 0])

            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_alpha(0.25)


def plot(
    freq,
    signal,
    fit=None,
    fig=None,
    params=None,
    fit_params=None,
    plot_not_corrected=True,
    font_size=None,
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

    # y_axis_str = r'S_{11}'
    # if fit_params is not None:
    #     if fit_params.resonator_func == resonator_dict['t']:
    #         y_axis_str = r'S_{21}'

    y_axis_str = r"S"

    if center_freq:
        freq = freq - fit_params.f_0

    if style == "Normal":
        size = 10
        zorder = 1
        # facecolors = 'C0'
        facecolors = "none"
    if style == "Leghtas":
        size = 10
        zorder = -10
        facecolors = "none"

    if font_size is not None:
        matplotlib.rcParams.update({"font.size": font_size})

    freq_disp, freq_prefix = get_prefix(freq)
    if params is not None:
        # params_label = params.str(latex=True, separator="\n", precision=precision, only_f_and_kappa=only_f_and_kappa)
        params_label = params.str(
            latex=True,
            separator=", ",
            precision=precision,
            only_f_and_kappa=only_f_and_kappa,
            red_warning=True,
        )
    else:
        params_label = None
    if fit_params is not None:
        # fit_params_label = fit_params.str(latex=True, separator="\n", precision=precision, only_f_and_kappa=only_f_and_kappa)
        fit_params_label = fit_params.str(
            latex=True,
            separator=", ",
            precision=precision,
            only_f_and_kappa=only_f_and_kappa,
            red_warning=True,
        )
    else:
        fit_params_label = None

    # fig = fig or plt.figure(figsize=(18, 6))
    fig = fig or plt.figure(figsize=(21 * cm, 8.5 * cm))

    # grid = GridSpec(2, 2, fig, wspace=0.2, hspace=0.3, width_ratios=[1.5, 1], left=0.3)
    width_ratios = (2.1, 1)
    grid = grid_spec_inches(
        fig,
        width_ratios=width_ratios,
        height_ratios=(1, 1),
        left=2.1 * cm,
        right=2.1 * cm,
        top=1.8 * cm,
        bottom=1.1 * cm,
    )

    if plot_circle:
        circle_ax = fig.add_subplot(grid[:, 1])

        if corrected_signal is None:
            # ax.plot(np.real(signal), np.imag(signal), ".C0")
            circle_ax.scatter(
                np.real(signal),
                np.imag(signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=alpha_fit,
            )
            if fit is not None:
                circle_ax.plot(np.real(fit), np.imag(fit), "-C1", zorder=zorder)
        else:
            if plot_not_corrected:
                # circle_ax.plot(np.real(signal), np.imag(signal), ".C0", alpha=0.15)
                circle_ax.scatter(
                    np.real(signal),
                    np.imag(signal),
                    s=size,
                    facecolors=facecolors,
                    edgecolors="C0",
                    alpha=0.15 * alpha_fit,
                )
            # circle_ax.plot(np.real(corrected_signal), np.imag(corrected_signal), ".C0")
            circle_ax.scatter(
                np.real(corrected_signal),
                np.imag(corrected_signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=alpha_fit,
            )
            if fit is not None:
                if plot_not_corrected:
                    circle_ax.plot(
                        np.real(fit), np.imag(fit), "-C1", alpha=0.15, zorder=zorder
                    )
                circle_ax.plot(
                    np.real(corrected_fit), np.imag(corrected_fit), "-C1", zorder=zorder
                )

        circle_ax.plot(0, 0, "+C3")
        circle_ax.set_xlabel("I")
        circle_ax.yaxis.set_label_position("right")
        circle_ax.yaxis.set_ticks_position("right")
        circle_ax.set_ylabel("Q")
        ratio = get_ax_ratio(circle_ax)
        if ratio > 1:
            ylim = circle_ax.get_ylim()
            center = 0.5 * (ylim[1] + ylim[0])
            delta = 0.5 * (ylim[1] - ylim[0])
            delta *= ratio
            circle_ax.set_ylim(center - delta, center + delta)
        else:
            xlim = circle_ax.get_xlim()
            center = 0.5 * (xlim[1] + xlim[0])
            delta = 0.5 * (xlim[1] - xlim[0])
            delta /= ratio
            circle_ax.set_xlim(center - delta, center + delta)

        # circle_ax.set_aspect("equal")
        circle_ax.grid(alpha=0.3)

    if plot_circle:
        mag_ax = fig.add_subplot(grid[0, 0])
    else:
        mag_ax = fig.add_subplot(grid[0, :])

    if title is not None:
        mag_ax.set_title(title)

    # mag_ax.plot(freq_disp, dB(signal), ".C0")
    mag_ax.scatter(
        freq_disp,
        dB(signal),
        s=size,
        facecolors=facecolors,
        edgecolors="C0",
        alpha=alpha_fit,
        label=params_label,
    )
    if fit is not None:
        mag_ax.plot(freq_disp, dB(fit), "-C1", label=fit_params_label, zorder=zorder)

    if fit_params_label is not None:
        mag_ax.legend(
            bbox_to_anchor=(0.5 * (1 + width_ratios[1] / width_ratios[0]), 1),
            loc="lower center",
        )

    mag_ax.grid(alpha=0.3)
    mag_ax.set_ylabel(r"$|%s|$ [dB]" % y_axis_str)
    mag_ax.xaxis.set_ticklabels([])

    if plot_circle:
        arg_ax = fig.add_subplot(grid[1, 0])
    else:
        arg_ax = fig.add_subplot(grid[1, :])

    if corrected_signal is None:
        # arg_ax.plot(freq_disp, deg(signal), ".C0", label=params_label)
        arg_ax.scatter(
            freq_disp,
            deg(signal),
            s=size,
            facecolors=facecolors,
            edgecolors="C0",
            alpha=alpha_fit,
        )  # , label=params_label)
        if fit is not None:
            arg_ax.plot(freq_disp, deg(fit), "-C1", zorder=zorder)
    else:
        if plot_not_corrected:
            # arg_ax.plot(freq_disp, deg(signal), ".C0", alpha=0.15)
            arg_ax.scatter(
                freq_disp,
                deg(signal),
                s=size,
                facecolors=facecolors,
                edgecolors="C0",
                alpha=0.15 * alpha_fit,
            )
        # arg_ax.plot(freq_disp, deg(corrected_signal), ".C0", label=params_label)
        arg_ax.scatter(
            freq_disp,
            deg(corrected_signal),
            s=size,
            facecolors=facecolors,
            edgecolors="C0",
            alpha=alpha_fit,
        )  # , label=params_label)
        if fit is not None:
            if plot_not_corrected:
                arg_ax.plot(freq_disp, deg(fit), "-C1", alpha=0.15, zorder=zorder)
            arg_ax.plot(freq_disp, deg(corrected_fit), "-C1", zorder=zorder)

        angle_min, angle_max = (
            np.min(deg(corrected_signal)),
            np.max(deg(corrected_signal)),
        )
        angle_center, angle_span = (
            0.5 * (angle_min + angle_max),
            0.5 * (angle_max - angle_min),
        )
        arg_ax.set_ylim(
            angle_center - 1.1 * angle_span,
            angle_center + 1.1 * angle_span,
        )

    # if params_label is not None:
    #     arg_ax.legend(bbox_to_anchor=(0, 0), loc='upper left')
    arg_ax.grid(alpha=0.3)
    arg_ax.set_ylabel(r"$\arg(%s)$ [deg]" % y_axis_str)
    arg_ax.set_xlabel("f [%sHz]" % freq_prefix)

    fig.align_ylabels([mag_ax, arg_ax])
    if plot_circle:
        fig.align_xlabels([arg_ax, circle_ax])

    format_fig(fig)
