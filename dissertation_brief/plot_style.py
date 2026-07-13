import matplotlib.pyplot as plt

def set_paper_style(use_latex=True):
    plt.rcParams.update({
        # LaTeX-style fonts
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],

        # Font sizes
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Figure style
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # Axes
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Export text properly in PDF/SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# Automatically apply the style when the file is imported
set_paper_style(use_latex=True)

# Figure dimensions
FIG_SINGLE_SQ = (6, 6)
FIG_DOUBLE_SQ = (11, 6)
FIG_QUAD_SQ = (11, 11)
FIG_SINGLE_WD = (6.5, 4)
FIG_DOUBLE_WD = (13, 4)
FIG_QUAD_WD = (13, 8)

FIG_SMALL = (5.5, 3.5)
