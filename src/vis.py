import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(data: pd.DataFrame):
    g = sns.displot(
        data=data,
        x="norm",
        col="model",
        row="layer",
        kind="hist",
        height=2.5,
        aspect=1,
        stat="probability",
        bins=20,
        common_norm=False,
        common_bins=False,
        edgecolor="white",
        linewidth=0.5,
        kde=True,
        facet_kws={"sharex": False, "sharey": False},
    )

    for ax in g.axes.flatten():
        if not ax.patches and not ax.lines:
            ax.set_visible(False)
        else:
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            ax.tick_params(labelbottom=True, labelleft=True)

    g.figure.subplots_adjust(hspace=0.5, wspace=0.8)
    plt.savefig("./figs/plot.png", dpi=300)
