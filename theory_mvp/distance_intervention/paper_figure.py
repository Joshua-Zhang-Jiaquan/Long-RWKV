"""Reflow the frozen figure for paper-width legibility; data stay unchanged."""
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import figures


def coordinates(fig):
    return [
        dict(lines=[line.get_xydata().tolist() for line in ax.lines],
             bands=[[p.vertices.tolist() for p in collection.get_paths()] for collection in ax.collections])
        for ax in fig.axes
    ]


def main():
    figures.main()
    fig = plt.gcf()
    before = coordinates(fig)
    fig.set_size_inches(7.2, 5.3)
    titles = ['Joint KL\n(nats)', 'Second-pass\nconditional error',
              'Counterfactual\ntarget CE', 'Signed evidence\nlogit contrast',
              'Counterfactual\ncentering penalty', 'Unaffected-target\nprobability change']
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=10)
    for legend in list(fig.legends):
        legend.remove()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('16K evidence-position intervention\nThree adaptation seeds; one selected starting checkpoint', fontsize=10)
    after = coordinates(fig)
    if before != after:
        raise ValueError('Publication reflow changed plotted values or uncertainty bands')
    folder = figures.FOLDER
    fig.savefig(folder / 'mechanism_distance_paper.pdf', bbox_inches='tight')
    fig.savefig(folder / 'mechanism_distance_paper.png', dpi=180, bbox_inches='tight')
    receipt = dict(
        purpose='Paper-width typography only; frozen renderer and all analysis remain unchanged.',
        plotted_coordinates_unchanged=True,
        coordinates_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        frozen_renderer_sha256=hashlib.sha256(Path(figures.__file__).read_bytes()).hexdigest())
    (folder / 'publication_figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
