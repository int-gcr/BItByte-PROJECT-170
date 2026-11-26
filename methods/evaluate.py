
# runs the three methods and returns the numbers they report.

import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import knn
import naive_bayes
import perceptron


FIGURES = "methods/results/figures"
METHODS = ["Naive Bayes", "k-NN", "Perceptron"]
LINE_COLOR = "blue"
SELECTED_COLOR = "red"


def run_methods():
    # Each method's main() prints its usual output and returns its numbers.
    return {
        "Naive Bayes": naive_bayes.main(),
        "k-NN": knn.main(),
        "Perceptron": perceptron.main()
    }


# ---------- graphs ----------

def split_pairs(pairs):
    # [(1, 10), (2, 20)] becomes [1, 2] and [10, 20]
    firsts = []
    seconds = []

    for first, second in pairs:
        firsts.append(first)
        seconds.append(second)

    return firsts, seconds


def clean_axes(ax):
    ax.yaxis.grid(True, color="lightgray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, file_name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES + "/" + file_name, dpi=200)
    plt.close(fig)


def draw_k_tuning(results):
    ks, f1s = split_pairs(results["k-NN"]["k_tuning"])
    best = ks.index(results["k-NN"]["selected_k"])

    fig, ax = plt.subplots(figsize=(7, 4))

    positions = range(len(ks))

    ax.plot(positions, f1s, marker="o", color=LINE_COLOR)

    ax.plot(
        best,
        f1s[best],
        marker="o",
        markersize=11,
        markerfacecolor="none",
        color=SELECTED_COLOR
    )

    ax.annotate(
        f"selected k = {ks[best]} (F1 {f1s[best]:.2f}%)",
        (best, f1s[best]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9
    )

    tick_labels = []

    for k in ks:
        tick_labels.append(str(k))

    ax.set_ylim(min(f1s) - 0.4, max(f1s) + 0.9)
    ax.set_xticks(positions, tick_labels)
    ax.tick_params(axis="x", labelsize=5)  # Smaller bottom labels
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Validation F1-Score (%)")
    ax.set_title(
        "k-NN: choosing k on the validation set "
        "(y-axis does not start at 0)",
        loc="left",
        fontsize=10,
        fontweight="bold"
    )

    clean_axes(ax)
    save(fig, "knn_k_tuning.png")


def draw_epochs(results):
    epoch_numbers, updates = split_pairs(
        results["Perceptron"]["epochs"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epoch_numbers, updates, color=LINE_COLOR)
    ax.set_ylim(0, max(updates) * 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training updates (mistakes) in the epoch")
    ax.set_title(
        "Perceptron: updates per epoch "
        "(0 updates would mean convergence)",
        loc="left",
        fontsize=10,
        fontweight="bold"
    )

    clean_axes(ax)
    save(fig, "perceptron_epochs.png")


def draw_perceptron_errors(results):
    history = results["Perceptron"]["history"]

    epoch_numbers = [item["epoch"] for item in history]
    errors = [item["errors"] for item in history]
    false_positives = [item["false_positives"] for item in history]
    false_negatives = [item["false_negatives"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epoch_numbers, errors, label="Total errors")
    ax.plot(epoch_numbers, false_positives, label="False positives")
    ax.plot(epoch_numbers, false_negatives, label="False negatives")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassified training tweets")
    ax.set_title(
        "Perceptron: remaining training errors after each epoch",
        loc="left",
        fontsize=10,
        fontweight="bold"
    )

    ax.legend()

    clean_axes(ax)
    save(fig, "perceptron_error_breakdown.png")


def draw_metric_bars(results, split_name, title, file_name):
    # One group of four bars per method using the numbers
    # printed by print_metrics.
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    metric_colors = [
        "#3366cc",
        "#dc3912",
        "#ff9900",
        "#109618"
    ]

    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4))

    for m, name in enumerate(metric_names):
        positions = []
        values = []

        for i, method in enumerate(METHODS):
            positions.append(i + (m - 1.5) * bar_width)
            values.append(results[method][split_name]["printed"][m])

        ax.bar(
            positions,
            values,
            bar_width,
            label=name,
            color=metric_colors[m]
        )

    ax.set_ylim(55, 85)
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        title + " (y-axis does not start at 0)",
        loc="left",
        fontsize=10,
        fontweight="bold"
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False
    )

    clean_axes(ax)
    save(fig, file_name)


def draw_graphs(results):
    # k-NN graph
    draw_k_tuning(results)

    # Perceptron training graphs
    draw_epochs(results)
    draw_perceptron_errors(results)

    # Model comparison graphs
    draw_metric_bars(
        results,
        "VALIDATION",
        "Validation set comparison",
        "validbar-graph.png"
    )

    draw_metric_bars(
        results,
        "TEST",
        "Test set comparison",
        "test_bar.png"
    )


def main():
    results = run_methods()
    draw_graphs(results)


if __name__ == "__main__":
    main()
