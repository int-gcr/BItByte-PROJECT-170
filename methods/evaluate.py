# runs the three methods and returns the numbers they report.

import knn
import naive_bayes
import perceptron

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

FIGURES = "results/figures"
METHODS = ["Naive Bayes", "k-NN", "Perceptron"]
LINE_COLOR = "blue"
SELECTED_COLOR = "red"

def run_methods():
  # each method's main() prints its usual output and returns its numbers
  return {"Naive Bayes": naive_bayes.main(), "k-NN": knn.main(), "Perceptron": perceptron.main()}



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
  best = ks.index(results["k-NN"]["selected_k"])  # position of the selected k in the list

  fig, ax = plt.subplots(figsize=(7, 4))
  positions = range(len(ks))  # the k values are unevenly spaced (1, 3, 5, ... 41), so one tick per candidate
  ax.plot(positions, f1s, marker="o", color=LINE_COLOR)
  ax.plot(best, f1s[best], marker="o", markersize=11, markerfacecolor="none", color=SELECTED_COLOR)  # circle around the selected k
  ax.annotate(f"selected k = {ks[best]} (F1 {f1s[best]:.2f}%)", (best, f1s[best]),
              textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)

  tick_labels = []
  for k in ks:
    tick_labels.append(str(k))

  ax.set_ylim(min(f1s) - 0.4, max(f1s) + 0.9)  # headroom so the label does not touch the title
  ax.set_xticks(positions, tick_labels)
  ax.set_xlabel("k (number of neighbours)")
  ax.set_ylabel("Validation F1-Score (%)")
  ax.set_title("k-NN: choosing k on the validation set (y-axis does not start at 0)", loc="left", fontsize=10, fontweight="bold")
  clean_axes(ax)
  save(fig, "knn_k_tuning.png")


def draw_epochs(results):
  epoch_numbers, updates = split_pairs(results["Perceptron"]["epochs"])

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(epoch_numbers, updates, color=LINE_COLOR)
  ax.set_ylim(0, max(updates) * 1.05)
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Training updates (mistakes) in the epoch")
  ax.set_title("Perceptron: updates per epoch (0 updates would mean convergence)", loc="left", fontsize=10, fontweight="bold")
  clean_axes(ax)
  save(fig, "perceptron_epochs.png")


def draw_graphs(results):
  draw_k_tuning(results)
  draw_epochs(results)


def main():
  run_methods()


if __name__ == "__main__":
  main()
