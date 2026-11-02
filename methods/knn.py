# knn method implementation

import csv
import heapq
import math
import os
import re
from collections import Counter

# csv files are in the dataset folder next to the method folder
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


def clean(text):
    text = re.sub(r"https?://\S+", " ", text.lower())
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]


def load_split(filename):
  texts = []
  labels = []
  filename = os.path.join(DATASET_DIR, filename)
  # newline="" keeps the carriage returns inside tweets from splitting records
  with open(filename, encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
      # skip broken fragments that have no valid label
      if row.get("label") in ("0", "1"):
        texts.append(row["text"])
        labels.append(int(row["label"]))

  return texts, labels

# normalizes a vector to unit length using L2 norm, returns an empty vector if the input is all zeros
def l2_normalize(vector):
  norm = math.sqrt(sum(value * value for value in vector.values()))
  if norm == 0:
    return {}

  return {token: value / norm for token, value in vector.items()}


def build_idf(token_lists):
  # document frequency is computed from the training set only
  df = Counter()
  for tokens in token_lists:
    df.update(set(tokens))

  n = len(token_lists)
  return {token: math.log((1 + n) / (1 + count)) + 1 for token, count in df.items()}


def vectorize(tokens, idf):
  # tf-idf then l2 normalization so that tweet length does not matter
  counts = Counter(token for token in tokens if token in idf)
  vector = {token: count * idf[token] for token, count in counts.items()}

  return l2_normalize(vector)


def build_index(train_vectors):
  # inverted index maps each token to the training documents that contain it
  index = {}
  for doc_id, vector in enumerate(train_vectors):
    for token, weight in vector.items():
      index.setdefault(token, []).append((doc_id, weight))

  return index


def nearest_labels(x, index, train_labels, k_max):
  # cosine similarity of l2 normalized vectors is just the dot product
  scores = [0.0] * len(train_labels)
  for token, weight in x.items():
    for doc_id, train_weight in index.get(token, []):
      scores[doc_id] += weight * train_weight

  # larger similarity means a smaller cosine distance
  nearest = heapq.nlargest(k_max, range(len(scores)), key=scores.__getitem__)

  return [train_labels[doc_id] for doc_id in nearest]


def vote(neighbor_labels, k):
  hate_count = 0
  non_hate_count = 0

  # iterate through each neighbor and check the label
  for label in neighbor_labels[:k]:
    if label == 1:
      hate_count += 1
    else:
      non_hate_count += 1

  # return majority vote
  if hate_count > non_hate_count:
    return 1
  else:
    return 0


def get_neighbors(vectors, index, train_labels, k_max, default_label):
  all_neighbors = []
  for x in vectors:
    # tweets with no known words fall back to the majority class
    if not x:
      all_neighbors.append([default_label] * k_max)
    else:
      all_neighbors.append(nearest_labels(x, index, train_labels, k_max))

  return all_neighbors


def compute_metrics(predicted_labels, actual_labels):
  tp = tn = fp = fn = 0
  for predicted, actual in zip(predicted_labels, actual_labels):
    if predicted == 1 and actual == 1:
      tp += 1
    elif predicted == 0 and actual == 0:
      tn += 1
    elif predicted == 1 and actual == 0:
      fp += 1
    else:
      fn += 1

  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp) if tp + fp > 0 else 0.0
  recall = tp / (tp + fn) if tp + fn > 0 else 0.0
  if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
  else:
    f1 = 0.0

  return {
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "accuracy": accuracy * 100,
    "precision": precision * 100,
    "recall": recall * 100,
    "f1": f1 * 100,
  }


def print_results(name, k, metrics):
  print(f"{name.upper()} SET RESULTS (k = {k})")
  print(f"Confusion matrix: TP = {metrics['tp']}  TN = {metrics['tn']}  "
        f"FP = {metrics['fp']}  FN = {metrics['fn']}")
  print(f"{'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1-Score':>10}")
  print(f"{'k-Nearest Neighbors':<24}{metrics['accuracy']:>9.2f}%"
        f"{metrics['precision']:>10.2f}%{metrics['recall']:>8.2f}%"
        f"{metrics['f1']:>9.2f}%")
  print()


# main program flow

def main():
  train_texts, train_labels = load_split("train.csv")
  valid_texts, valid_labels = load_split("valid.csv")
  test_texts, test_labels = load_split("test.csv")

  print(f"Number of training records: {len(train_labels)}")
  print(f"Number of validation records: {len(valid_labels)}")
  print(f"Number of testing records: {len(test_labels)}\n")

  # build the vocabulary and idf weights from training data only
  train_tokens = [clean(text) for text in train_texts]
  idf = build_idf(train_tokens)
  train_vectors = [vectorize(tokens, idf) for tokens in train_tokens]
  valid_vectors = [vectorize(clean(text), idf) for text in valid_texts]
  test_vectors = [vectorize(clean(text), idf) for text in test_texts]
  index = build_index(train_vectors)
  default_label = Counter(train_labels).most_common(1)[0][0]

  print("Finding nearest neighbors...\n")
  k_max = max(K_CANDIDATES)
  valid_neighbors = get_neighbors(valid_vectors, index, train_labels, k_max, default_label)
  test_neighbors = get_neighbors(test_vectors, index, train_labels, k_max, default_label)

  # choose k using the validation set so the test set stays untouched
  best_k = K_CANDIDATES[0]
  best_f1 = -1.0
  print("Tuning k on the validation set")
  for k in K_CANDIDATES:
    predicted = [vote(neighbors, k) for neighbors in valid_neighbors]
    f1 = compute_metrics(predicted, valid_labels)["f1"]
    print(f"k = {k:<3} F1-Score = {f1:.2f}%")
    if f1 > best_f1:
      best_f1 = f1
      best_k = k
  print(f"\nSelected k = {best_k}\n")

  valid_predicted = [vote(neighbors, best_k) for neighbors in valid_neighbors]
  test_predicted = [vote(neighbors, best_k) for neighbors in test_neighbors]

  print_results("Validation", best_k, compute_metrics(valid_predicted, valid_labels))
  print_results("Test", best_k, compute_metrics(test_predicted, test_labels))


if __name__ == "__main__":
  main()