# knn method implementation

import heapq
from collections import Counter
from preprocess import prepare_data, prepare_tfidf, confusion_matrix, print_metrics

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]


def build_index(train_vectors):
  # inverted index maps each word to the training tweets that contain it
  index = {}
  for doc_id, vector in enumerate(train_vectors):
    for word, weight in vector.items():
      index.setdefault(word, []).append((doc_id, weight))

  return index


def nearest_labels(x, index, train_labels, k_max):
  # cosine similarity of l2 normalized vectors is just the dot product
  scores = [0.0] * len(train_labels)
  for word, weight in x.items():
    for doc_id, train_weight in index.get(word, []):
      scores[doc_id] += weight * train_weight

  # larger similarity means a smaller cosine distance
  nearest = heapq.nlargest(k_max, range(len(scores)), key=scores.__getitem__)

  return [train_labels[doc_id] for doc_id in nearest]


def get_neighbors(vectors, index, train_labels, k_max, default_label):
  all_neighbors = []
  for x in vectors:
    # tweets with no usable words fall back to the majority class
    if sum(x.values()) == 0:
      all_neighbors.append([default_label] * k_max)
    else:
      all_neighbors.append(nearest_labels(x, index, train_labels, k_max))

  return all_neighbors


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


def f1_score(actual_labels, predicted_labels):
  tp, fp, fn, tn = confusion_matrix(actual_labels, predicted_labels)

  precision = tp / (tp + fp) if tp + fp > 0 else 0.0
  recall = tp / (tp + fn) if tp + fn > 0 else 0.0
  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0


# main program flow

def main():
  (train_vectors, train_labels, valid_vectors, valid_labels,
   test_vectors, test_labels, vocabulary) = prepare_data()

  # tf-idf with l2 normalization from the training data only
  train_tfidf, valid_tfidf, test_tfidf = prepare_tfidf(
    train_vectors, valid_vectors, test_vectors, len(vocabulary), True)

  index = build_index(train_tfidf)
  default_label = Counter(train_labels).most_common(1)[0][0]

  print("\nFinding nearest neighbors...")
  k_max = max(K_CANDIDATES)
  valid_neighbors = get_neighbors(valid_tfidf, index, train_labels, k_max, default_label)
  test_neighbors = get_neighbors(test_tfidf, index, train_labels, k_max, default_label)

  # choose k using the validation set so the test set stays untouched
  best_k = K_CANDIDATES[0]
  best_f1 = -1.0
  print("\nTuning k on the validation set")
  for k in K_CANDIDATES:
    predicted = [vote(neighbors, k) for neighbors in valid_neighbors]
    f1 = f1_score(valid_labels, predicted)
    print(f"k = {k:<3} F1-Score = {f1 * 100:.2f}%")
    if f1 > best_f1:
      best_f1 = f1
      best_k = k
  print(f"\nSelected k = {best_k}")

  valid_predicted = [vote(neighbors, best_k) for neighbors in valid_neighbors]
  test_predicted = [vote(neighbors, best_k) for neighbors in test_neighbors]

  print_metrics("VALIDATION", valid_labels, valid_predicted)
  print_metrics("TEST", test_labels, test_predicted)


if __name__ == "__main__":
  main()