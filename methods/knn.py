# knn method implementation

import heapq
from collections import Counter
from preprocess import prepare_data, prepare_tfidf, confusion_matrix, print_metrics

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51, 61, 71, 81, 91, 101, 151, 201, 251, 301, 401, 501]

# each neighbor votes with weight similarity to the power p
# p of 0 is the plain majority vote where every neighbor counts the same
POWER_CANDIDATES = [0, 1, 2, 3, 5]

# a tweet is labeled hate when the hate share of the weighted vote is at least the threshold
# a threshold of one half is the usual majority rule
THRESHOLD_CANDIDATES = [round(0.30 + 0.02 * i, 2) for i in range(16)]


def build_index(train_vectors):
  # inverted index maps each word to the training tweets that contain it
  index = {}
  for doc_id, vector in enumerate(train_vectors):
    for word, weight in vector.items():
      index.setdefault(word, []).append((doc_id, weight))

  return index


def nearest_neighbors(x, index, train_labels, k_max):
  # cosine similarity of l2 normalized vectors is just the dot product
  scores = [0.0] * len(train_labels)
  for word, weight in x.items():
    for doc_id, train_weight in index.get(word, []):
      scores[doc_id] += weight * train_weight

  # larger similarity means a smaller cosine distance
  nearest = heapq.nlargest(k_max, range(len(scores)), key=scores.__getitem__)

  # keep the similarity so closer neighbors can count more in the vote
  return [(train_labels[doc_id], scores[doc_id]) for doc_id in nearest]


def get_neighbors(vectors, index, train_labels, k_max, default_label):
  all_neighbors = []
  for x in vectors:
    # tweets with no usable words fall back to the majority class
    if sum(x.values()) == 0:
      all_neighbors.append([(default_label, 1.0)] * k_max)
    else:
      all_neighbors.append(nearest_neighbors(x, index, train_labels, k_max))

  return all_neighbors


def vote_weight(similarity, power):
  if power == 0:
    return 1.0
  return max(similarity, 0.0) ** power


def hate_shares(neighbors, power):
  # running totals give the hate share of the vote for every k in one pass
  # shares k is the hate share among the k nearest neighbors
  shares = {}
  hate_total = 0.0
  weight_total = 0.0
  wanted = set(K_CANDIDATES)
  for position, (label, similarity) in enumerate(neighbors, start=1):
    weight = vote_weight(similarity, power)
    weight_total += weight
    if label == 1:
      hate_total += weight
    if position in wanted:
      shares[position] = hate_total / weight_total if weight_total > 0 else 0.0

  return shares


def vote(neighbors, k, power, threshold):
  hate_weight = 0.0
  total_weight = 0.0

  # iterate through the k nearest neighbors and add up their weighted votes
  for label, similarity in neighbors[:k]:
    weight = vote_weight(similarity, power)
    total_weight += weight
    if label == 1:
      hate_weight += weight

  if total_weight == 0:
    return 0

  # hate wins when its share of the vote reaches the threshold
  if hate_weight / total_weight >= threshold:
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


def tune(valid_neighbors, valid_labels):
  # try every power k and threshold on the validation set and keep the best f1
  best = (-1.0, 0, K_CANDIDATES[0], 0.5)
  f1_table = {}
  for power in POWER_CANDIDATES:
    all_shares = [hate_shares(neighbors, power) for neighbors in valid_neighbors]
    for k in K_CANDIDATES:
      for threshold in THRESHOLD_CANDIDATES:
        predicted = [1 if shares[k] >= threshold else 0 for shares in all_shares]
        f1 = f1_score(valid_labels, predicted)
        f1_table[(power, k, threshold)] = f1
        if f1 > best[0]:
          best = (f1, power, k, threshold)

  return best, f1_table


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

  # choose power k and threshold using the validation set so the test set stays untouched
  (best_f1, best_power, best_k, best_threshold), f1_table = tune(valid_neighbors, valid_labels)

  # validation f1 for each k at the chosen power and threshold for the tuning graph
  k_tuning = []
  print(f"\nTuning k on the validation set (power = {best_power}, threshold = {best_threshold:.2f})")
  for k in K_CANDIDATES:
    f1 = f1_table[(best_power, k, best_threshold)]
    print(f"k = {k:<3} F1-Score = {f1 * 100:.2f}%")
    k_tuning.append((k, round(f1 * 100, 2)))

  plain_f1 = max(f1_table[(0, k, 0.5)] for k in K_CANDIDATES)
  print(f"\nPlain majority vote best validation F1 = {plain_f1 * 100:.2f}%")
  print(f"Selected k = {best_k}, power = {best_power}, threshold = {best_threshold:.2f} "
        f"(validation F1 = {best_f1 * 100:.2f}%)")

  valid_predicted = [vote(n, best_k, best_power, best_threshold) for n in valid_neighbors]
  test_predicted = [vote(n, best_k, best_power, best_threshold) for n in test_neighbors]

  return {"VALIDATION": print_metrics("VALIDATION", valid_labels, valid_predicted),
          "TEST": print_metrics("TEST", test_labels, test_predicted),
          "k_tuning": k_tuning, "selected_k": best_k,
          "selected_power": best_power, "selected_threshold": best_threshold}


if __name__ == "__main__":
  main()