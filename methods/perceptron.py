from collections import Counter

from preprocess import prepare_data, prepare_tfidf, print_metrics

MAX_EPOCH = 1000
LEARNING_RATE = 0.001
BIAS = 0.0
NORMALIZE = True
TOP_WORDS = 15

# Step 1: initialize the bias and all feature weights to zero.
def initialization(bias, number_of_features):
  weights = [bias]
  for i in range(number_of_features):
    weights.append(0.0)
  return weights

# Step 2: add the bias feature (index 0) to the sparse tweet vector.
def activation(vector):
  x = {0: 1.0}
  for index, count in vector.items():
    x[index + 1] = float(count)
  return x

# Step 3: calculate the score and predict -1 or 1.
def classify(weights, x):
  score = 0.0
  for index, value in x.items():
    score += weights[index] * value
  if score >= 0:
    return score, 1
  return score, -1

# Step 4: update the weights only when the prediction is wrong.
def update_weight_vector(weights, x, desired, predicted, learning_rate):
  updated = False
  if predicted != desired:
    for index, value in x.items():
      weights[index] += learning_rate * (desired - predicted) * value
    updated = True
  return weights, updated

# Step 6: zero updates in an entire epoch means the training set is classified correctly.
def convergence_check(updates):
  return updates == 0

# Dataset labels are 0 and 1; the Perceptron uses -1 and 1.
def to_desired(label):
  return 1 if label == 1 else -1

# Count the model's mistakes after an epoch has finished.
def evaluate_training_errors(vectors, labels, weights):
  false_positives = 0
  false_negatives = 0

  for i in range(len(vectors)):
    _, predicted = classify(weights, activation(vectors[i]))
    actual = to_desired(labels[i])

    if predicted != actual:
      if predicted == 1:
        false_positives += 1
      else:
        false_negatives += 1

  errors = false_positives + false_negatives
  accuracy = (len(labels) - errors) / len(labels) * 100
  return errors, false_positives, false_negatives, accuracy

# An identical feature vector with both labels cannot be separated by any classifier.
def count_conflicting_vectors(vectors, labels):
  seen_labels = {}

  for vector, label in zip(vectors, labels):
    key = tuple(sorted(vector.items()))
    if key not in seen_labels:
      seen_labels[key] = set()
    seen_labels[key].add(label)

  conflicts = 0
  for label_set in seen_labels.values():
    if len(label_set) > 1:
      conflicts += 1
  return conflicts

# Count words that appear in incorrectly classified training tweets.
def most_misclassified_words(vectors, labels, weights, vocabulary, top_n=TOP_WORDS):
  word_counts = Counter()
  index_to_word = {index: word for word, index in vocabulary.items()}

  for vector, label in zip(vectors, labels):
    _, predicted = classify(weights, activation(vector))
    if predicted != to_desired(label):
      for index in vector:
        if index in index_to_word:
          word_counts[index_to_word[index]] += 1

  return word_counts.most_common(top_n)


def perceptron(train_vectors, train_labels, vocab_size):
  weights = initialization(BIAS, vocab_size)
  epochs = []
  history = []

  for epoch in range(1, MAX_EPOCH + 1):
    updates = 0

    for i in range(len(train_vectors)):
      x = activation(train_vectors[i])
      desired = to_desired(train_labels[i])
      _, predicted = classify(weights, x)
      weights, updated = update_weight_vector(weights, x, desired, predicted, LEARNING_RATE)
      if updated:
        updates += 1

    errors, fp, fn, accuracy = evaluate_training_errors(train_vectors, train_labels, weights)
    epochs.append((epoch, updates))
    history.append({"epoch": epoch, "updates": updates, "errors": errors,
                    "false_positives": fp, "false_negatives": fn, "accuracy": accuracy})

    print(f"Epoch {epoch}: {updates} updates | {errors} errors remain "
          f"(FP: {fp}, FN: {fn}) | Training accuracy: {accuracy:.2f}%")

    if convergence_check(updates):
      return True, weights, epochs, history

  return False, weights, epochs, history


def predict_all(vectors, weights):
  predictions = []
  for vector in vectors:
    _, predicted = classify(weights, activation(vector))
    predictions.append(1 if predicted == 1 else 0)
  return predictions


def main():
  train_vectors, train_labels, valid_vectors, valid_labels, test_vectors, test_labels, vocabulary = prepare_data()
  train_vectors, valid_vectors, test_vectors = prepare_tfidf(
    train_vectors, valid_vectors, test_vectors, len(vocabulary), NORMALIZE)

  print(f"\nPERCEPTRON (learning rate = {LEARNING_RATE}, bias = {BIAS}, max epoch = {MAX_EPOCH})")
  converged, weights, epochs, history = perceptron(train_vectors, train_labels, len(vocabulary))

  if converged:
    print("\nConverged: the model classified every training example correctly.")
  else:
    print("\nDid not converge: the model still makes training errors at the epoch limit.")

  conflicts = count_conflicting_vectors(train_vectors, train_labels)
  print(f"Identical feature vectors with conflicting labels: {conflicts}")
  if conflicts:
    print("These conflicting examples prove that this training representation is not perfectly linearly separable.")
  else:
    print("No exact conflicting vectors were found. This does not prove that the full dataset is linearly separable.")

  print(f"\nTOP {TOP_WORDS} WORDS IN MISCLASSIFIED TRAINING TWEETS")
  top_words = most_misclassified_words(train_vectors, train_labels, weights, vocabulary)
  if top_words:
    for word, count in top_words:
      print(f"{word}: {count} misclassified tweets")
  else:
    print("No misclassified training tweets.")

  return {
    "VALIDATION": print_metrics("VALIDATION", valid_labels, predict_all(valid_vectors, weights)),
    "TEST": print_metrics("TEST", test_labels, predict_all(test_vectors, weights)),
    "epochs": epochs,
    "history": history,
    "top_misclassified_words": top_words,
    "conflicting_vectors": conflicts
  }


if __name__ == "__main__":
  main()
