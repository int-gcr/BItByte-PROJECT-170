from preprocess import prepare_data, prepare_tfidf, print_metrics

MAX_EPOCH = 100
LEARNING_RATE = 0.5
BIAS = 0.0
NORMALIZE = True

# Perceptron Algorithm
# Step 1
def initialization(bias, number_of_features):
  # Set the weight vector to 0, w0 is our bias
  weights = [bias]

  for i in range(number_of_features):
    weights.append(0.0)

  return weights

# Step 2
def activation(vector):
  # x0 is always 1. x is sparse (index -> value) because there are thousands of words out there
  # but only a handful actually show up in a tweet. The 0s won't affect our sum or updates anyway.
  x = {0: 1.0}

  for index, count in vector.items():
    x[index + 1] = float(count)

  return x

# Step 3
def classify(weights, x):
  v = 0.0
  for index, value in x.items():
    v += weights[index] * value

  if v >= 0:
    y = 1
  else:
    y = -1

  return v, y

# Step 4
# w(n+1) = w(n) + eta * (d(n) - y(n)) * x(n)
def update_weight_vector(weights, x, d, y, learning_rate):
  updated = False

  if y != d:
    for index, value in x.items():
      weights[index] = weights[index] + learning_rate * (d - y) * value
    updated = True

  return weights, updated

# Step 6
def convergence_check(updates):
  if updates == 0:
    return True
  return False

# Label 0 becomes -1 since our signum function only spits out -1 and 1
def to_desired(label):
  if label == 1:
    return 1
  return -1

def perceptron(train_vectors, train_labels, vocab_size):
  # Step 1
  weights = initialization(BIAS, vocab_size)

  for epoch in range(1, MAX_EPOCH + 1):
    updates = 0

    for i in range(len(train_vectors)):
      # Step 2
      x = activation(train_vectors[i])
      d = to_desired(train_labels[i])

      # Step 3
      v, y = classify(weights, x)

      # Step 4
      weights, updated = update_weight_vector(weights, x, d, y, LEARNING_RATE)
      if updated:
        updates += 1

      # Step 5: next training data

    print(f"Epoch {epoch}: {updates} updates")

    # Step 6
    if convergence_check(updates):
      return True, weights

  return False, weights

# Predict labels for a whole list of tweets
def predict_all(vectors, weights):
  predictions = []

  for vector in vectors:
    v, y = classify(weights, activation(vector))
    # classify() gives us -1 or 1, but our original labels use 0 or 1, so we map -1 back to 0
    if y == 1:
      predictions.append(1)
    else:
      predictions.append(0)

  return predictions


# Load all the raw text, build the vocabulary, and turn them into Bag-of-Words
train_vectors, train_labels, valid_vectors, valid_labels, test_vectors, test_labels, vocabulary = prepare_data()

# Convert word counts into TF-IDF scores so super common words don't mess up our math
train_vectors, valid_vectors, test_vectors = prepare_tfidf(train_vectors, valid_vectors, test_vectors, len(vocabulary), NORMALIZE)

# Train the perceptron
print(f"\nPERCEPTRON (learning rate = {LEARNING_RATE}, bias = {BIAS}, max epoch = {MAX_EPOCH})")
converged, weights = perceptron(train_vectors, train_labels, len(vocabulary))

# If updates hit 0 before MAX_EPOCH, it means it found a clean boundary line between the classes
if converged:
  print("Data is linearly separable")
else:
  print("Data may not be linearly separable")

# Finally, check how well our weights actually perform on data it hasn't learned from yet
print_metrics("VALIDATION", valid_labels, predict_all(valid_vectors, weights))
print_metrics("TEST", test_labels, predict_all(test_vectors, weights))