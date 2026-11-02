# knn method implementation

from collections import Counter
import math
from preprocess import clean, load_split


def load_data(filename):
    texts, labels = load_split(filename)
    data = []

    for text, label in zip(texts, labels):
      counts = Counter(clean(text))
      data.append((counts, label))

    return data

def minkowski_distance(x, v, p):
  # Missing tokens have a count of zero in the other document.
  distance = 0.0
  for token in set(x).union(v):
    distance += abs(x.get(token, 0) - v.get(token, 0)) ** p

  return math.pow(distance, 1 / p)

def classify(x, train_data, k, p):
  distances = []
  # calculate distance for each training data
  for v,label in train_data:
    distance = minkowski_distance(x, v, p)
    if distance == -1:
      continue
    # append if valid
    distances.append((distance,label))

  distances.sort(key=lambda item: item[0])
  nearest_neighbors = distances[:k]

  rain_count = 0
  no_rain_count = 0

# Iterate through each distance and check the label
  for d,label in nearest_neighbors:
    if label == 1:
      rain_count += 1
    else:
      no_rain_count += 1

  # Return majority vote
  if rain_count > no_rain_count:
    return 1
  else:
    return 0

# function to convert the text from initial num value
# back to its actual label
def label_to_text(label):
  if label == 1:
    return "rain"
  else:
    return "no rain"


# main program flow

def main():
  training_data = load_data("train.csv")
  testing_data = load_data("test.csv")

  print(f"Number of training records: {len(training_data)}")
  print(f"Number of testing records: {len(testing_data)}\n")

  while True:
    try:
      value_k = int(input("Enter an odd value of k: "))
    except ValueError:
      print("Error: k must be a positive odd integer.")
      continue
    if value_k <= 0 or value_k % 2 == 0:
      print("Error: k must be a positive odd integer.")
    else:
      break
  while True:
    try:
      value_p = float(input("Enter the value of p for Minkowski distance: "))
    except ValueError:
      print("Error: p must be a number greater than or equal to 1.")
      continue
    if value_p < 1.0:
      print("Error: p must be greater than or equal to 1.")
    else:
      break
  print("\nClassifying test instances...\n")

  predicted_labels = []
  actual_labels = []
  for features, actual_label in testing_data:
    predicted_labels.append(classify(features, training_data, value_k, value_p))
    actual_labels.append(actual_label)

  correct = sum(
    predicted == actual
    for predicted, actual in zip(predicted_labels, actual_labels)
  )
  incorrect = len(actual_labels) - correct
  accuracy = correct / len(actual_labels) * 100

  print("KNN CLASSIFICATION RESULTS")
  print("k = ", value_k)
  print("p = ", value_p)
  print()
  print(f"Number of correct predictions: {correct}")
  print(f"Number of incorrect predictions: {incorrect}")
  print(f"Accuracy: {accuracy:.2f}%")
  print()
  print("Incorrect classifications")
  print("Data Record #  Actual Predicted")
  for i, (predicted, actual) in enumerate(zip(predicted_labels, actual_labels)):
    if predicted != actual:
      print(f"{i+1:^10} {label_to_text(actual):^13} {label_to_text(predicted)}")


if __name__ == "__main__":
  main()







