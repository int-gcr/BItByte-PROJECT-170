import csv
import math
import os
import re

# Path for dataset
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


# Loads Data
def load_split(file_name):
  path = os.path.join(DATASET_FOLDER, file_name)
  texts = []
  labels = []
  skipped = 0


  with open(path, "r", encoding="utf-8", newline="\n") as file:
    content = file.read()

  content = content.replace("\r", " ")
  lines = content.split("\n")

  reader = csv.reader(lines)
  header = next(reader) # Ignores header

  for row in reader:
    if len(row) == 0:
      continue

    # Skips row that isn't exactly 2 rows
    if len(row) != 2:
      skipped += 1
      continue

    texts.append(row[0]) # Appends first element to text
    labels.append(int(row[1])) # Appends second element to label

  print(f"Loaded {file_name}: {len(texts)} tweets, skipped {skipped}")
  return texts, labels


# Cleans text
def clean(text):
  lowered = text.lower()

  # Removes URL
  no_urls = re.sub(r'https?://\S+|www\.\S+', '', lowered)

  # Removes "@" at mentions
  no_mentions = re.sub(r'@\w+', '', no_urls)

  # Removes all non-alphanumeric text
  removed = re.sub(r'[^a-zA-Z0-9\s]', '', no_mentions)
  tokens = removed.split() # Tokenizes
  return tokens

# Builds vocabulary of all tokens
def build_vocabulary(train_texts):
  vocabulary = {}

  for text in train_texts:
    tokens = clean(text)
    for word in tokens:
      if word not in vocabulary:
        vocabulary[word] = len(vocabulary)

  return vocabulary

# Computes Bag of Words
def compute_bag_of_words(text, vocabulary):
  bow = {}
  tokens = clean(text)

  for word in tokens:
    if word in vocabulary:
      index = vocabulary[word]
      if index in bow:
        bow[index] += 1
      else:
        bow[index] = 1

  return bow

def vectorize(texts, vocabulary):
  vectors = []

  for text in texts:
    vectors.append(compute_bag_of_words(text, vocabulary))

  return vectors

# Entry point of three CSVs
def prepare_data():
  train_texts, train_labels = load_split("train.csv")
  valid_texts, valid_labels = load_split("valid.csv")
  test_texts, test_labels = load_split("test.csv")

  vocabulary = build_vocabulary(train_texts)

  train_vectors = vectorize(train_texts, vocabulary)
  valid_vectors = vectorize(valid_texts, vocabulary)
  test_vectors = vectorize(test_texts, vocabulary)

  print(f"Vocabulary size: {len(vocabulary)}")
  print(f"Training instances  : {len(train_vectors)}")
  print(f"Validation instances: {len(valid_vectors)}")
  print(f"Testing instances   : {len(test_vectors)}")

  return train_vectors, train_labels, valid_vectors, valid_labels, test_vectors, test_labels, vocabulary


# Computes for Document Frequenct, how many tweets does this word appear
def compute_document_frequencies(train_vectors, vocab_size):
  document_frequencies = []

  for i in range(vocab_size):
    document_frequencies.append(0)

  for vector in train_vectors:
    for index in vector:
      document_frequencies[index] += 1

  return document_frequencies

# Computes for IDF: idf = log(N / df)
def compute_idf(document_frequencies, number_of_documents):
  idf = []

  for df in document_frequencies:
    idf.append(math.log(number_of_documents / df))

  return idf

# Computes for TF: tf = count / total words
def compute_tfidf(vector, idf):
  total_words = 0
  for index, count in vector.items():
    total_words += count

  tfidf = {}
  for index, count in vector.items():
    tfidf[index] = (count / total_words) * idf[index]

  return tfidf

# L2 normalization: hatiin ang bawat value sa haba ng vector (sqrt ng sum ng squares)
# para pantay ang haba ng lahat ng tweet

# Normalizes a vector so its length is 1
def l2_normalize(vector):
  square_sum = 0.0
  for index, value in vector.items():
    square_sum += value * value

  length = math.sqrt(square_sum)

  normalized = {}
  if length > 0:
    for index, value in vector.items():
      normalized[index] = value / length
  else:
    for index, value in vector.items():
      normalized[index] = value

  return normalized

def vectorize_tfidf(vectors, idf, normalize):
  tfidf_vectors = []

  for vector in vectors:
    tfidf = compute_tfidf(vector, idf)
    if normalize:
      tfidf = l2_normalize(tfidf)
    tfidf_vectors.append(tfidf)

  return tfidf_vectors

# Applies TF-IDF across vectors
def prepare_tfidf(train_vectors, valid_vectors, test_vectors, vocab_size, normalize):
  document_frequencies = compute_document_frequencies(train_vectors, vocab_size)
  idf = compute_idf(document_frequencies, len(train_vectors))

  return (vectorize_tfidf(train_vectors, idf, normalize),
          vectorize_tfidf(valid_vectors, idf, normalize),
          vectorize_tfidf(test_vectors, idf, normalize))


# Confusion matrix implementation
def confusion_matrix(actual_labels, predicted_labels):
  tp = 0 # True Positive
  fp = 0 # False positive
  fn = 0 # False Negative
  tn = 0 # True Negative

  for i in range(len(actual_labels)):
    if actual_labels[i] == 1 and predicted_labels[i] == 1:
      tp += 1
    elif actual_labels[i] == 0 and predicted_labels[i] == 1:
      fp += 1
    elif actual_labels[i] == 1 and predicted_labels[i] == 0:
      fn += 1
    else:
      tn += 1

  return tp, fp, fn, tn

def print_metrics(split_name, actual_labels, predicted_labels):
  tp, fp, fn, tn = confusion_matrix(actual_labels, predicted_labels)

  accuracy = (tp + tn) / len(actual_labels)

  
  precision = 0.0
  if tp + fp > 0:
    precision = tp / (tp + fp)

  recall = 0.0
  if tp + fn > 0:
    recall = tp / (tp + fn)

  f1 = 0.0
  if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)

  print(f"\nRESULTS ON {split_name}")
  print(f"Accuracy : {accuracy * 100:.2f}%")
  print(f"Precision: {precision * 100:.2f}%")
  print(f"Recall   : {recall * 100:.2f}%")
  print(f"F1-Score : {f1 * 100:.2f}%")
  print("\nConfusion Matrix")
  print("\t\tPredicted 0\tPredicted 1")
  print(f"Actual 0\t{tn}\t\t{fp}")
  print(f"Actual 1\t{fn}\t\t{tp}")
