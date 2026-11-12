from decimal import Decimal as dec
from preprocess import load_split, clean, print_metrics

# smoothing factor k (k = 1 ay Laplace smoothing sa Lab 04)
k = 1

# combined bag-of-words ng isang class
# {word: count}
def compute_combined_dictionary(texts):
  combined_dictionary = {}

  for text in texts:
    for word in clean(text):
      combined_dictionary[word] = combined_dictionary.get(word, 0) + 1

  return combined_dictionary

def train():
  # Training: hatiin muna ang tweets based sa label
  train_texts, train_labels = load_split("train.csv")
  hate_texts = []
  nonhate_texts = []
  for i in range(len(train_texts)):
    if train_labels[i] == 1:
      hate_texts.append(train_texts[i])
    else:
      nonhate_texts.append(train_texts[i])

  hate_dictionary = compute_combined_dictionary(hate_texts)
  nonhate_dictionary = compute_combined_dictionary(nonhate_texts)

  # class priors
  total_tweets = dec(len(train_texts))
  p_hate = dec(len(hate_texts)) / total_tweets
  p_nonhate = dec(len(nonhate_texts)) / total_tweets

  # vocabulary ay union ng dalawang dictionary
  vocabulary = set(hate_dictionary.keys()).union(set(nonhate_dictionary.keys()))
  vocab_size = dec(len(vocabulary))

  # total words per class
  total_words_hate = dec(sum(hate_dictionary.values()))
  total_words_nonhate = dec(sum(nonhate_dictionary.values()))

  print(f"\nNAIVE BAYES (k = {k})")
  print(f"P(Non-hate) = {p_nonhate:.4f}")
  print(f"P(Hate)     = {p_hate:.4f}")

  return hate_dictionary, nonhate_dictionary, p_hate, p_nonhate, vocab_size, total_words_hate, total_words_nonhate

# 1 = hate, 0 = nonhate
def classify_tweet(text):
  num_hate_prob = p_hate
  num_nonhate_prob = p_nonhate

  for token in clean(text):
    count_hate = dec(hate_dictionary.get(token, 0))
    count_nonhate = dec(nonhate_dictionary.get(token, 0))

    num_hate_prob *= (count_hate + k) / (total_words_hate + (k * vocab_size))
    num_nonhate_prob *= (count_nonhate + k) / (total_words_nonhate + (k * vocab_size))

  # dito dapat ang normalizing step (score / (num_hate_prob + num_nonhate_prob)),
  # pero pareho ang denominator ng dalawang class kaya hindi na kailangan
  
  if num_hate_prob >= num_nonhate_prob:
    return 1
  return 0

def classify_all(texts):
  predictions = []

  for text in texts:
    predictions.append(classify_tweet(text))

  return predictions

def main():
  # classify_tweet reads these as globals, so main has to set the globals
  global hate_dictionary, nonhate_dictionary, p_hate, p_nonhate, vocab_size, total_words_hate, total_words_nonhate

  valid_texts, valid_labels = load_split("valid.csv")
  test_texts, test_labels = load_split("test.csv")
  hate_dictionary, nonhate_dictionary, p_hate, p_nonhate, vocab_size, total_words_hate, total_words_nonhate = train()

  return {"VALIDATION": print_metrics("VALIDATION", valid_labels, classify_all(valid_texts)),
          "TEST": print_metrics("TEST", test_labels, classify_all(test_texts))}

if __name__ == "__main__":
  main()

