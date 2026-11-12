# runs the three methods and returns the numbers they report.

import knn
import naive_bayes
import perceptron


def run_methods():
  # each method's main() prints its usual output and returns its numbers
  return {"Naive Bayes": naive_bayes.main(), "k-NN": knn.main(), "Perceptron": perceptron.main()}


def main():
  run_methods()


if __name__ == "__main__":
  main()
