# Filipino Hate Speech Classification

CMSC 170 (Introduction to Artificial Intelligence) project by **BItByte**.

This project sorts Filipino and Taglish election tweets into hate (`1`) or non-hate (`0`). It compares three methods that we wrote from scratch in Python:

- **Naive Bayes**: bag-of-words with Laplace smoothing
- **k-NN**: TF-IDF vectors with cosine similarity, with k tuned on the validation set
- **Perceptron**: TF-IDF vectors with a perceptron trained over several epochs

## Members

| Group Name | BItByte |
| --- | --- |
| Members | Ralph Crisostomo |
| | Jorge Paguirigan |
| | Geoseffe Rufo |
| | Raymond Sariño |

## Project Structure

```
.
├── dataset/
│   ├── train.csv        # training tweets (text, label)
│   ├── valid.csv        # validation tweets
│   └── test.csv         # test tweets
├── methods/
│   ├── preprocess.py    # loading, cleaning, bag-of-words and tf-idf
│   ├── naive_bayes.py
│   ├── knn.py
│   ├── perceptron.py
│   ├── evaluate.py      # runs all three methods and draws the graphs
│   └── results/figures/ # generated graphs
└── index.html           # project write-up page
```

## Running

You need Python 3. The graphs also need `matplotlib`:

```bash
pip install matplotlib
```

Run the scripts from inside the `methods/` folder:

```bash
cd methods

# run one method
python naive_bayes.py
python knn.py
python perceptron.py

# run all three and save the graphs to results/figures/
python evaluate.py
```
