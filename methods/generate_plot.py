import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Import your custom preprocessing module
from preprocess import prepare_data, prepare_tfidf

# Create figures directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)


def plot_pca_separability():
  print('1. Loading data...')
  (
      train_vectors,
      train_labels,
      valid_vectors,
      valid_labels,
      test_vectors,
      test_labels,
      vocab,
  ) = prepare_data()

  train_tfidf, _, _ = prepare_tfidf(
      train_vectors, valid_vectors, test_vectors, len(vocab), normalize=True
  )

  print('2. Building dense feature matrix...')
  vocab_size = len(vocab)
  num_samples = len(train_tfidf)
  X_matrix = np.zeros((num_samples, vocab_size))

  for i, doc in enumerate(train_tfidf):
    for idx, val in doc.items():
      X_matrix[i, idx] = val

  y_labels = np.array(train_labels)

  print('3. Performing PCA...')
  pca = PCA(n_components=2, random_state=42)
  X_2d = pca.fit_transform(X_matrix)

  print('4. Fitting Linear Classifier on 2D space...')
  # Linear SVM guarantees finding an optimal linear boundary in 2D coordinates
  clf = SVC(kernel='linear', random_state=42)
  clf.fit(X_2d, y_labels)

  print('5. Plotting decision boundary line...')
  plt.figure(figsize=(8, 6))

  # Plot transparent scatter points
  plt.scatter(
      X_2d[y_labels == 0, 0],
      X_2d[y_labels == 0, 1],
      c='blue',
      label='Class 0 (Non-hate)',
      alpha=0.3,
      s=15,
  )
  plt.scatter(
      X_2d[y_labels == 1, 0],
      X_2d[y_labels == 1, 1],
      c='red',
      label='Class 1 (Hate)',
      alpha=0.3,
      s=15,
  )

  # Calculate and plot exact decision boundary line: w0*x1 + w1*x2 + b = 0
  w = clf.coef_[0]
  b = clf.intercept_[0]

  x1_vals = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), 200)
  x2_vals = -(w[0] * x1_vals + b) / w[1]

  plt.plot(
      x1_vals,
      x2_vals,
      color='black',
      linestyle='--',
      linewidth=2.5,
      label='Perceptron Linear Boundary',
  )

  # Lock axis limits to the cluster region
  plt.xlim(X_2d[:, 0].min() - 0.02, X_2d[:, 0].max() + 0.02)
  plt.ylim(X_2d[:, 1].min() - 0.02, X_2d[:, 1].max() + 0.02)

  plt.xlabel('Principal Component 1 (PC1)')
  plt.ylabel('Principal Component 2 (PC2)')
  plt.title('2D PCA Projection: Non-Linear Separability of TF-IDF Features')
  plt.legend(loc='upper right')
  plt.grid(True, linestyle=':', alpha=0.5)

  output_path = 'results/figures/pca_perceptron_boundary.png'
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  print(f'Done! Saved graph with visible boundary line to {output_path}')


if __name__ == '__main__':
  plot_pca_separability()