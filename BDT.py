import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from tensorflow.keras.datasets import mnist

class BayesianDecisionTree:
    def __init__(self, max_depth=16, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _calculate_gini(self, y):
        """Calculate Gini impurity."""
        m = len(y)
        if m == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / m
        return 1 - np.sum(probabilities**2)

    def _split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_params = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_left = self._calculate_gini(y[left_mask])
                gini_right = self._calculate_gini(y[right_mask])
                
                weighted_gini = (
                    len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right
                ) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
                    best_params = (np.mean(thresholds), np.std(thresholds))

        return best_feature, best_threshold, best_params

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {
                'type': 'leaf',
                'class': np.bincount(y).argmax(),
            }

        feature, threshold, params = self._split(X, y)
        if feature is None:
            return {
                'type': 'leaf',
                'class': np.bincount(y).argmax(),
            }

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            'type': 'node',
            'feature': feature,
            'threshold_dist': norm(loc=params[0], scale=params[1] if params[1] > 0 else 1e-6),
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        """Fit the Bayesian Decision Tree."""
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_sample(self, x, node):
        """Predict a single sample using the tree."""
        if node['type'] == 'leaf':
            return node['class']

        threshold = node['threshold_dist'].rvs()  # Sample threshold from the distribution
        if x[node['feature']] <= threshold:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X, n_samples=10):
        """Predict with Bayesian sampling over thresholds."""
        predictions = []
        for _ in range(n_samples):
            sample_preds = [self._predict_sample(x, self.tree) for x in X]
            predictions.append(sample_preds)
        predictions = np.array(predictions)
        # Return the mode of the sampled predictions
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten and normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Train Bayesian Decision Tree
    bdt = BayesianDecisionTree(max_depth=16, min_samples_split=10)
    bdt.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = bdt.predict(X_test, n_samples=50)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on MNIST: {accuracy:.4f}")
