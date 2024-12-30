import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist


class BayesianDecisionTree:
    def __init__(self, max_depth=16, min_samples_split=10, learning_rate=0.01, n_steps=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.learning_rate = learning_rate
        self.n_steps = n_steps  # 控制优化步数

    def _calculate_entropy(self, y):
        """Calculate entropy."""
        m = len(y)
        if m == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / m
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log(probabilities))

    def _calculate_information_gain(self, y, left_mask, right_mask):
        """Calculate information gain for a split."""
        m = len(y)
        m_left = np.sum(left_mask)
        m_right = np.sum(right_mask)

        if m_left == 0 or m_right == 0:
            return 0

        H_parent = self._calculate_entropy(y)
        H_left = self._calculate_entropy(y[left_mask])
        H_right = self._calculate_entropy(y[right_mask])

        weighted_entropy = (m_left / m) * H_left + (m_right / m) * H_right
        return H_parent - weighted_entropy

    def _elbo(self, thresholds, information_gains, mu_q, sigma_q, mu_0, sigma_0):
        """Compute the Evidence Lower Bound (ELBO)."""
        # Ensure sigma_q and sigma_0 are positive
        sigma_q = max(sigma_q, 1e-6)
        sigma_0 = max(sigma_0, 1e-6)

        # Approximate likelihood term
        likelihood_term = np.mean(np.log(information_gains + 1e-8))

        # KL divergence between q(t) and P(t)
        kl_divergence = (
            np.log(sigma_0 / sigma_q)
            + (sigma_q**2 + (mu_q - mu_0) ** 2) / (2 * sigma_0**2)
            - 0.5
        )

        return likelihood_term - kl_divergence

    def _update_distribution(self, thresholds, information_gains, mu_0, sigma_0):
        """Optimize the Gaussian distribution parameters using gradient ascent."""
        mu_q = np.mean(thresholds)  # Initialize mu_q
        sigma_q = np.std(thresholds)  # Initialize sigma_q

        # Ensure sigma_0 is positive
        sigma_0 = max(sigma_0, 1e-6)

        for _ in range(self.n_steps):  # Perform gradient steps
            # Ensure sigma_q remains positive
            sigma_q = max(sigma_q, 1e-6)

            # Compute ELBO
            elbo = self._elbo(thresholds, information_gains, mu_q, sigma_q, mu_0, sigma_0)

            # Gradients of ELBO w.r.t mu_q and sigma_q
            d_mu_q = (np.sum(information_gains * (thresholds - mu_q)) / sigma_q**2) - (
                (mu_q - mu_0) / sigma_0**2
            )
            d_sigma_q = (
                -1 / sigma_q
                + np.sum(information_gains * ((thresholds - mu_q) ** 2)) / sigma_q**3
                - sigma_q / sigma_0**2
            )

            # Regularization to prevent overflow
            d_mu_q = np.clip(d_mu_q, -1e3, 1e3)  # Clip gradients
            d_sigma_q = np.clip(d_sigma_q, -1e3, 1e3)

            # Update parameters with gradient
            mu_q += self.learning_rate * d_mu_q
            sigma_q += self.learning_rate * d_sigma_q

            # Ensure sigma_q remains in a reasonable range
            sigma_q = max(min(sigma_q, 1e2), 1e-6)

            # Stop if gradients explode
            if abs(d_mu_q) > 1e5 or abs(d_sigma_q) > 1e5:
                break

        return mu_q, sigma_q

    def _split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_params = None
        max_information_gain = float("-inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            information_gains = []

            valid_thresholds = []
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # 如果分裂后的左右子集为空，跳过该分裂点
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                info_gain = self._calculate_information_gain(y, left_mask, right_mask)
                information_gains.append(info_gain)
                valid_thresholds.append(threshold)

            if len(information_gains) > 0:
                information_gains = np.array(information_gains)
                valid_thresholds = np.array(valid_thresholds)

                # Prior parameters
                mu_0 = np.mean(valid_thresholds)
                sigma_0 = np.std(valid_thresholds)

                # Update posterior distribution
                mu_q, sigma_q = self._update_distribution(
                    valid_thresholds, information_gains, mu_0, sigma_0
                )

                if np.max(information_gains) > max_information_gain:
                    max_information_gain = np.max(information_gains)
                    best_feature = feature
                    best_threshold = mu_q  # Use the updated mean as the threshold
                    best_params = (mu_q, sigma_q)

        return best_feature, best_threshold, best_params

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or len(y) == 0
        ):
            return {"type": "leaf", "class": np.bincount(y).argmax() if len(y) > 0 else -1}

        feature, threshold, params = self._split(X, y)
        if feature is None:
            return {"type": "leaf", "class": np.bincount(y).argmax()}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 如果分裂后左右子集为空，则直接返回叶节点
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {"type": "leaf", "class": np.bincount(y).argmax()}

        return {
            "type": "node",
            "feature": feature,
            "threshold_dist": norm(loc=params[0], scale=params[1]),
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        """Fit the Bayesian Decision Tree."""
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_sample(self, x, node):
        """Predict a single sample using the tree."""
        if node["type"] == "leaf":
            return node["class"]

        threshold = node["threshold_dist"].rvs()  # Sample threshold from the learned distribution
        if x[node["feature"]] <= threshold:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])

    def predict(self, X, n_samples=10):
        """Predict with Bayesian sampling over thresholds."""
        predictions = []
        for _ in range(n_samples):
            sample_preds = [self._predict_sample(x, self.tree) for x in X]
            predictions.append(sample_preds)
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


# Example usage
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    bdt = BayesianDecisionTree(max_depth=16, min_samples_split=10, learning_rate=0.01, n_steps=50)
    bdt.fit(X_train, y_train)

    y_pred = bdt.predict(X_test, n_samples=100)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy on MNIST: {accuracy:.4f}")
