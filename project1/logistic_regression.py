import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        
        # Model parameters - will be initialized in fit()
        self.theta = None  # weights
        self.bias = None   # bias term
        
        # Training history for ablation analysis
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': []
        }

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
            Train the logistic regression model using stochastic gradient descent.
            
            Args:
                X: Training features
                Y: Training labels
                X_val: Validation features (optional, for loss tracking)
                Y_val: Validation labels (optional, for loss tracking)
        """
        # Initialize parameters
        n_features = X.shape[1]
        self.theta = np.random.normal(0, 0.01, n_features)  # Small random weights
        self.bias = 0.0  # Start bias at zero
        
        # Clear training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': []
        }
        
        # Training loop
        for epoch in tqdm.tqdm(range(self.num_epochs), disable=not self.verbose):
            # Shuffle data for this epoch
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            # Process data in batches
            for i in range(0, len(X), self.batch_size):
                # Get current batch
                batch_end = min(i + self.batch_size, len(X))
                X_batch = X_shuffled[i:batch_end]
                Y_batch = Y_shuffled[i:batch_end]
                
                # Calculate gradients on this batch
                theta_grad, bias_grad = self.gradient(X_batch, Y_batch)
                
                # Update parameters using gradient descent
                self.theta = self.theta - self.learning_rate * theta_grad
                self.bias = self.bias - self.learning_rate * bias_grad
            
            # Track losses every epoch for ablation analysis
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:  # Every 5 epochs + final
                train_loss = self.compute_loss(X, Y)
                self.training_history['train_losses'].append(train_loss)
                self.training_history['epochs'].append(epoch)
                
                if X_val is not None and Y_val is not None:
                    val_loss = self.compute_loss(X_val, Y_val)
                    self.training_history['val_losses'].append(val_loss)
                else:
                    self.training_history['val_losses'].append(None)

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        # Get current predictions using our stable sigmoid
        predictions = self.predict_proba(X)
        
        # Calculate prediction errors: (p - y)
        errors = predictions - Y
        
        # Number of samples
        m = X.shape[0]
        
        # Gradient for weights: X^T * (p - y) + λ*θ
        # We divide by m to get average gradient across batch
        theta_gradient = (1/m) * np.dot(X.T, errors) + self.regularization_lambda * self.theta
        
        # Gradient for bias: sum(p - y) 
        # We divide by m to get average gradient across batch
        bias_gradient = (1/m) * np.sum(errors)
        
        return theta_gradient, bias_gradient

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        # Compute linear combination: z = X * theta + bias
        z = np.dot(X, self.theta) + self.bias
        
        # Apply stable sigmoid function
        return self._stable_sigmoid(z)
    
    def _stable_sigmoid(self, z):
        """
            Numerically stable sigmoid function.
            Uses different formulas for positive and negative inputs to avoid overflow.
        """
        # For positive z: use 1 / (1 + exp(-z))
        # For negative z: use exp(z) / (1 + exp(z))
        
        positive_mask = z >= 0
        
        # Initialize output array
        sigmoid_output = np.zeros_like(z)
        
        # Handle positive values: 1 / (1 + exp(-z))
        sigmoid_output[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        
        # Handle negative values: exp(z) / (1 + exp(z))
        negative_mask = ~positive_mask
        exp_z = np.exp(z[negative_mask])
        sigmoid_output[negative_mask] = exp_z / (1 + exp_z)
        
        return sigmoid_output
    
    def compute_loss(self, X, Y):
        """
            Compute the binary cross-entropy loss with L2 regularization.
            
            Args:
                X: Feature matrix
                Y: True labels
                
            Returns:
                loss: Average loss across all samples
        """
        # Get predictions
        predictions = self.predict_proba(X)
        
        # Prevent log(0) by clipping predictions
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]
        bce_loss = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
        
        # Add L2 regularization: λ/2 * ||θ||²
        l2_penalty = 0.5 * self.regularization_lambda * np.sum(self.theta ** 2)
        
        return bce_loss + l2_penalty

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        # Get probabilities and convert to binary predictions
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)