import numpy as np

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        # Compute mean and standard deviation from training values
        mean = np.mean(values)
        std = np.std(values)
        
        if verbose:
            print(f"Numerical feature - Mean: {mean:.2f}, Std: {std:.2f}")
        
        # Handle edge case where std is 0 (all values are the same)
        if std == 0:
            std = 1.0  # Avoid division by zero
        
        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score (z-score normalization)
            """
            # Handle missing or invalid values
            if x is None or x == '' or x == 'nan':
                return 0.0  # Handle missing values by returning 0 (mean)
            
            try:
                x_float = float(x)
                if np.isnan(x_float):
                    return 0.0
                return (x_float - mean) / std
            except (ValueError, TypeError):
                return 0.0  # Handle any conversion errors

        return vectorizer

    def get_histogram_vectorizer(self, values):
        raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values, verbose=False):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        # Get unique values and sort them for consistent ordering
        unique_values = sorted(list(set(str(v) for v in values if v is not None and v != '')))
        
        if verbose:
            print(f"Categorical feature - Unique values: {unique_values}")
        
        # Create mapping from value to index
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}
        
        def vectorizer(x):
            """
            :param x: categorical value
            Return one-hot encoded vector
            """
            # Handle missing values
            if x is None or x == '' or x == 'nan':
                return [0.0] * len(unique_values)  # All zeros for missing
            
            # Convert to string for consistent lookup
            x_str = str(x)
            
            # Create one-hot vector
            one_hot = [0.0] * len(unique_values)
            if x_str in value_to_index:
                one_hot[value_to_index[x_str]] = 1.0
            # If unknown value, leave all zeros (like missing)
            
            return one_hot
        
        return vectorizer

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        
        
        self.feature_transforms = {}
        
        # Handle numerical features
        if "numerical" in self.feature_config:
            for feature_name in self.feature_config["numerical"]:
                # Extract values for this feature from all training samples
                values = [float(sample[feature_name]) for sample in X if sample[feature_name] is not None and sample[feature_name] != '']
                
                if len(values) == 0:
                    raise ValueError(f"No valid values found for numerical feature: {feature_name}")
                
                # Create and store the vectorizer function for this feature
                vectorizer_fn = self.get_numerical_vectorizer(values, verbose=True)
                self.feature_transforms[feature_name] = vectorizer_fn
        
        # Handle categorical features
        if "categorical" in self.feature_config:
            for feature_name in self.feature_config["categorical"]:
                # Extract values for this feature from all training samples
                values = [sample[feature_name] for sample in X if sample[feature_name] is not None and sample[feature_name] != '']
                
                if len(values) == 0:
                    raise ValueError(f"No valid values found for categorical feature: {feature_name}")
                
                # Create and store the vectorizer function for this feature
                vectorizer_fn = self.get_categorical_vectorizer(values, verbose=True)
                self.feature_transforms[feature_name] = vectorizer_fn
        
        # Handle ordinal features (treat as numerical but with integer values)
        if "ordinal" in self.feature_config:
            for feature_name in self.feature_config["ordinal"]:
                # Extract values for this feature from all training samples
                values = [float(sample[feature_name]) for sample in X if sample[feature_name] is not None and sample[feature_name] != '']
                
                if len(values) == 0:
                    raise ValueError(f"No valid values found for ordinal feature: {feature_name}")
                
                # Create and store the vectorizer function (same as numerical)
                vectorizer_fn = self.get_numerical_vectorizer(values, verbose=True)
                self.feature_transforms[feature_name] = vectorizer_fn
        
        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []
        
        for sample in X:
            feature_vector = []
            
            # Process numerical features in consistent order
            if "numerical" in self.feature_config:
                for feature_name in self.feature_config["numerical"]:
                    vectorizer_fn = self.feature_transforms[feature_name]
                    raw_value = sample.get(feature_name)
                    normalized_value = vectorizer_fn(raw_value)
                    feature_vector.append(normalized_value)
            
            # Process categorical features (one-hot encoded)
            if "categorical" in self.feature_config:
                for feature_name in self.feature_config["categorical"]:
                    vectorizer_fn = self.feature_transforms[feature_name]
                    raw_value = sample.get(feature_name)
                    one_hot_vector = vectorizer_fn(raw_value)
                    feature_vector.extend(one_hot_vector)  # extend, not append!
            
            # Process ordinal features (normalized like numerical)
            if "ordinal" in self.feature_config:
                for feature_name in self.feature_config["ordinal"]:
                    vectorizer_fn = self.feature_transforms[feature_name]
                    raw_value = sample.get(feature_name)
                    normalized_value = vectorizer_fn(raw_value)
                    feature_vector.append(normalized_value)
            
            transformed_data.append(feature_vector)

        return np.array(transformed_data)