import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np

class DataLoader:
    def __init__(self, filepath, test_size=0.2, random_state=None):
        """
        DataLoader initialization.
        :param filepath: str, path to the CSV file.
        :param test_size: float, the proportion of the dataset to include in the test split.
        :param random_state: int, controls the shuffling applied to the data before applying the split.
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Loads data from the CSV file and splits it into training and testing datasets.
        """
        
        data = pd.read_csv(self.filepath)

        # Extract the feature columns.
        X = data.iloc[:, :-3].to_numpy()

        # Extract the target column
        y = data.iloc[:, -1:].to_numpy()
        print("Unique values in target column are:", np.unique(y))
        
        # Oversample class 1
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=self.random_state)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)

        # Perform stratified sampling on the resampled data
        X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, 
                                                                                                      test_size=self.test_size, 
                                                                                                      random_state=self.random_state, 
                                                                                                      stratify=y_resampled)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled

        """
        # Find indices of samples belonging to class 1 (minority class)
        minority_indices = np.where(y == 1)[0]

        # Calculate the number of samples in the minority class
        minority_samples = len(minority_indices)

        # Find indices of samples belonging to class 0 (majority class)
        majority_indices = np.where(y == 0)[0]

        # Randomly select samples from the minority class with replacement to match the majority class
        oversampled_minority_indices = np.random.choice(minority_indices, size=len(majority_indices) - minority_samples, replace=True)

        # Concatenate the original and oversampled minority class indices
        oversampled_indices = np.concatenate((minority_indices, oversampled_minority_indices))

        # Use the oversampled indices to create the oversampled dataset
        X_oversampled = X[oversampled_indices]
        y_oversampled = y[oversampled_indices]

        # Perform stratified sampling on the resampled data
        X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_oversampled, y_oversampled, 
                                                                                                      test_size=self.test_size, 
                                                                                                      random_state=self.random_state, 
                                                                                                      stratify=y_oversampled)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled

        """


    def get_data(self):
        """
        Returns the training and testing data.
        :return: X_train, X_test, y_train, y_test
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        return self.X_train, self.X_test, self.y_train, self.y_test

