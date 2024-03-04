from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self):
        """
        DataPreprocessor initialization.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

    def fit_transform(self, X):
        """
        Fit the preprocessing pipeline to the data and transform it.
        :param X: DataFrame, features to be transformed
        :return: transformed features
        """
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        """
        Transform the data using the fitted pipeline.
        :param X: DataFrame, features to be transformed
        :return: transformed features
        """
        return self.pipeline.transform(X)

