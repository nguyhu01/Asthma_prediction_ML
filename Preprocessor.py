from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

class Preprocessor(TransformerMixin):
    def __init__(self, use_scaler=True):
        """
        DataPreprocessor initialization.
        :param use_scaler: bool, whether to use StandardScaler or not
        """
        if use_scaler:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
        else:
            self.pipeline = Pipeline([
                ('passthrough', FunctionTransformer(lambda x: x))
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

