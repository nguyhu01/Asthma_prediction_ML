import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

class SeverityLogisticRegressionClassifier:
    def __init__(self, C=1.0, max_iter=100, random_state=None):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self.best_params_ = None

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [100, 200, 300]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        # Search
        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train(self, X_train, y_train):
        if self.best_params_:
            print(f"Training with best parameters: {self.best_params_}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SeverityGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        self.best_params_ = None

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        # Search
        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train(self, X_train, y_train):
        if self.best_params_:
            print(f"Training with best parameters: {self.best_params_}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SeverityXGBoostClassifier:
    def __init__(self):
        self.model = XGBClassifier()
        self.best_params_ = None

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        # Search
        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train(self, X_train, y_train):
        if self.best_params_:
            print(f"Training with best parameters: {self.best_params_}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SeverityKMeansClusterClassifier:
    def __init__(self, k=3, random_state=None):
        self.model = KMeans(n_clusters=k, random_state=random_state)
        self.best_params_ = None

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'n_clusters': [3, 5, 7],
            'max_iter': [100, 200, 300]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        # Search
        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train(self, X_train, y_train):
        if self.best_params_:
            print(f"Training with best parameters: {self.best_params_}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)    

class SeverityRandomForestClassifier:
    def __init__(self, n_estimators=100, random_state=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', random_state=random_state)
        self.best_params_ = None

    def tune_hyperparameters(self, X_train, y_train):
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        # Search
        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train(self, X_train, y_train):
        """
        Train the classifier.
        :param X_train: training features
        :param y_train: training labels
        """
        if self.best_params_:
            print(f"Training with best parameters: {self.best_params_}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions with the classifier.
        :param X_test: test features
        :return: predictions
        """
        return self.model.predict(X_test)

