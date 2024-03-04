import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from DataLoader import DataLoader
from Preprocessor import Preprocessor
from DataAugmentor import DataAugmentor

from models import SeverityRandomForestClassifier
from models import SeverityKMeansClusterClassifier
from models import SeverityGradientBoostingClassifier
from models import SeverityXGBoostClassifier
from models import SeverityLogisticRegressionClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SeverityClassificationPipeline:
    def __init__(self, file_path, test_size=0.2, random_state=42, learning_rate= None, max_depth=None):
        """
        Initialize the pipeline.
        :param file_path: Path to the CSV file.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random state for reproducibility.
        """
        self.data_loader = DataLoader(file_path, test_size, random_state)
        self.preprocessor = Preprocessor()
        self.augmentor = DataAugmentor(noise_level=0.2)
        # self.classifier = SeverityRandomForestClassifier(n_estimators =100, random_state=random_state)
        # self.classifier = SeverityXGBoostClassifier()
        # self.classifier = SeverityGradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=None, random_state=None)
        self.classifier = SeverityLogisticRegressionClassifier(max_iter = 100, random_state = random_state)

        self.train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        self.test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    def run_augmented_data(self):                                                                                
        """                                                                                       
        Run the pipeline: load data, augment features, preprocess, train and evaluate the model.  
        """                                                                                       
        self.data_loader.load_data()                                                              

        X_train, X_test, y_train, y_test = self.data_loader.get_data()

        print("Shape of X_train before is:", X_train.shape)
        print("Train Class labels 0 before augmentation:", len(y_train[y_train == 0]))
        print("Train Class labels 1 before augmentation:", len(y_train[y_train == 1]))

        # Augment 
        print("------------ Augmenting Features ------------")
        
        X_train_augmented = self.augmentor.add_noise(X_train)
        X_train_combined = np.concatenate((X_train, X_train_augmented), axis = 0)
        print("Shape of X_train after is:", X_train_combined.shape)

        y_train_combined = np.ravel(np.concatenate((y_train, y_train), axis = 0))

        print("Train Class labels 0 after augmentation:", len(y_train_combined[y_train_combined == 0]))
        print("Train Class labels 1 after augmentation:", len(y_train_combined[y_train_combined == 1]))

        # Preprocess the data
        X_train_preprocessed = self.preprocessor.fit_transform(X_train_combined)
        X_test_preprocessed = self.preprocessor.transform(X_test)

        # Tune hyperparameters
        print("------------ Tuning Hyperparameters ------------")
        self.classifier.tune_hyperparameters(X_train_preprocessed, y_train_combined)

        # Train the model
        print("------------ Training ------------")
        self.classifier.train(X_train_preprocessed, y_train_combined)

        # Predict and evaluate on training data
        train_predictions = self.classifier.predict(X_train_preprocessed)
        print("|Training Data Evaluation:|\n")
        self.evaluate_model(y_train_combined, train_predictions, mode='train')

        # Predict and evaluate on test data
        test_predictions = self.classifier.predict(X_test_preprocessed)
        print("\n|Test Data Evaluation:|\n")
        self.evaluate_model(y_test, test_predictions, mode='test')  

        print( "class 0:",len(test_predictions[test_predictions==0]))
        print( "class 1:",len(test_predictions[test_predictions==1]))

        self.plot_metrics() 

    def run(self):
        """
        Run the pipeline: load data, preprocess, train and evaluate the model.
        """
        self.data_loader.load_data()
        X_train, X_test, y_train, y_test = self.data_loader.get_data()
        
        # Preprocess the data
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        # Tune hyperparameters
        print("------------ Tuning Hyperparameters ------------")
        self.classifier.tune_hyperparameters(X_train, y_train)

        # Train the model
        print("------------ Training ------------")
        self.classifier.train(X_train, y_train)

        # Predict and evaluate on training data
        train_predictions = self.classifier.predict(X_train)
        print("\n|Training Data Evaluation:|\n")
        self.evaluate_model(y_train, train_predictions, mode='train')

        # Predict and evaluate on test data
        test_predictions = self.classifier.predict(X_test)
        print("\n|Test Data Evaluation:|\n")
        self.evaluate_model(y_test, test_predictions, mode='test')

        print( "class 0:",len(test_predictions[test_predictions==0]))
        print( "class 1:",len(test_predictions[test_predictions==1]))

        self.plot_metrics()
       

    def evaluate_model(self, y_true, y_pred, mode=None):
        """
        Evaluate the model using various metrics.
        :param y_true: Actual labels.
        :param y_pred: Predicted labels.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"{mode.capitalize()} Accuracy: {accuracy:.4f}")
        print(f"{mode.capitalize()} Precision: {precision:.4f}")
        print(f"{mode.capitalize()} Recall: {recall:.4f}")
        print(f"{mode.capitalize()} F1 Score: {f1:.4f}")

        if mode == 'train':
            self.train_metrics['accuracy'].append(accuracy)
            self.train_metrics['precision'].append(precision)
            self.train_metrics['recall'].append(recall)
            self.train_metrics['f1'].append(f1)
        elif mode == 'test':
            self.test_metrics['accuracy'].append(accuracy)
            self.test_metrics['precision'].append(precision)
            self.test_metrics['recall'].append(recall)
            self.test_metrics['f1'].append(f1)


    def plot_metrics(self):
        """
        Plot train and test metrics.
        """
        # Prepare data for plotting
        train_data = pd.DataFrame({'Iterations': range(1, len(self.train_metrics['accuracy']) + 1),
                                'Accuracy': self.train_metrics['accuracy'],
                                'F1 Score': self.train_metrics['f1'],
                                'Type': 'Train'})
        test_data = pd.DataFrame({'Iterations': range(1, len(self.test_metrics['accuracy']) + 1),
                                'Accuracy': self.test_metrics['accuracy'],
                                'F1 Score': self.test_metrics['f1'],
                                'Type': 'Test'})
        combined_data = pd.concat([train_data, test_data])

        # Plot
        plt.figure(figsize=(12, 5))
        sns.barplot(data=combined_data, x='Iterations', y='Accuracy', hue='Type', palette='pastel')
        plt.title('Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend(title='Type', loc='upper right')

        plt.figure(figsize=(12, 5))
        sns.barplot(data=combined_data, x='Iterations', y='F1 Score', hue='Type', palette='pastel')
        plt.title('F1 Score')
        plt.xlabel('Iterations')
        plt.ylabel('F1 Score')
        plt.legend(title='Type', loc='upper right')

        plt.tight_layout()
        plt.show()





