# model_trainer.py

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ModelTrainer:
    
    def __init__(self, X_train, X_test, y_train, y_test, model_type):
        """
        Initialize the ModelTrainer with training and test data.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)  # Scale the data
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type  # 'regressor' or 'classifier'
        self.models = []
        self.results = {}

    
    def train_regressors(self):
        """
        Train regression models and evaluate their performance.
        """
        regressors = {
            'Linear Regression': LinearRegression(),
            'Lasso Regression': Lasso(max_iter=10000),  
            'Ridge Regression': Ridge(),
            'ElasticNet Regression': ElasticNet(max_iter=10000),  
            'Support Vector Regression (SVR)': SVR(),
            'Artificial Neural Network (ANN)': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
        }
        
        for name, model in regressors.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = root_mean_squared_error(self.y_test, y_pred, squared=False)  
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[name] = {
                'Best Model': model,
                'MAE': mae,
                'RMSE': rmse,
                'R2 Score': r2
            }
            # Display results 
            print(f"{name} Results:")
            print(f"{'-'*40}")
            print(f"Mean Absolute Error (MAE)  : {mae:,.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
            print(f"R-Squared (R2 Score)     : {r2:.4f}")
            print(f"{'-'*40}")
        
       
        self.display_best_regressor()

    
    def display_best_regressor(self):
        """
        Display the best regressor based on evaluation metrics (RMSE).
        """
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        print(f"\nBest Regressor: {best_model[0]}")
        print(f"Root Mean Squared Error (RMSE): {best_model[1]['RMSE']:,.2f}")
    
    
    def train_classifiers(self):
        """
        Train classification models and evaluate their performance.
        """
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
            'Support Vector Classifier (SVC)': SVC(),
            'Artificial Neural Network (ANN)': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
        }

        # Use StratifiedKFold to ensure balanced splits
        skf = StratifiedKFold(n_splits=5)  

        for name, model in classifiers.items():
            print(f"\nTraining {name}...")
            param_grid = {}  # Define appropriate parameter grid for GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy')  
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            cr = classification_report(self.y_test, y_pred)
            
            self.results[name] = {
                'Best Model': best_model,
                'Confusion Matrix': cm,
                'Classification Report': cr
            }
            # Display classification results
            print(f"{name} Results:")
            print(f"{'-'*40}")
            print(f"Confusion Matrix:\n{cm}\n")
            print(f"Classification Report:\n{cr}")
            print(f"{'-'*40}")

        self.display_best_classifier()

    def display_best_classifier(self):
        """
        Display the best classifier based on evaluation metrics.
        """
        # Use accuracy from the classification report to determine the best model
        best_model = max(self.results.items(), key=lambda x: x[1]['Classification Report']['accuracy'])
        print(f"\nBest Classifier: {best_model[0]}")
        print(f"Accuracy: {best_model[1]['Classification Report']['accuracy']:.4f}")
