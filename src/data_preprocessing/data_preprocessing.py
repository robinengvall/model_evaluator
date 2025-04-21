# data_preprocessing.py

import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    
    def __init__(self, file_path: str):
        """
        Initialize the DataPreprocessor with the path to the CSV file.
        """
        self.file_path = file_path
        self.df = None  

    
    def fill_missing_data(self):
        """
        Fill missing data in the DataFrame.
        """
        if self.df is not None:
            # Example: Fill numeric columns with the mean and categorical with mode
            for column in self.df.columns:
                if self.df[column].dtype == 'O':  # Object type (categorical)
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:  # Numeric type
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
            print("Missing data handled.")
        else:
            print("No data to handle missing values.")

    
    def load_data(self):
        """
        Load data from a CSV file. Validates if the 
        file path is correct and if the file is a valid CSV.
        """
        if not os.path.exists(self.file_path):
            print(f"Error: The file '{self.file_path}' does not exist.")
            return False

        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            return True
        
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            return False

    
    def validate_data(self):
        """
        Validates the loaded data for basic correctness.
        """
        if self.df is None:
            print("No data loaded to validate.")
            return False

        # Checking for missing values
        missing_values = self.df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        
        if not missing_columns.empty:
            print("The following columns have missing values:")
            print(missing_columns)
            print("Please fill in missing data or handle it appropriately.")
            return False
        else:
            print("No missing values found in the data.")

        # Checking for sufficient data
        if self.df.shape[0] < 10 or self.df.shape[1] < 2:
            print("Warning: The dataset seems to have insufficient rows or columns for ML processing.")
            return False
        
        print("Data validation passed.")
        return True

    
    def check_data_readiness(self, fill_missing='no', convert_categorical='no'):
        """
        Checks if the data is ready for machine learning. 
        Handles missing data and converts categorical data to dummy variables.
        """
        # Handle missing values based on the parameter
        if self.df.isnull().values.any():
            print("Data is not ready: Missing values found.")
            print("Columns with missing values:")
            print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
            
            if fill_missing == 'yes':
                self.fill_missing_data()
                print("Missing values filled.")
            elif fill_missing == 'no':
                print("Please handle missing values and try again.")
                return False

        
        # Convert categorical columns to dummy variables based on the parameter
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"Data contains categorical/string columns: {list(categorical_columns)}")
            if convert_categorical == 'yes':
                self.df = pd.get_dummies(self.df, columns=categorical_columns)
                print("Categorical columns converted to dummy variables.")
            elif convert_categorical == 'no':
                print("Please convert categorical data and try again.")
                return False

        print("Data is ready for the machine learning process.")
        return True

    
    def get_target_variable(self):
        """
        Displays column names and allows the user to select the dependent variable (target y).
        Validates if the selected variable is continuous or categorical.
        """
        # Normalize column names to lowercase for comparison
        columns = {col.lower(): col for col in self.df.columns}
        print("Available columns in the dataset:")
        print(self.df.columns.tolist())
        
        while True:
            # Normalize user input by stripping whitespace and converting to lowercase
            target_column_input = input("Please enter the column name for the target variable (y): ").strip().lower()
            
            if target_column_input not in columns:
                print(f"Error: '{target_column_input}' is not a valid column name. Please try again.")
            else:
                # Use the original column name from the DataFrame
                target_column = columns[target_column_input]
                
                # Check if the target variable is continuous or categorical
                if pd.api.types.is_numeric_dtype(self.df[target_column]):
                    print(f"The target variable '{target_column}' is continuous (numeric).")
                else:
                    print(f"The target variable '{target_column}' is categorical.")
                return target_column
    
    