# main.py
from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.model_training.model_trainer import ModelTrainer
from src.model_evaluation.model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    while True:  # Outer loop to restart if needed
        # Step 1: Ask the user for the type of supervised ML they need (regressor or classifier)
        while True:
            ml_type = input("Do you want to find a regressor or a classifier? (Enter 'regressor' or 'classifier'): ").strip().lower()
            if ml_type in ['regressor', 'classifier']:
                break
            else:
                print("Invalid input. Please enter either 'regressor' or 'classifier'.")

        # Step 2: Read in and validate the path to the CSV file
        while True:
            file_path = input("Please enter the path to your CSV file: ").strip()
            data_preprocessor = DataPreprocessor(file_path)

            # Attempt to load data; if it fails, prompt the user again
            if data_preprocessor.load_data():
                break
            else:
                print("Failed to load the data. Please check the file path and format and try again.")

        # Step 3: Validate the data
        if not data_preprocessor.validate_data():
            print("Data validation failed. Please address the issues and try again.")
            return

        # Check if the data is ready for the machine learning process
        if not data_preprocessor.check_data_readiness():
            print("Data is not ready for machine learning.")
            return

        # Step 4: Read in and validate the dependent target (y)
        while True:
            target_column = data_preprocessor.get_target_variable()
            if not target_column:
                print("Failed to select a valid target variable.")
                continue
            
            # Validate the target variable type
            is_numeric = pd.api.types.is_numeric_dtype(data_preprocessor.df[target_column])
            
            if ml_type == 'regressor' and not is_numeric:
                print(f"The selected target variable '{target_column}' is not numeric. Please select a numeric target variable for regression.")
                continue  # Ask the user to select a new target variable
            elif ml_type == 'classifier' and is_numeric:
                print(f"The selected target variable '{target_column}' is numeric. Please select a categorical target variable for classification.")
                # Offer to go back to the start
                go_back = input("Do you want to go back and choose 'regressor' or 'classifier' again? (y/n): ").strip().lower()
                if go_back == 'y':
                    break  # Exit the inner loop and go back to the start
                elif go_back == 'n':
                    # Convert to categorical if needed
                    while True:
                        convert_to_categorical = input("Do you want to convert this numeric target to categorical classes? (y/n): ").strip().lower()
                        if convert_to_categorical == 'y':
                            try:
                                bins = int(input("Enter the number of bins to categorize the target variable (e.g., 3 for 'low', 'medium', 'high'): ").strip())
                                labels = [f"Class_{i+1}" for i in range(bins)]
                                data_preprocessor.df[target_column] = pd.cut(data_preprocessor.df[target_column], bins=bins, labels=labels)
                                print(f"Target variable '{target_column}' has been converted to categorical classes: {labels}.")
                                break  # Exit the inner loop
                            except ValueError:
                                print("Invalid input. Please enter a valid number of bins.")
                        elif convert_to_categorical == 'no':
                            print("Please select a different categorical target variable for classification.")
                            break  # Exit the inner loop
                        else:
                            print("Invalid input. Please enter 'yes' or 'no'.")
                    
                    if not pd.api.types.is_numeric_dtype(data_preprocessor.df[target_column]):  # Re-check after conversion
                        break  # If successfully converted, break the outer loop
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            else:
                break  # If the correct type is chosen, break the loop

        # Prepare data for training
        X = data_preprocessor.df.drop(columns=[target_column])
        y = data_preprocessor.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Train models and evaluate
        model_trainer = ModelTrainer(X_train, X_test, y_train, y_test, ml_type)
        
        if ml_type == 'regressor':
            model_trainer.train_regressors()
        else:
            model_trainer.train_classifiers()

        # Step 6: Provide user feedback and save the model
        model_evaluator = ModelEvaluator(model_trainer.results)
        model_evaluator.provide_feedback()

        # After completing the process, ask if the user wants to run another round
        restart = input("Do you want to start again with a new task? (yes/no): ").strip().lower()
        if restart != 'yes':
            print("Exiting the program. Goodbye!")
            break  # Exit the outer loop and end the program

if __name__ == "__main__":
    main()
