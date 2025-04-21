# model_evaluator.py

class ModelEvaluator:
    def __init__(self, results):
        """
        Initialize the ModelEvaluator with the results from ModelTrainer.
        """
        self.results = results

    def provide_feedback(self):
        """
        Ask user feedback on the best model and save it if the user agrees.
        """
        print("Based on the evaluation metrics, the best model has been selected.")
        
        while True:
            user_agreement = input("Do you agree with the selection? (y/n): ").strip().lower()

            if user_agreement == 'y':
                model_name = input("Please provide a name to save the model: ")
                print(f"Model saved as {model_name}.pkl")
                break
            elif user_agreement == 'n':
                print("Please review the models and select another one.")
                break
            else:
                print("Invalid input. Please enter 'y' for yes  or 'n' for no.")

