import sys
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import configure_environment
from src.data.preprocessing import DataPreprocessor
from src.data.visualization import DataVisualizer
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.prediction.interface import PredictionSystem

class DataExplorer:
    """Handles data exploration and visualization"""
    
    @staticmethod
    def explore_data(data: pd.DataFrame) -> None:
        """Interactive data exploration with visualizations"""
        if data is None:
            return
        
        print("\n=== DATA EXPLORATION ===")
        print(tabulate(data.describe(), headers='keys', tablefmt='psql'))
        
        viz_data = data.copy()
        viz_data['LUNG_CANCER_STR'] = viz_data['LUNG_CANCER'].map({0: 'No Cancer', 1: 'Cancer'})
        
        # Plot class distribution
        DataVisualizer.plot_class_distribution(viz_data['LUNG_CANCER'])
        
        # Plot feature distributions
        DataVisualizer.plot_distributions(viz_data, 'LUNG_CANCER_STR')
        DataVisualizer.plot_boxplots(viz_data, 'LUNG_CANCER_STR')
        DataVisualizer.plot_correlation_heatmap(data, 'LUNG_CANCER')
        
        # Plot top correlated features
        top_features = data.corr()['LUNG_CANCER'].abs().sort_values(ascending=False).index[1:6]
        sns.pairplot(data, vars=top_features, hue='LUNG_CANCER',
                    palette={0: 'dodgerblue', 1: 'crimson'},
                    diag_kind='kde', corner=True)
        plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
        plt.show()
        
        # Plot age distribution by gender
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=viz_data, x='GENDER', y='AGE', hue='LUNG_CANCER_STR',
                      split=True, palette={'No Cancer':'dodgerblue', 'Cancer':'crimson'})
        plt.title('Age Distribution by Gender and Cancer Status')
        plt.xlabel('Gender (0=Female, 1=Male)')
        plt.ylabel('Age')
        plt.legend(title='Diagnosis')
        plt.show()

def main():
    """Main execution function"""
    configure_environment()
    
    print("=== LUNG CANCER PREDICTION SYSTEM ===")
    print("Loading and preparing data...")
    
    # Load and preprocess data
    data = DataPreprocessor.load_dataset("data/raw/lcs.csv")
    if data is None:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)
    
    # Explore data
    DataExplorer.explore_data(data)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, scaler = DataPreprocessor.prepare_data(data)
    
    # Visualize PCA
    DataVisualizer.plot_pca_visualization(X_train, y_train)
    
    # Initialize and train models
    models = ModelTrainer.initialize_models(X_train, y_train)
    trained_models = ModelTrainer.train_models(models, X_train, y_train)
    
    # Evaluate models
    results = ModelEvaluator.evaluate_models(trained_models, X_train, X_test, y_train, y_test)
    
    # Main interaction loop
    while True:
        print("\nOptions:")
        print("1. Select a model and make predictions")
        print("2. Exit")
        choice = input("Enter your choice (1-2): ")
        
        if choice == '1':
            PredictionSystem.manual_prediction(trained_models, results, feature_names, scaler)
        elif choice == '2':
            print("\nExiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1 or 2")

if __name__ == "__main__":
    main()