# src/models/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List

class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    @staticmethod
    def plot_feature_importance(models: Dict, X_test: np.ndarray, 
                              y_test: pd.Series, feature_names: List[str] = None) -> None:
        """Plot feature importance for tree-based models"""
        tree_models = {name: model for name, model in models.items() 
                      if hasattr(model, 'feature_importances_')}
        
        if not tree_models:
            return
        
        plt.figure(figsize=(15, 8))
        for i, (name, model) in enumerate(tree_models.items(), 1):
            plt.subplot(2, 2, i)
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]
            
            if feature_names is not None:
                feature_labels = [feature_names[i] for i in indices]
            else:
                feature_labels = indices
                
            plt.barh(range(len(indices)), importances[indices], color='dodgerblue')
            plt.yticks(range(len(indices)), feature_labels)
            plt.title(f'{name} Feature Importance')
        
        plt.subplot(2, 2, 4)
        for name, model in models.items():
            result = permutation_importance(model, X_test, y_test, n_repeats=10, 
                                          random_state=42)
            sorted_idx = result.importances_mean.argsort()[-10:]
            
            if feature_names is not None:
                feature_labels = [feature_names[i] for i in sorted_idx]
            else:
                feature_labels = sorted_idx
                
            plt.barh(range(len(sorted_idx)), 
                    result.importances_mean[sorted_idx], 
                    alpha=0.6, label=name)
        
        plt.yticks(range(len(sorted_idx)), feature_labels)
        plt.title('Permutation Importance (Top 10)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_model_performance_comparison(results: Dict) -> None:
        """Compare all models' performance metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(results.keys())
        
        perf_df = pd.DataFrame({
            'Model': models,
            'Accuracy': [results[m]['accuracy'] for m in models],
            'Precision': [precision_score(results[m]['y_true'], results[m]['y_pred']) for m in models],
            'Recall': [recall_score(results[m]['y_true'], results[m]['y_pred']) for m in models],
            'F1': [f1_score(results[m]['y_true'], results[m]['y_pred']) for m in models],
            'ROC AUC': [auc(results[m]['fpr'], results[m]['tpr']) for m in models],
            'PR AUC': [results[m]['pr_auc'] for m in models]
        })
        
        melt_df = perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(15, 6))
        sns.barplot(data=melt_df, x='Model', y='Score', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1.1)
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.show()
        
        plt.figure(figsize=(10, 10))
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'PR AUC']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
        plt.ylim(0, 1)
        
        for model in models:
            values = perf_df[perf_df['Model'] == model].values[0][1:].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', 
                   label=model, marker='o')
        
        plt.title('Model Performance Radar Chart', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()

    @staticmethod
    def plot_calibration_curves(models: Dict, X_test: np.ndarray, 
                              y_test: pd.Series) -> None:
        """Plot calibration curves for all models"""
        plt.figure(figsize=(10, 10))
        
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                prob_pos = model.predict_proba(X_test)[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, prob_pos, n_bins=10)
                
                plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                        label=f"{name}")
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title('Calibration Plots (Reliability Curves)')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def evaluate_models(models: Dict, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Evaluate multiple classification models"""
        results = {}
        
        print("\n=== MODEL EVALUATION ===")
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer'], zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            fpr, tpr, _ = roc_curve(y_test, y_proba) if y_proba is not None else (None, None, None)
            roc_auc = auc(fpr, tpr) if fpr is not None else None
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba) if y_proba is not None else (None, None, None)
            pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'y_proba': y_proba,
                'y_true': y_test,
                'y_pred': y_pred,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall,
                'pr_auc': pr_auc
            }
            
            print(f"\n{name} Performance:")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"ROC AUC: {roc_auc:.1%}" if roc_auc is not None else "ROC AUC: Not available")
            print(f"PR AUC: {pr_auc:.1%}" if pr_auc is not None else "PR AUC: Not available")
            print(report)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Cancer', 'Cancer'], 
                       yticklabels=['No Cancer', 'Cancer'])
            plt.title(f'{name} Confusion Matrix\nAccuracy: {accuracy:.3f}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        
        ModelEvaluator.plot_model_performance_comparison(results)
        ModelEvaluator.plot_calibration_curves(models, X_test, y_test)
        
        return results