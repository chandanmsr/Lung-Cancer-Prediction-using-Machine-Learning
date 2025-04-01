# src/prediction/interface.py
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler

class PredictionSystem:
    """Handles interactive prediction functionality"""
    
    @staticmethod
    def manual_prediction(models: Dict, results: Dict, feature_names: List[str], 
                         scaler: StandardScaler = None) -> None:
        """Interactive function for manual model selection and prediction"""
        print("\n=== MANUAL PREDICTION ===")
        
        print("\nAvailable Models (with accuracy scores):")
        for i, (name, model) in enumerate(models.items(), 1):
            accuracy = results[name]['accuracy'] if name in results else "Not evaluated"
            print(f"{i}. {name} (Accuracy: {accuracy:.1%})" if isinstance(accuracy, float) else f"{i}. {name} (Accuracy: {accuracy})")
        
        while True:
            try:
                choice = input("\nSelect a model (1-{}): ".format(len(models)))
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    selected_model = list(models.values())[int(choice)-1]
                    model_name = list(models.keys())[int(choice)-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            case_data = {}
            print(f"\nMaking prediction using {model_name}")
            print("Enter patient details:")
            
            case_data['GENDER'] = 1 if input("Gender (M/F): ").strip().upper() == 'M' else 0
            case_data['AGE'] = int(input("Age: "))
            
            symptoms = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                      'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                      'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                      'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
            
            for symptom in symptoms:
                while True:
                    try:
                        val = input(f"{symptom.replace('_', ' ')} (0=No, 1=Yes): ")
                        case_data[symptom] = int(val)
                        if case_data[symptom] in (0, 1):
                            break
                        print("Please enter 0 or 1")
                    except ValueError:
                        print("Invalid input. Please enter 0 or 1.")
            
            case_df = pd.DataFrame([case_data])[feature_names]
            
            if scaler:
                case_scaled = scaler.transform(case_df)
            else:
                case_scaled = case_df.values
            
            try:
                prediction = selected_model.predict(case_scaled)[0]
                proba = selected_model.predict_proba(case_scaled)[0][1] if hasattr(selected_model, "predict_proba") else None
                
                print("\n=== PREDICTION RESULT ===")
                print(f"Prediction: {'Cancer' if prediction == 1 else 'No Cancer'}")
                if proba is not None:
                    print(f"Probability: {proba:.1%}")
                    PredictionSystem._print_recommendation(prediction, proba)
                
                try:
                    PredictionSystem._print_feature_importance(selected_model, case_data, feature_names)
                except Exception as e:
                    print(f"\nCould not display feature importance: {str(e)}")
                
            except Exception as e:
                print(f"\nPrediction error: {str(e)}")
            
            if input("\nPredict another case with this model? (y/n): ").lower() != 'y':
                break

    @staticmethod
    def _print_recommendation(prediction: int, probability: float) -> None:
        """Print recommendation based on prediction probability"""
        if prediction == 1:
            if probability > 0.7:
                print("Recommendation: Immediate medical consultation recommended")
            elif probability > 0.5:
                print("Recommendation: Schedule doctor appointment")
            else:
                print("Recommendation: Monitor symptoms")
        else:
            if probability > 0.3:
                print("Recommendation: Consider follow-up tests")
            else:
                print("Recommendation: Low risk detected")

    @staticmethod
    def _print_feature_importance(model: Any, case_data: Dict, 
                                feature_names: List[str]) -> None:
        """Print top contributing features if available"""
        if not hasattr(model, 'feature_importances_'):
            return
            
        print("\nKey contributing factors:")
        importances = model.feature_importances_
        top_features = pd.Series(importances, index=feature_names)
        top_features = top_features.sort_values(ascending=False).head(3)
        
        for feat, imp in top_features.items():
            val = case_data[feat]
            feat_name = feat.replace('_', ' ').title()
            print(f"- {feat_name}: {'Present' if val == 1 else 'Absent'} (Importance: {imp:.2f})")