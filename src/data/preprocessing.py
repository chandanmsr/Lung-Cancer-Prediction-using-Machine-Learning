from src.config import TEST_SIZE, RANDOM_STATE
TEST_SIZE = 0.2
RANDOM_STATE = 42
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
        """Load and preprocess the dataset"""
        try:
            data = pd.read_csv(file_path)
            

            data.columns = data.columns.str.strip().str.upper().str.replace(' ', '_')
            

            binary_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                         'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                         'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                         'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
            
            # Converting binary columns to 0/1A
            for col in binary_cols:
                data[col] = data[col].astype(str).str.strip().map({'2': 1, '1': 0, 'YES': 1, 'NO': 0}).astype(int)
            
            # Converting gender and target to binary
            data['GENDER'] = data['GENDER'].astype(str).str.strip().str.upper().map({'M': 1, 'F': 0}).astype(int)
            data['LUNG_CANCER'] = data['LUNG_CANCER'].astype(str).str.strip().str.upper().map({'YES': 1, 'NO': 0}).astype(int)
            
            return data
        
        except Exception as e:
            print(f"\nError loading dataset: {str(e)}")
            return None

    @staticmethod
    def prepare_data(data: pd.DataFrame, test_size: float = TEST_SIZE, 
                    random_state: int = RANDOM_STATE) -> Tuple:
        """Prepare data for modeling"""
        X = data.drop(['LUNG_CANCER'], axis=1, errors='ignore')
        y = data['LUNG_CANCER']
        feature_names = X.columns.tolist()
        
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Handling class imbalance using SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test, feature_names, scaler