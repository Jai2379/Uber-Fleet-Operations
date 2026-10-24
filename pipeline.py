import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

class OperationsPipeline:
    def __init__(self, model_path='models/active_model.pkl'):
        self.model_path = model_path

    def auto_retrain(self, data_path):
        print(f"initiating model rebuild on {data_path}...")
        df = pd.read_csv(data_path)

        # --- 1. DATA CLEANING & FEATURE ENGINEERING ---
        print("Engineering features and translating text to math...")
        
        # Replace the literal text "null" with actual empty values, then fill with 0
        df = df.replace('null', np.nan)
        df = df.fillna(0)

        # Target Definition
        target_col = 'Booking Status' 
        df[target_col] = df[target_col].apply(lambda x: 0 if str(x).strip().lower() in ['completed', 'success'] else 1)
        target = target_col

        # Convert Time into a numeric "Hour of the Day" (0-23)
        # This tells the model IF a specific time of day is causing cancellations
        if 'Time' in df.columns:
            df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
            df['Hour'] = df['Hour'].fillna(0) # Catch any formatting errors

        # Convert Categorical Text into Numbers (One-Hot Encoding)
        # This turns 'Vehicle Type' into multiple columns: 'Vehicle Type_eBike' (1 or 0), 'Vehicle Type_Go Sedan' (1 or 0)
        categorical_cols = ['Vehicle Type'] 
        existing_cats = [col for col in categorical_cols if col in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

        # Force numerical columns to actually be floats (fixes the "null" string corruption)
        cols_to_convert = ['Avg VTAT', 'Avg CTAT', 'Cancelled Rides by Customer']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- 2. FEATURE DISCOVERY ---
        # Now we pull the numbers, and the list will be full of valuable data
        all_numeric = df.select_dtypes(include=['number', 'bool']).columns.tolist()
        exclude = [target.lower(), target, 'id', 'rowid', 'time', 'unnamed: 0']
        features = [col for col in all_numeric if col.lower() not in exclude]
        
        X = df[features]
        y = df[target]

        print(f"analyzing {len(features)} potential features...")

        # --- 3. DYNAMIC SELECTION ---
        k_val = min(len(features), 8) # Upped to 8 since we generated more features via dummies
        selector = SelectKBest(score_func=f_classif, k=k_val)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        # ... (The rest of the SMOTE and Training code remains exactly the same) ...
        # 4. Balancing Operations Data
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_new, y)

        # 5. Training the Model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_res, y_res)

        # 6. Package Serialization
        joblib.dump({
            'model': model, 
            'features': selected_features,
            'target_name': target
        }, self.model_path)
        
        print(f"pipeline complete. model relies on: {selected_features}")

# initialize the pipeline
pipeline = OperationsPipeline()
pipeline.auto_retrain('data/ncr_ride_bookings.csv')