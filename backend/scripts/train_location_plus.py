import duckdb
import pandas as pd
import xgboost as xgb
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(current_dir), "data")
DB_FILE = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "location_plus_model.pkl")

def train_location_model():
    print("🦆 Loading Data for Location+ Training...")
    con = duckdb.connect(DB_FILE, read_only=True)
    
    # Location+는 '위치'와 '상황'만 봅니다.
    query = """
        SELECT 
            pitch_type,
            plate_x, 
            plate_z, 
            balls, 
            strikes, 
            stand,
            delta_run_exp as target
        FROM pitches
        WHERE game_date >= '2022-01-01'
          AND delta_run_exp IS NOT NULL
          AND plate_x IS NOT NULL
    """
    df = con.execute(query).df()
    con.close()
    
    # 전처리
    df['pitch_type'] = df['pitch_type'].astype('category')
    df['stand'] = df['stand'].astype('category') # 좌타/우타
    
    X = df[['pitch_type', 'plate_x', 'plate_z', 'balls', 'strikes', 'stand']]
    y = df['target']
    
    print("🧠 Training XGBoost Model (This represents 'Command')...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        enable_categorical=True,
        tree_method='hist'
    )
    
    model.fit(X, y)
    
    print(f"💾 Saving Location+ Model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    train_location_model()