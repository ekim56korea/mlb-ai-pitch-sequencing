import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_ai():
    print("🧠 AI 모델 학습을 시작합니다...")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv('savant_data.csv')
        print(f"📊 데이터 로드 완료: {len(df)}개의 투구 데이터")
    except FileNotFoundError:
        print("❌ 'savant_data.csv' 파일이 없습니다.")
        return

    # 2. 학습에 필요한 컬럼 선택 (Feature Engineering)
    # 입력(문제): 투수이름, 타자유형(좌/우), 볼, 스트라이크, 아웃, 이닝, 주자상황
    # 출력(정답): 구종(pitch_type)
    features = ['player_name', 'stand', 'balls', 'strikes', 'outs_when_up', 'inning']
    target = 'pitch_type'

    # 결측치 제거
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    print("⚙️ 데이터 전처리 및 파이프라인 구축 중...")

    # 3. 데이터 전처리 파이프라인
    # 숫자 데이터: 그대로 사용
    # 문자 데이터(투수이름, 타자유형): 숫자로 변환 (One-Hot Encoding)
    numeric_features = ['balls', 'strikes', 'outs_when_up', 'inning']
    categorical_features = ['player_name', 'stand']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. 모델 정의 (Random Forest Classifier)
    model = make_pipeline(
        preprocessor,
        RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    )

    # 5. 학습 (Training)
    print("💪 학습 시작 (데이터 양에 따라 1~2분 소요될 수 있습니다)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # 6. 평가
    score = model.score(X_test, y_test)
    print(f"✅ 모델 학습 완료! 정확도: {score:.2%}")

    # 7. 모델 저장
    joblib.dump(model, 'pitch_predictor.pkl')
    print("💾 'pitch_predictor.pkl' 파일로 두뇌를 저장했습니다.")

if __name__ == "__main__":
    train_ai()