# 🎯 MLB 프로페셔널 레벨 달성 로드맵

**프로젝트:** Pitch Commander Pro (MLB AI Pitch Sequencing)  
**목표:** MLB 현업 팀에서 실전 배포 가능한 수준으로 개선  
**타임라인:** 16주 (4개월)  
**작성일:** 2026년 5월 5일

---

## 📊 **현재 상태 평가 (Baseline Assessment)**

### **현재 시스템 수준**
- **등급:** Prototype/Educational (A-) → Production (C)
- **예상 정확도:** ~75% (과대평가 가능성 높음)
- **주요 강점:** DuckDB 기반 빅데이터 처리, Docker 컨테이너화, 풀스택 통합
- **치명적 약점:** 
  - ❌ 시계열 검증 미적용 (데이터 누수 위험)
  - ❌ 클래스 불균형 미처리
  - ❌ 비즈니스 임팩트 측정 부재

### **목표 성과 (16주 후)**
- **Top-1 Accuracy:** 72% (정확한 측정 기준)
- **Top-3 Accuracy:** 88%
- **Runs Saved per Game:** 0.2 (시즌 32 Runs ≈ 3.2 WAR)
- **Inference Latency:** < 100ms (P99)
- **기술 수준:** MLB 상위 10개 팀 수준

---

## 🔴 **Phase 1: Foundation Fix (Week 1-3)**
### **목표:** 검증 체계 재구축 및 정확한 베이스라인 측정

### **Week 1: 시계열 검증 체계 구축**

#### **Task 1.1: Temporal Train/Test Split 구현**
**문제점:**
- 현재 `sklearn.train_test_split()`으로 랜덤 분할
- 시계열 데이터에서 미래 정보가 학습에 유출되는 **Data Leakage** 발생

**해결책:**
```python
# backend/app/utils/validation.py (신규 생성)
class MLBTemporalValidator:
    @staticmethod
    def create_holdout_split(df, train_years, test_years):
        """시간 기반 고정 분할"""
        df['season'] = pd.to_datetime(df['game_date']).dt.year
        train_df = df[df['season'].isin(train_years)]
        test_df = df[df['season'].isin(test_years)]
        return train_df, test_df
    
    @staticmethod
    def walk_forward_validation(df, initial_train_years=5):
        """Walk-Forward Cross Validation
        - 2015-2019 → test 2020
        - 2015-2020 → test 2021
        - ...
        """
        # Implementation
```

**구현 단계:**
1. `backend/app/utils/` 디렉토리 생성
2. `validation.py` 파일 작성
3. `train.py` 수정: `train_test_split` 제거 → `MLBTemporalValidator` 사용
4. 테스트 실행 및 검증

**검증 기준:**
- ✅ 2024-2025 테스트 정확도가 현재 대비 10-15% 하락 (정상)
- ✅ 데이터 누수 없음 확인 (train max date < test min date)

---

#### **Task 1.2: 평가 지표 확장**
**문제점:**
- 단순 Accuracy만 측정
- MLB 비즈니스 임팩트 측정 불가

**해결책:**
```python
# backend/app/utils/metrics.py (신규 생성)
class MLBMetrics:
    @staticmethod
    def comprehensive_report(y_true, y_pred, y_proba, pitch_names):
        """MLB 팀 표준 평가 리포트"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'top3_accuracy': top_k_accuracy_score(y_true, y_proba, k=3),
            'top5_accuracy': top_k_accuracy_score(y_true, y_proba, k=5),
            'per_pitch_metrics': classification_report(...),
            'confusion_matrix': confusion_matrix(...),
        }
    
    @staticmethod
    def calculate_expected_run_value_impact(df):
        """예측 정확도가 실점에 미치는 영향"""
        df['pred_correct'] = (df['pred'] == df['pitch_type'])
        rv_correct = df[df['pred_correct']]['run_value'].mean()
        rv_incorrect = df[~df['pred_correct']]['run_value'].mean()
        
        impact_per_game = (rv_incorrect - rv_correct) * 150  # 경기당 투구 수
        return {
            'runs_saved_per_game': impact_per_game,
            'runs_saved_per_season': impact_per_game * 162
        }
```

**새로운 평가 지표:**
1. **Top-K Accuracy** (K=3, 5): 상위 예측에 정답 포함 여부
2. **구종별 Precision/Recall/F1**: 희귀 구종 성능 측정
3. **Expected Calibration Error (ECE)**: 확률 신뢰도 평가
4. **Run Value Impact**: 경기당/시즌당 실점 방어 기여도

---

### **Week 2: Class Imbalance 해결**

#### **Task 2.1: Focal Loss 구현**
**문제점:**
- 패스트볼(FF): 35%, 너클볼(KN): 0.1% → 극심한 불균형
- 희귀 구종 예측 정확도 < 5%

**해결책:**
```python
# backend/app/losses/focal_loss.py (신규 생성)
class FocalLoss(nn.Module):
    """Focal Loss: FL(pt) = -α(1-pt)^γ * log(pt)
    
    Lin et al. (2017) - "Focal Loss for Dense Object Detection"
    - γ=0: Cross Entropy와 동일
    - γ↑: 쉬운 샘플(잘 맞춤) 가중치 감소, 어려운 샘플 집중
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 클래스별 가중치
        self.gamma = gamma  # focusing parameter
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class WeightedFocalLoss(nn.Module):
    """구종별 빈도 기반 자동 가중치 계산"""
    def __init__(self, class_counts, gamma=2.0, beta=0.999):
        # Effective Number of Samples (Cui et al. 2019)
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        self.alpha = torch.FloatTensor(weights)
        self.focal = FocalLoss(alpha=self.alpha, gamma=gamma)
```

**적용 방법:**
```python
# train.py 수정
# 기존: criterion = nn.CrossEntropyLoss()
# 개선:
pitch_counts = df['pitch_type_encoded'].value_counts().sort_index().values
criterion = WeightedFocalLoss(class_counts=pitch_counts, gamma=2.0)
```

**검증 기준:**
- ✅ 희귀 구종(KN, EP) F1-Score > 0.3 (기존 ~0.05)
- ✅ 전체 Macro F1-Score 증가

---

### **Week 3: 베이스라인 재측정**

**전체 파이프라인 재학습:**
```bash
# 시계열 검증 + Focal Loss 적용
docker-compose exec backend python app/train.py \
    --validation temporal \
    --test_years 2024,2025 \
    --loss focal \
    --save_metrics baseline_v2.json

# 평가
docker-compose exec backend python app/evaluate_baseline.py
```

**산출물:**
- 📊 `baseline_report_v2.json` (개선 전후 비교표)
- 📈 Calibration curve 그래프
- 📉 구종별 Precision-Recall 곡선
- 📋 Confusion Matrix (희귀 구종 개선도 확인)

**의사결정 기준:**
- 정확도 하락(10-15%)은 **정상** → 과대평가 제거
- 희귀 구종 F1 개선 확인 → 다음 단계 진행

---

## 🟡 **Phase 2: Feature Engineering (Week 4-7)**
### **목표:** 예측 정확도 향상을 위한 고급 피처 추가

### **Week 4: 터널링 & 시퀀싱 피처**

#### **Task 3.1: 릴리스 포인트 터널링**
**배경:**
- **터널링(Tunneling)**: 초반 궤적이 동일하다가 후반에 급격히 분리되는 투구 조합
- 타자의 인식 지연 유발 → 헛스윙률 증가

**구현:**
```python
# backend/app/features/tunneling.py (신규 생성)
class TunnelingFeatures:
    @staticmethod
    def calculate_release_point_distance(df):
        """연속 투구 간 릴리스 포인트 거리"""
        df['prev_release_x'] = df.groupby(['game_pk', 'at_bat_number'])['release_pos_x'].shift(1)
        df['prev_release_z'] = df.groupby(['game_pk', 'at_bat_number'])['release_pos_z'].shift(1)
        
        distance = np.sqrt(
            (df['release_pos_x'] - df['prev_release_x'])**2 +
            (df['release_pos_z'] - df['prev_release_z'])**2
        )
        return distance.fillna(0)
    
    @staticmethod
    def calculate_trajectory_divergence(df, evaluation_point=40.0):
        """40ft 지점에서의 궤적 차이"""
        # 물리 계산으로 40ft 위치 추정
        # ...
        return divergence.fillna(0)
    
    @staticmethod
    def pitch_sequencing_patterns(df, window=5):
        """최근 N개 투구의 패턴 인코딩"""
        # 구종별 카운트
        for pitch_type in ['FF', 'SL', 'CH', 'CU']:
            df[f'{pitch_type}_count_last_{window}'] = (
                df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1)
                          .apply(lambda s: (s == pitch_type).sum()))
            )
        
        # 속구 → 변화구 전환 패턴
        df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
        df['prev_is_fastball'] = df.groupby(['game_pk', 'at_bat_number'])['is_fastball'].shift(1)
        df['fb_to_breaking'] = ((df['prev_is_fastball'] == 1) & (df['is_fastball'] == 0)).astype(int)
        
        return df
```

**새로운 피처:**
1. `tunnel_distance`: 릴리스 포인트 거리 (낮을수록 터널링 효과↑)
2. `trajectory_div`: 40ft 지점 궤적 분리도
3. `FF_count_last_5`, `SL_count_last_5`: 최근 5투구 구종 분포
4. `fb_to_breaking`: 속구→변화구 전환 횟수

---

### **Week 5-6: Batter vs Pitcher History**

#### **Task 3.2: BvP 누적 통계**
**배경:**
- 투수-타자 대결 히스토리는 MLB 스카우팅의 핵심
- "오타니는 커쇼에게 약하다" 같은 매칭 데이터

**구현:**
```python
# backend/app/features/batter_pitcher.py (신규 생성)
class BvPFeatures:
    @staticmethod
    def aggregate_career_stats(df):
        """투수-타자 과거 대전 성적 (누적)"""
        df = df.sort_values('game_date')
        bvp_group = df.groupby(['pitcher', 'batter'])
        
        # 누적 타율 (현재 행 제외)
        df['bvp_ab'] = bvp_group['pitch_number'].cumcount()
        df['bvp_hits'] = bvp_group['events'].transform(
            lambda x: x.shift(1).isin(['single', 'double', 'triple', 'home_run']).cumsum()
        )
        df['bvp_ba'] = (df['bvp_hits'] / df['bvp_ab'].replace(0, np.nan)).fillna(0.250)
        
        # 누적 헛스윙률
        df['bvp_swings'] = bvp_group['description'].transform(...)
        df['bvp_whiff_rate'] = (df['bvp_whiffs'] / df['bvp_swings']).fillna(0.24)
        
        return df
```

**외부 데이터 통합 (선택사항):**
- FanGraphs API: 타자 wOBA, wRC+
- Baseball Reference: 투수 FIP, xFIP
- MLB Statcast: Barrel%, xwOBA

---

### **Week 7: 환경 & 피로도 피처**

#### **Task 3.3: Contextual Features**
```python
# backend/app/features/contextual.py (신규 생성)
class ContextualFeatures:
    @staticmethod
    def add_environmental_features(df):
        """경기장 환경"""
        # 경기장별 고도 (Coors Field: 5200ft → 공 잘 나감)
        stadium_altitude = {
            'Coors Field': 5200,
            'Fenway Park': 20,
            # ...
        }
        df['altitude'] = df['stadium'].map(stadium_altitude).fillna(600)
        
        # 날씨 API 연동 (OpenWeatherMap)
        # df['temperature'], df['humidity'], df['wind_speed']
        
        return df
    
    @staticmethod
    def calculate_pitcher_fatigue(df):
        """투수 피로도"""
        df = df.sort_values(['pitcher', 'game_date'])
        
        # 최근 7일 투구 수
        df['pitches_last_7d'] = (
            df.groupby('pitcher')
            .rolling('7D', on='game_date')['pitch_number'].count()
            .reset_index(0, drop=True)
        )
        
        # 휴식일
        df['rest_days'] = df.groupby('pitcher')['game_date'].diff().dt.days.fillna(4)
        
        # 시즌 누적 이닝
        df['season_innings'] = df.groupby(['pitcher', df['game_date'].dt.year]).cumcount() / 3
        
        return df
```

**검증 (Feature Importance):**
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=FEATURES)
```

**예상 효과:**
- 정확도 +3-5%p (65% → 70%)
- Top-3 정확도 +5-7%p (80% → 87%)

---

## 🟢 **Phase 3: Advanced Model Architecture (Week 8-11)**
### **목표:** LSTM을 넘어선 최신 아키텍처 적용

### **Week 8-9: LSTM + Attention**

#### **Task 4.1: Multi-Head Attention 구현**
**배경:**
- 기존 LSTM: 장기 의존성 학습 제한
- Attention: 중요한 과거 투구에 집중

**구현:**
```python
# backend/app/model_attention.py (신규 생성)
class PitchLSTMAttention(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, num_layers=3, 
                 num_classes=10, num_heads=8, dropout=0.3):
        super().__init__()
        
        # 1. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # 2. Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Layer Normalization + Residual Connection
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size * 2)
        
        # 4. Feed Forward Network
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_attention=False):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        
        # Self-Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln2(attn_out + lstm_out)  # Residual
        
        # Classification
        final_hidden = attn_out[:, -1, :]
        x = self.dropout(F.relu(self.fc1(final_hidden)))
        logits = self.fc2(x)
        
        if return_attention:
            return logits, attn_weights
        return logits
```

#### **Task 4.2: Attention Visualization**
```python
# backend/app/visualize_attention.py
def plot_attention_heatmap(attention_weights, pitch_sequence):
    """어느 투구에 모델이 집중하는지 시각화"""
    sns.heatmap(
        attention_weights,
        xticklabels=pitch_sequence,  # ['FF', 'SL', 'CH', ...]
        yticklabels=pitch_sequence,
        cmap='YlOrRd',
        annot=True
    )
    plt.savefig('attention_heatmap.png')
```

**사용 사례:**
- 코치에게 "모델이 2투구 전 슬라이더를 중요하게 봤다" 설명
- 전략 수립: "상대 투수는 X 상황에서 Y 패턴 의존"

---

### **Week 10-11: Transformer 실험 (선택사항)**

```python
# backend/app/model_transformer.py
class PitchTransformer(nn.Module):
    """Pure Transformer (Vaswani et al. 2017)"""
    def __init__(self, input_size=50, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.pos_encoder(self.input_fc(x))
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)
```

**A/B Test:**
```bash
python app/compare_models.py \
    --models lstm_base,lstm_attention,transformer \
    --metric top3_accuracy,calibration_error
```

---

## 🚀 **Phase 4: Production Engineering (Week 12-14)**
### **목표:** MLB 현업 배포 가능한 인프라 구축

### **Week 12: MLOps 파이프라인**

#### **Task 5.1: Experiment Tracking (Weights & Biases)**
```python
# train.py에 추가
import wandb

wandb.init(
    project="mlb-pitch-sequencing",
    config={
        "architecture": "LSTM-Attention",
        "epochs": 50,
        "lr": 0.001,
    }
)

# 학습 중 로깅
wandb.log({
    "train_loss": loss.item(),
    "val_top3_acc": top3_acc,
    "val_calibration_error": ece
})
```

#### **Task 5.2: Model Registry**
```python
# backend/app/utils/model_manager.py
class ModelRegistry:
    def save_model(self, model, metadata):
        """버전 관리 + 메타데이터 저장"""
        version = f"v{timestamp}"
        model_dir = self.path / version
        
        torch.save(model.state_dict(), model_dir / "model.pth")
        
        metadata.update({"version": version, "created_at": timestamp})
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        self._update_registry(version, metadata)
        return version
    
    def load_best_model(self, metric='val_top3_acc'):
        """특정 지표 기준 최고 모델 로드"""
        with open(self.path / "registry.json") as f:
            registry = json.load(f)
        best = max(registry['models'], key=lambda x: x['metrics'].get(metric, 0))
        return torch.load(self.path / best['version'] / "model.pth"), best
```

---

### **Week 13: Real-time Inference API**

```python
# backend/app/api/inference.py
from fastapi import APIRouter
router = APIRouter()

@router.post("/predict")
async def predict_next_pitch(request: PredictionRequest):
    """
    MLB-grade 실시간 예측
    - Latency: < 100ms (P99)
    - Throughput: > 1000 req/s
    """
    start_time = time.time()
    
    # 1. Feature Engineering
    features = preprocess_sequence(request.sequence, request.game_context)
    
    # 2. Model Inference
    with torch.no_grad():
        logits = model(features.unsqueeze(0))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # 3. Top-K Results
    top_k = 5
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = [
        {"pitch_type": PITCH_TYPES[idx], "probability": float(probs[idx])}
        for idx in top_indices
    ]
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "predictions": predictions,
        "confidence": float(probs[top_indices[0]]),
        "model_version": model_metadata['version'],
        "latency_ms": latency
    }
```

**Load Testing:**
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000 --users=1000 --spawn-rate=100
```

---

### **Week 14: Monitoring & Alerting**

```python
# backend/app/middleware/monitoring.py
from prometheus_client import Counter, Histogram, Gauge

prediction_counter = Counter('pitch_predictions_total', 'Total predictions')
prediction_latency = Histogram('pitch_prediction_latency_seconds', 'Latency')
model_confidence = Gauge('pitch_prediction_confidence', 'Average confidence')

@app.middleware("http")
async def add_monitoring(request, call_next):
    if request.url.path == "/api/predict":
        start_time = time.time()
        response = await call_next(request)
        
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        prediction_counter.inc()
        
        return response
    return await call_next(request)
```

**Grafana Dashboard:**
```yaml
# docker-compose.yml에 추가
services:
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana
    ports: ["3001:3000"]
```

---

## 🎓 **Phase 5: Advanced Topics (Week 15-16)**

### **Week 15: Personalized Models (Transfer Learning)**

```python
# backend/app/personalization/pitcher_specific.py
class PersonalizedModelTrainer:
    def finetune_for_pitcher(self, pitcher_id, pitcher_data, freeze_layers=True):
        """투수별 파인튜닝"""
        personal_model = copy.deepcopy(self.base_model)
        
        # 하위 레이어 동결
        if freeze_layers:
            for param in personal_model.lstm.parameters():
                param.requires_grad = False
        
        # 마지막 레이어만 학습
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, personal_model.parameters()),
            lr=1e-4
        )
        
        for epoch in range(10):
            loss = train_one_epoch(personal_model, pitcher_data, optimizer)
        
        self.save_personal_model(pitcher_id, personal_model)
        return personal_model
```

---

### **Week 16: Explainable AI (SHAP)**

```python
# backend/app/explainability/shap_analysis.py
class PitchExplainer:
    def __init__(self, model, background_data):
        self.explainer = shap.DeepExplainer(model, torch.FloatTensor(background_data))
    
    def explain_prediction(self, input_sequence):
        """SHAP 값으로 예측 설명"""
        shap_values = self.explainer.shap_values(torch.FloatTensor(input_sequence))
        
        shap.force_plot(
            self.explainer.expected_value[predicted_class],
            shap_values[predicted_class],
            input_sequence,
            feature_names=FEATURE_NAMES
        )
        
        return shap_values
```

---

## 📊 **최종 검증 체크리스트**

### **MLB Production Readiness Criteria**

| Category | Metric | Target | Status |
|----------|--------|--------|--------|
| **정확도** | Top-1 Accuracy | > 65% | ⬜ |
| | Top-3 Accuracy | > 85% | ⬜ |
| | Macro F1-Score | > 0.60 | ⬜ |
| **보정** | ECE | < 0.10 | ⬜ |
| **비즈니스** | Runs Saved/Game | > 0.15 | ⬜ |
| **성능** | Latency (P99) | < 100ms | ⬜ |
| | Throughput | > 1000 req/s | ⬜ |
| **신뢰성** | Uptime (30d) | > 99.9% | ⬜ |
| **설명가능성** | SHAP Coverage | 100% | ⬜ |

---

## 🎯 **예상 성과 (16주 후)**

### **모델 성능**
- **현재 (추정):** Top-1 75% (과대평가)
- **Phase 1 후:** Top-1 62% (정확한 측정)
- **Phase 2-3 후:** Top-1 72%, Top-3 88%
- **최종:** MLB 프로덕션 레벨

### **비즈니스 임팩트**
- 경기당 **0.2 Runs Saved**
- 시즌 **32.4 Runs Saved** → **3.2 WAR**
- 팀 연봉 절감: **$9M-12M** (1 WAR ≈ $8M)

### **기술 수준**
- ✅ MLB 30개 구단 중 **상위 10개 팀 수준**
- ✅ SLOAN Sports Analytics Conference 논문 발표 가능
- ✅ Kaggle Grandmaster 포트폴리오

---

## 📚 **필수 학습 자료**

### **Papers**
1. "Focal Loss for Dense Object Detection" (Lin et al., 2017)
2. "Attention Is All You Need" (Vaswani et al., 2017)
3. "SHAP: A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
4. "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

### **MLB Analytics**
- FanGraphs Library (sabermetrics)
- Baseball Prospectus (advanced metrics)
- MLB Statcast Glossary
- Driveline Baseball R&D Blog

### **MLOps**
- "Machine Learning Engineering" by Andriy Burkov
- Google's "Rules of Machine Learning"
- "Designing Data-Intensive Applications" by Martin Kleppmann

---

## 🔄 **진행 상황 트래킹**

### **Week 1 (현재주)**
- [ ] `backend/app/utils/validation.py` 생성
- [ ] `backend/app/utils/metrics.py` 생성
- [ ] `train.py` 시계열 검증 적용
- [ ] 베이스라인 재측정

### **주간 체크인**
- 매주 금요일: 진행 상황 리뷰
- 매 Phase 종료: 성과 측정 및 문서화
- 이슈 발생 시: 우선순위 재조정

---

**이 로드맵을 완수하면 당신의 프로젝트는 MLB 프론트 오피스에서 실제로 사용할 수 있는 수준이 됩니다. 🚀⚾️**
