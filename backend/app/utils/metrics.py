"""
MLB-specific evaluation metrics
Comprehensive metrics beyond simple accuracy for production-grade models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report, 
    confusion_matrix, 
    top_k_accuracy_score,
    log_loss
)
from sklearn.calibration import calibration_curve
import warnings


class MLBMetrics:
    """
    MLB 팀 표준 평가 지표 모음
    
    단순 정확도를 넘어 비즈니스 임팩트, 확률 신뢰도, 
    구종별 성능 등을 종합적으로 평가합니다.
    
    Examples:
        >>> metrics = MLBMetrics()
        >>> report = metrics.comprehensive_report(
        ...     y_true, y_pred, y_proba, pitch_names=['FF', 'SL', 'CH']
        ... )
        >>> print(f"Top-3 Accuracy: {report['top3_accuracy']:.3f}")
    """
    
    @staticmethod
    def comprehensive_report(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        pitch_names: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        MLB 팀 표준 평가 리포트
        
        Args:
            y_true: 실제 정답 레이블 (정수 인코딩)
            y_pred: 예측 레이블 (정수 인코딩)
            y_proba: 예측 확률 (N, num_classes) 형태
            pitch_names: 구종 이름 리스트 (['FF', 'SL', 'CH', ...])
            verbose: 상세 출력 여부
            
        Returns:
            평가 지표 딕셔너리
        """
        report = {}
        
        # 1. 기본 분류 지표
        report['accuracy'] = accuracy_score(y_true, y_pred)
        report['top3_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=3)
        report['top5_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=5)
        
        # 2. Log Loss (확률 예측 품질)
        try:
            report['log_loss'] = log_loss(y_true, y_proba)
        except:
            report['log_loss'] = np.nan
        
        # 3. 구종별 상세 지표
        clf_report = classification_report(
            y_true, y_pred, 
            target_names=pitch_names,
            output_dict=True,
            zero_division=0
        )
        report['per_pitch_metrics'] = clf_report
        
        # 4. Macro/Weighted 평균
        report['macro_f1'] = clf_report['macro avg']['f1-score']
        report['weighted_f1'] = clf_report['weighted avg']['f1-score']
        
        # 5. 혼동 행렬
        report['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 6. 예측 신뢰도 통계
        max_proba = np.max(y_proba, axis=1)
        report['confidence_stats'] = {
            'mean': float(np.mean(max_proba)),
            'median': float(np.median(max_proba)),
            'std': float(np.std(max_proba)),
            'min': float(np.min(max_proba)),
            'max': float(np.max(max_proba))
        }
        
        # 출력
        if verbose:
            print("\n" + "="*60)
            print("📊 MLB EVALUATION REPORT")
            print("="*60)
            print(f"✅ Top-1 Accuracy:     {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
            print(f"✅ Top-3 Accuracy:     {report['top3_accuracy']:.4f} ({report['top3_accuracy']*100:.2f}%)")
            print(f"✅ Top-5 Accuracy:     {report['top5_accuracy']:.4f} ({report['top5_accuracy']*100:.2f}%)")
            print(f"✅ Macro F1-Score:     {report['macro_f1']:.4f}")
            print(f"✅ Weighted F1-Score:  {report['weighted_f1']:.4f}")
            print(f"✅ Log Loss:           {report['log_loss']:.4f}")
            print(f"✅ Avg Confidence:     {report['confidence_stats']['mean']:.4f}")
            print("\n📋 Per-Pitch Performance:")
            print("-" * 60)
            for pitch in pitch_names:
                if pitch in clf_report:
                    p = clf_report[pitch]['precision']
                    r = clf_report[pitch]['recall']
                    f1 = clf_report[pitch]['f1-score']
                    sup = clf_report[pitch]['support']
                    print(f"  {pitch:4s} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f} | N: {sup:>6}")
            print("="*60 + "\n")
        
        return report
    
    @staticmethod
    def calculate_expected_run_value_impact(
        df: pd.DataFrame,
        prediction_col: str = 'pred',
        actual_col: str = 'pitch_type',
        run_value_col: str = 'run_value'
    ) -> Dict:
        """
        예측 정확도가 실점에 미치는 영향 계산
        
        MLB에서 가장 중요한 지표: "모델이 팀의 실점을 얼마나 막아주는가?"
        
        Args:
            df: 예측 결과가 포함된 데이터프레임
            prediction_col: 예측 컬럼명
            actual_col: 정답 컬럼명
            run_value_col: Run Value 컬럼명
            
        Returns:
            Run Value 영향 분석 딕셔너리
            
        Example:
            >>> impact = metrics.calculate_expected_run_value_impact(results_df)
            >>> print(f"Runs saved per game: {impact['runs_saved_per_game']:.2f}")
        """
        # 예측 정확성 플래그
        df['pred_correct'] = (df[prediction_col] == df[actual_col])
        
        # Run Value 평균 계산
        rv_correct = df[df['pred_correct']][run_value_col].mean()
        rv_incorrect = df[~df['pred_correct']][run_value_col].mean()
        
        # 경기당 투구 수 (평균 150개)
        avg_pitches_per_game = 150
        
        # 정확도
        accuracy = df['pred_correct'].mean()
        
        # 경기당 임팩트
        # 정답 맞춤 투구: accuracy * rv_correct
        # 틀린 투구: (1-accuracy) * rv_incorrect
        expected_rv_per_pitch = accuracy * rv_correct + (1 - accuracy) * rv_incorrect
        
        # 베이스라인 (랜덤 예측)
        baseline_rv = df[run_value_col].mean()
        
        # 개선도
        impact_per_game = (baseline_rv - expected_rv_per_pitch) * avg_pitches_per_game
        
        return {
            'rv_when_correct': float(rv_correct),
            'rv_when_incorrect': float(rv_incorrect),
            'rv_difference': float(rv_incorrect - rv_correct),
            'accuracy': float(accuracy),
            'expected_rv_per_pitch': float(expected_rv_per_pitch),
            'baseline_rv_per_pitch': float(baseline_rv),
            'runs_saved_per_game': float(impact_per_game),
            'runs_saved_per_season': float(impact_per_game * 162),  # MLB 정규 시즌 경기 수
            'wars_equivalent': float(impact_per_game * 162 / 10)  # 대략 10 runs ≈ 1 WAR
        }
    
    @staticmethod
    def pitch_probability_calibration(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = 'quantile'
    ) -> Dict:
        """
        확률 보정도 평가 (Probability Calibration)
        
        "70% 확률이라 했을 때 실제로 70% 맞는가?"를 검증합니다.
        잘 보정된 모델은 예측 확률을 신뢰할 수 있습니다.
        
        Args:
            y_true: 실제 정답 레이블
            y_proba: 예측 확률 (N, num_classes)
            n_bins: 구간 개수
            strategy: 'uniform' or 'quantile'
            
        Returns:
            보정 분석 결과 딕셔너리
            
        Example:
            >>> calib = metrics.pitch_probability_calibration(y_true, y_proba)
            >>> print(f"Expected Calibration Error: {calib['ece']:.4f}")
        """
        # 최고 확률 구종에 대한 보정 곡선
        max_proba = np.max(y_proba, axis=1)
        y_pred = np.argmax(y_proba, axis=1)
        y_binary = (y_true == y_pred).astype(int)
        
        # Calibration Curve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob_true, prob_pred = calibration_curve(
                y_binary, max_proba, 
                n_bins=n_bins, 
                strategy=strategy
            )
        
        # Expected Calibration Error (ECE)
        # ECE = Σ |acc(bin) - conf(bin)| * (|bin| / N)
        bin_counts = np.histogram(max_proba, bins=n_bins, range=(0, 1))[0]
        bin_total = np.sum(bin_counts)
        
        if bin_total > 0:
            ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts / bin_total))
        else:
            ece = 0.0
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(prob_true - prob_pred))
        
        return {
            'calibration_curve': (prob_true.tolist(), prob_pred.tolist()),
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'n_bins': n_bins,
            'is_well_calibrated': ece < 0.1  # ECE < 0.1이면 잘 보정됨
        }
    
    @staticmethod
    def high_leverage_accuracy(
        df: pd.DataFrame,
        y_true_col: str = 'pitch_type',
        y_pred_col: str = 'pred',
        leverage_conditions: Optional[Dict] = None
    ) -> Dict:
        """
        High-Leverage 상황에서의 정확도 측정
        
        중요한 순간(득점권, 2아웃, 풀카운트 등)에서의 성능을 
        별도로 측정합니다.
        
        Args:
            df: 예측 결과 데이터프레임
            y_true_col: 정답 컬럼
            y_pred_col: 예측 컬럼
            leverage_conditions: 상황 정의 딕셔너리
            
        Returns:
            상황별 정확도 딕셔너리
        """
        if leverage_conditions is None:
            leverage_conditions = {
                'scoring_position': (df['on_2b'] == 1) | (df['on_3b'] == 1),
                'two_outs': df['outs_when_up'] == 2,
                'full_count': (df['balls'] == 3) & (df['strikes'] == 2),
                'behind_count': df['balls'] > df['strikes'],
                'ahead_count': df['strikes'] > df['balls']
            }
        
        results = {}
        
        for situation, mask in leverage_conditions.items():
            if mask.sum() == 0:
                results[situation] = {
                    'accuracy': 0.0,
                    'count': 0
                }
                continue
            
            situation_df = df[mask]
            accuracy = (situation_df[y_true_col] == situation_df[y_pred_col]).mean()
            
            results[situation] = {
                'accuracy': float(accuracy),
                'count': int(mask.sum())
            }
        
        # 전체 정확도
        overall_accuracy = (df[y_true_col] == df[y_pred_col]).mean()
        results['overall'] = {
            'accuracy': float(overall_accuracy),
            'count': len(df)
        }
        
        return results
    
    @staticmethod
    def confusion_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pitch_names: List[str],
        top_n: int = 5
    ) -> Dict:
        """
        혼동 행렬 상세 분석
        
        어떤 구종들이 자주 혼동되는지 분석하여
        모델 개선 방향을 제시합니다.
        
        Args:
            y_true: 실제 정답
            y_pred: 예측값
            pitch_names: 구종 이름 리스트
            top_n: 상위 N개 혼동 패턴 추출
            
        Returns:
            혼동 분석 결과
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # 가장 많이 혼동되는 패턴 찾기
        confusion_pairs = []
        for i in range(len(pitch_names)):
            for j in range(len(pitch_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_pitch': pitch_names[i],
                        'pred_pitch': pitch_names[j],
                        'count': int(cm[i, j]),
                        'true_total': int(cm[i].sum())
                    })
        
        # 빈도순 정렬
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        # 비율 계산
        for pair in confusion_pairs:
            pair['error_rate'] = pair['count'] / pair['true_total']
        
        return {
            'confusion_matrix': cm,
            'top_confusions': confusion_pairs[:top_n],
            'total_errors': int(np.sum(cm) - np.trace(cm))
        }
    
    @staticmethod
    def print_summary(report: Dict):
        """
        평가 리포트 요약 출력
        
        Args:
            report: comprehensive_report()의 결과
        """
        print("\n" + "🎯" + "="*58 + "🎯")
        print("           MLB MODEL EVALUATION SUMMARY")
        print("🎯" + "="*58 + "🎯\n")
        
        print("📈 ACCURACY METRICS")
        print(f"   Top-1:  {report['accuracy']*100:6.2f}%")
        print(f"   Top-3:  {report['top3_accuracy']*100:6.2f}%")
        print(f"   Top-5:  {report['top5_accuracy']*100:6.2f}%")
        
        print("\n📊 F1-SCORE")
        print(f"   Macro:    {report['macro_f1']:.4f}")
        print(f"   Weighted: {report['weighted_f1']:.4f}")
        
        print("\n🎲 CALIBRATION")
        print(f"   Log Loss:      {report['log_loss']:.4f}")
        print(f"   Avg Confidence: {report['confidence_stats']['mean']:.4f}")
        
        print("\n" + "="*60 + "\n")
