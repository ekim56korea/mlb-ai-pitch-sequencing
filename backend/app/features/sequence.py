"""
🆕 [WEEK 6] Sequence Entropy 계산
Shannon Entropy를 이용한 투구 시퀀스 예측 불가능성 측정
"""

import numpy as np
from scipy.stats import entropy


def calculate_sequence_entropy(pitch_sequence):
    """
    최근 투구 시퀀스의 Shannon Entropy 계산
    
    Args:
        pitch_sequence: 최근 N개 투구 타입 리스트 (e.g., ['FF', 'SL', 'FF', 'CH'])
    
    Returns:
        float: Shannon Entropy 값 (0.0 ~ log2(unique_types))
               - 0.0: 완전히 예측 가능 (모두 동일 구종)
               - 높을수록 예측 불가능 (다양한 구종 혼합)
    
    Examples:
        >>> calculate_sequence_entropy(['FF', 'FF', 'FF', 'FF'])
        0.0  # 완전히 예측 가능
        
        >>> calculate_sequence_entropy(['FF', 'SL', 'FF', 'SL'])
        1.0  # 50-50 분포 (log2(2) = 1.0)
        
        >>> calculate_sequence_entropy(['FF', 'SL', 'CH', 'CU'])
        2.0  # 균등 분포 (log2(4) = 2.0)
    """
    if not pitch_sequence or len(pitch_sequence) == 0:
        return 0.0
    
    # 구종별 카운트
    unique, counts = np.unique(pitch_sequence, return_counts=True)
    
    # 확률 분포
    probabilities = counts / len(pitch_sequence)
    
    # Shannon Entropy (base=2)
    return entropy(probabilities, base=2)


def batch_calculate_entropy(df, window_size=10):
    """
    DataFrame의 모든 행에 대해 롤링 윈도우로 엔트로피 계산
    
    Args:
        df: pandas DataFrame with 'pitch_type' column
        window_size: int, 엔트로피 계산에 사용할 이전 투구 개수
    
    Returns:
        np.ndarray: 각 행의 엔트로피 값
    
    Note:
        - 첫 window_size개 행은 데이터 부족으로 0.0 반환
        - game_pk와 at_bat_number가 변경되면 초기화
    """
    entropies = np.zeros(len(df), dtype=np.float32)
    
    for i in range(window_size, len(df)):
        # 이전 window_size개 투구 추출
        window = df.iloc[i-window_size:i]['pitch_type'].tolist()
        
        # 게임이 바뀌면 엔트로피 0 (시퀀스 단절)
        if df.iloc[i]['game_pk'] != df.iloc[i-1]['game_pk']:
            entropies[i] = 0.0
            continue
        
        # 타석이 바뀌면 엔트로피 재계산
        if df.iloc[i]['at_bat_number'] != df.iloc[i-1]['at_bat_number']:
            # 현재 타석 내에서만 계산 (타석 경계 고려)
            current_ab = df.iloc[i]['at_bat_number']
            same_ab_pitches = []
            for j in range(i-1, max(i-window_size-1, -1), -1):
                if df.iloc[j]['at_bat_number'] == current_ab:
                    same_ab_pitches.insert(0, df.iloc[j]['pitch_type'])
                else:
                    break
            
            if len(same_ab_pitches) > 0:
                entropies[i] = calculate_sequence_entropy(same_ab_pitches)
            else:
                entropies[i] = 0.0
        else:
            # 동일 타석 내에서 롤링 엔트로피
            entropies[i] = calculate_sequence_entropy(window)
    
    return entropies


def get_entropy_features(df, window_sizes=[5, 10, 20]):
    """
    여러 윈도우 크기로 엔트로피 계산하여 다중 피처 생성
    
    Args:
        df: pandas DataFrame with 'pitch_type', 'game_pk', 'at_bat_number'
        window_sizes: list of int, 엔트로피 계산 윈도우 크기들
    
    Returns:
        dict: {f'entropy_{size}': np.ndarray} 형태의 피처 딕셔너리
    
    Example:
        >>> features = get_entropy_features(df, window_sizes=[5, 10])
        >>> # {'entropy_5': array([...]), 'entropy_10': array([...])}
    """
    entropy_features = {}
    
    for size in window_sizes:
        entropy_features[f'entropy_{size}'] = batch_calculate_entropy(df, window_size=size)
    
    return entropy_features


# 🔧 실제 적용 예시
if __name__ == '__main__':
    import pandas as pd
    
    # 테스트 데이터
    test_df = pd.DataFrame({
        'game_pk': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        'at_bat_number': [1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
        'pitch_type': ['FF', 'FF', 'SL', 'FF', 'CH', 'FF', 'SL', 'CU', 'FF', 'SL']
    })
    
    # 단일 시퀀스 엔트로피
    sequence = ['FF', 'FF', 'SL', 'FF']
    print(f"Sequence {sequence} entropy: {calculate_sequence_entropy(sequence):.3f}")
    
    # 배치 엔트로피 계산
    entropies = batch_calculate_entropy(test_df, window_size=4)
    print(f"\nBatch entropies:\n{entropies}")
    
    # 다중 윈도우 피처
    multi_features = get_entropy_features(test_df, window_sizes=[3, 5])
    print(f"\nMulti-window features:")
    for key, values in multi_features.items():
        print(f"{key}: {values}")
