"""
MLB-grade temporal validation utilities
Prevents data leakage in time-series pitch prediction models
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class TemporalSplit:
    """Data class for temporal split configuration"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    validation_type: str  # 'holdout' or 'walk_forward'
    
    def __repr__(self):
        return (f"TemporalSplit({self.validation_type}: "
                f"Train={self.train_start}~{self.train_end}, "
                f"Test={self.test_start}~{self.test_end})")


class MLBTemporalValidator:
    """
    시계열 데이터에 적합한 검증 전략
    
    MLB 투구 데이터는 시간에 따라 변화하는 특성을 가지므로
    랜덤 분할 대신 시간 기반 분할을 사용해야 합니다.
    
    Examples:
        >>> validator = MLBTemporalValidator()
        >>> train_df, test_df = validator.create_holdout_split(
        ...     df, train_years=[2015, 2016, 2017], test_years=[2024, 2025]
        ... )
    """
    
    @staticmethod
    def create_holdout_split(
        df: pd.DataFrame, 
        train_years: List[int],
        test_years: List[int],
        date_column: str = 'game_date'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        시간 기반 고정 분할 (Hold-out Split)
        
        Args:
            df: 전체 데이터프레임
            train_years: 학습에 사용할 연도 리스트 (예: [2015, 2016, ..., 2023])
            test_years: 테스트에 사용할 연도 리스트 (예: [2024, 2025])
            date_column: 날짜 컬럼명
            
        Returns:
            (train_df, test_df) 튜플
            
        Example:
            >>> # 2015-2023년 학습, 2024-2025년 테스트
            >>> train_df, test_df = validator.create_holdout_split(
            ...     df, 
            ...     train_years=list(range(2015, 2024)),
            ...     test_years=[2024, 2025]
            ... )
        """
        # 날짜 컬럼이 datetime 타입이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 연도 컬럼 추가
        df['season'] = df[date_column].dt.year
        
        # 분할
        train_df = df[df['season'].isin(train_years)].copy()
        test_df = df[df['season'].isin(test_years)].copy()
        
        # 데이터 누수 검증
        train_max_date = train_df[date_column].max()
        test_min_date = test_df[date_column].min()
        
        if train_max_date >= test_min_date:
            print(f"⚠️ WARNING: Potential data leakage detected!")
            print(f"   Train max date: {train_max_date}")
            print(f"   Test min date: {test_min_date}")
        
        # 로그 출력
        print(f"📊 Temporal Holdout Split Results:")
        print(f"   Train: {len(train_df):,} pitches ({min(train_years)}-{max(train_years)})")
        print(f"   Test:  {len(test_df):,} pitches ({min(test_years)}-{max(test_years)})")
        print(f"   Train date range: {train_df[date_column].min()} ~ {train_df[date_column].max()}")
        print(f"   Test date range:  {test_df[date_column].min()} ~ {test_df[date_column].max()}")
        
        # season 컬럼 제거 (원본 유지)
        train_df = train_df.drop('season', axis=1)
        test_df = test_df.drop('season', axis=1)
        
        return train_df, test_df
    
    @staticmethod
    def walk_forward_validation(
        df: pd.DataFrame, 
        initial_train_years: int = 5,
        step_size: int = 1,
        date_column: str = 'game_date'
    ) -> List[TemporalSplit]:
        """
        Walk-Forward Cross Validation
        
        시계열 교차 검증의 표준 방법. 학습 데이터를 점진적으로 확장하면서
        미래 데이터로 검증합니다.
        
        Args:
            df: 전체 데이터프레임
            initial_train_years: 초기 학습 기간 (년)
            step_size: 검증 단계 크기 (년)
            date_column: 날짜 컬럼명
            
        Returns:
            TemporalSplit 객체 리스트
            
        Example:
            >>> # 5년 초기 학습 후 1년씩 검증
            >>> # 2015-2019 → test 2020
            >>> # 2015-2020 → test 2021
            >>> # 2015-2021 → test 2022
            >>> splits = validator.walk_forward_validation(df, initial_train_years=5)
        """
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        df['season'] = df[date_column].dt.year
        min_year = df['season'].min()
        max_year = df['season'].max()
        
        splits = []
        for test_year in range(min_year + initial_train_years, max_year + 1, step_size):
            split = TemporalSplit(
                train_start=str(min_year),
                train_end=str(test_year - 1),
                test_start=str(test_year),
                test_end=str(test_year),
                validation_type='walk_forward'
            )
            splits.append(split)
        
        print(f"📈 Walk-Forward CV: {len(splits)} splits created")
        for i, split in enumerate(splits, 1):
            print(f"   Fold {i}: {split}")
        
        return splits
    
    @staticmethod
    def blocked_time_series_cv(
        df: pd.DataFrame, 
        n_splits: int = 5,
        gap: int = 1000,
        date_column: str = 'game_date'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Blocked Time Series Cross Validation
        
        sklearn의 TimeSeriesSplit을 개선하여 train/test 사이에
        갭(gap)을 추가해 데이터 누수를 더욱 방지합니다.
        
        Args:
            df: 전체 데이터프레임 (시간순 정렬 필요)
            n_splits: 분할 개수
            gap: train/test 사이 갭 크기 (행 수)
            date_column: 날짜 컬럼명
            
        Returns:
            (train_df, test_df) 튜플 리스트
            
        Example:
            >>> splits = validator.blocked_time_series_cv(df, n_splits=5, gap=1000)
            >>> for i, (train, test) in enumerate(splits):
            ...     print(f"Fold {i+1}: Train={len(train)}, Test={len(test)}")
        """
        # 시간순 정렬
        df = df.sort_values(date_column).reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        splits = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            
            print(f"   Fold {i}/{n_splits}: Train={len(train_df):,}, Test={len(test_df):,}, Gap={gap}")
            
            splits.append((train_df, test_df))
        
        return splits
    
    @staticmethod
    def validate_no_leakage(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str = 'game_date'
    ) -> bool:
        """
        데이터 누수 검증
        
        학습 데이터의 최대 날짜가 테스트 데이터의 최소 날짜보다
        작은지 확인합니다.
        
        Args:
            train_df: 학습 데이터
            test_df: 테스트 데이터
            date_column: 날짜 컬럼명
            
        Returns:
            누수가 없으면 True, 있으면 False
        """
        if not pd.api.types.is_datetime64_any_dtype(train_df[date_column]):
            train_df[date_column] = pd.to_datetime(train_df[date_column])
            test_df[date_column] = pd.to_datetime(test_df[date_column])
        
        train_max = train_df[date_column].max()
        test_min = test_df[date_column].min()
        
        is_valid = train_max < test_min
        
        if is_valid:
            print(f"✅ No data leakage: Train ends at {train_max}, Test starts at {test_min}")
        else:
            print(f"❌ DATA LEAKAGE DETECTED!")
            print(f"   Train max: {train_max}")
            print(f"   Test min: {test_min}")
            print(f"   Overlap: {(train_max - test_min).days} days")
        
        return is_valid
    
    @staticmethod
    def get_train_test_indices(
        df: pd.DataFrame,
        train_years: List[int],
        test_years: List[int],
        date_column: str = 'game_date'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        연도 기반으로 train/test 인덱스 반환
        
        메모리 효율을 위해 데이터프레임을 복사하지 않고
        인덱스만 반환합니다.
        
        Args:
            df: 전체 데이터프레임
            train_years: 학습 연도 리스트
            test_years: 테스트 연도 리스트
            date_column: 날짜 컬럼명
            
        Returns:
            (train_indices, test_indices) numpy 배열 튜플
        """
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        season = df[date_column].dt.year
        
        train_mask = season.isin(train_years)
        test_mask = season.isin(test_years)
        
        train_indices = df[train_mask].index.values
        test_indices = df[test_mask].index.values
        
        print(f"📍 Indices extracted:")
        print(f"   Train: {len(train_indices):,} indices")
        print(f"   Test:  {len(test_indices):,} indices")
        
        return train_indices, test_indices
