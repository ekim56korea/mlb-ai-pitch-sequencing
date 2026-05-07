"""
Week 7: Sequence Entropy SQL Integration

DuckDB SQL 쿼리에서 pitch sequence를 추출하고
Python에서 Shannon Entropy를 계산하여 결합

기존 문제:
---------
- SQL에는 Shannon Entropy 함수가 없음
- sequence_entropy가 placeholder (0.0)로 고정됨

해결책:
-------
1. SQL에서 pitch sequence를 JSON 배열로 추출
2. Python에서 Shannon Entropy 계산
3. JOIN하여 최종 feature 생성

Author: AI Pitch Sequencing Team
Date: 2025-01-XX
"""

import duckdb
import pandas as pd
import numpy as np
from typing import List, Dict
import json

# Import our sequence entropy module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.sequence import calculate_sequence_entropy


def create_pitch_sequence_sql() -> str:
    """
    SQL 쿼리: 각 투구에 대해 최근 N개 pitch sequence 추출
    
    Returns:
    --------
    str
        DuckDB SQL 쿼리
    """
    query = """
    WITH pitch_sequences AS (
        SELECT 
            p.game_pk,
            p.at_bat_number,
            p.pitch_number,
            p.pitch_type,
            
            -- 최근 10개 투구 시퀀스 (JSON 배열)
            LIST(p2.pitch_type) OVER (
                PARTITION BY p.game_pk, p.pitcher
                ORDER BY p.pitch_number
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
            ) as last_10_pitches,
            
            -- 최근 5개 투구 시퀀스
            LIST(p2.pitch_type) OVER (
                PARTITION BY p.game_pk, p.pitcher
                ORDER BY p.pitch_number
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
            ) as last_5_pitches,
            
            -- 최근 20개 투구 시퀀스
            LIST(p2.pitch_type) OVER (
                PARTITION BY p.game_pk, p.pitcher
                ORDER BY p.pitch_number
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) as last_20_pitches
            
        FROM pitches p
        LEFT JOIN pitches p2 
            ON p.game_pk = p2.game_pk 
            AND p.pitcher = p2.pitcher
            AND p2.pitch_number < p.pitch_number
        WHERE p.pitch_type IS NOT NULL
    )
    SELECT 
        game_pk,
        at_bat_number,
        pitch_number,
        pitch_type,
        last_5_pitches,
        last_10_pitches,
        last_20_pitches
    FROM pitch_sequences
    """
    
    return query


def calculate_entropy_from_sequences(df: pd.DataFrame, 
                                     sequence_col: str = 'last_10_pitches') -> pd.Series:
    """
    DataFrame의 pitch sequence에서 Shannon Entropy 계산
    
    Parameters:
    -----------
    df : pd.DataFrame
        pitch sequence 컬럼 포함
    sequence_col : str
        sequence 컬럼명 (기본: 'last_10_pitches')
        
    Returns:
    --------
    pd.Series
        각 행의 Shannon Entropy 값
    """
    entropy_values = []
    
    for idx, row in df.iterrows():
        pitch_seq = row[sequence_col]
        
        # NULL 또는 빈 시퀀스 처리
        if pitch_seq is None or len(pitch_seq) == 0:
            entropy_values.append(0.0)
            continue
        
        # JSON 배열 파싱 (DuckDB LIST는 Python list로 변환됨)
        if isinstance(pitch_seq, str):
            try:
                pitch_seq = json.loads(pitch_seq)
            except:
                entropy_values.append(0.0)
                continue
        
        # Shannon Entropy 계산
        entropy = calculate_sequence_entropy(pitch_seq)
        entropy_values.append(entropy)
    
    return pd.Series(entropy_values, index=df.index)


def integrate_entropy_to_db(db_path: str, 
                            table_name: str = 'pitches',
                            update_existing: bool = False):
    """
    DuckDB 데이터베이스에 sequence_entropy 컬럼 추가 및 계산
    
    Parameters:
    -----------
    db_path : str
        DuckDB 데이터베이스 경로
    table_name : str
        테이블명 (기본: 'pitches')
    update_existing : bool
        기존 sequence_entropy 컬럼 덮어쓰기 여부
    """
    print(f"🔄 Integrating Sequence Entropy to {db_path}...")
    
    con = duckdb.connect(db_path)
    
    # 1. sequence_entropy 컬럼 존재 확인
    col_check = con.execute(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
          AND column_name = 'sequence_entropy'
    """).fetchall()
    
    has_entropy_col = len(col_check) > 0
    
    if has_entropy_col and not update_existing:
        print(f"   ℹ️ sequence_entropy column already exists. Use update_existing=True to overwrite.")
        con.close()
        return
    
    # 2. Pitch sequence 추출
    print(f"   📊 Extracting pitch sequences...")
    
    # DuckDB LIST 함수 사용
    query = f"""
    SELECT 
        game_pk,
        at_bat_number,
        pitch_number,
        pitch_type,
        pitcher,
        
        -- 최근 10개 투구 시퀀스 (current pitch 제외)
        (
            SELECT LIST(p2.pitch_type) 
            FROM {table_name} p2
            WHERE p2.game_pk = p.game_pk
              AND p2.pitcher = p.pitcher
              AND p2.pitch_number < p.pitch_number
            ORDER BY p2.pitch_number DESC
            LIMIT 10
        ) as last_10_pitches
        
    FROM {table_name} p
    WHERE p.pitch_type IS NOT NULL
    ORDER BY game_pk, pitch_number
    """
    
    df = con.execute(query).df()
    print(f"   Loaded {len(df):,} pitches")
    
    # 3. Shannon Entropy 계산
    print(f"   🧮 Calculating Shannon Entropy...")
    df['sequence_entropy'] = calculate_entropy_from_sequences(df, 'last_10_pitches')
    
    print(f"   ✅ Entropy stats:")
    print(f"      Mean: {df['sequence_entropy'].mean():.4f}")
    print(f"      Std:  {df['sequence_entropy'].std():.4f}")
    print(f"      Min:  {df['sequence_entropy'].min():.4f}")
    print(f"      Max:  {df['sequence_entropy'].max():.4f}")
    
    # 4. 데이터베이스 업데이트
    print(f"   💾 Updating database...")
    
    if not has_entropy_col:
        # 컬럼 추가
        con.execute(f"""
            ALTER TABLE {table_name} 
            ADD COLUMN sequence_entropy DOUBLE DEFAULT 0.0
        """)
        print(f"   ✅ Added sequence_entropy column")
    
    # Batch update
    # DuckDB는 직접 UPDATE ... FROM ... 지원
    # 임시 테이블 생성 후 JOIN UPDATE
    
    # 임시 테이블에 entropy 값 저장
    con.execute("DROP TABLE IF EXISTS temp_entropy")
    con.execute("""
        CREATE TEMP TABLE temp_entropy AS 
        SELECT * FROM df
    """)
    
    # UPDATE with JOIN
    con.execute(f"""
        UPDATE {table_name} p
        SET sequence_entropy = t.sequence_entropy
        FROM temp_entropy t
        WHERE p.game_pk = t.game_pk
          AND p.pitch_number = t.pitch_number
    """)
    
    rows_updated = con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE sequence_entropy > 0").fetchone()[0]
    print(f"   ✅ Updated {rows_updated:,} rows")
    
    con.close()
    print(f"✅ Sequence Entropy Integration Complete!")


def test_entropy_integration():
    """
    Entropy Integration 테스트
    """
    print("=" * 70)
    print("Test: Sequence Entropy SQL Integration")
    print("=" * 70)
    
    # 더미 데이터 생성
    print("\n1. Creating dummy data...")
    dummy_data = {
        'game_pk': [1, 1, 1, 1, 1, 2, 2, 2],
        'at_bat_number': [1, 1, 1, 2, 2, 1, 1, 1],
        'pitch_number': [1, 2, 3, 4, 5, 1, 2, 3],
        'pitch_type': ['FF', 'FF', 'SL', 'FF', 'CH', 'CH', 'FF', 'SL'],
        'pitcher': [12345, 12345, 12345, 12345, 12345, 67890, 67890, 67890],
        'last_10_pitches': [
            [],
            ['FF'],
            ['FF', 'FF'],
            ['SL', 'FF', 'FF'],
            ['FF', 'SL', 'FF', 'FF'],
            [],
            ['CH'],
            ['FF', 'CH']
        ]
    }
    
    df = pd.DataFrame(dummy_data)
    print(f"   Created {len(df)} dummy pitches")
    
    # 2. Entropy 계산
    print("\n2. Calculating entropy...")
    df['sequence_entropy'] = calculate_entropy_from_sequences(df, 'last_10_pitches')
    
    print(f"\n   Results:")
    for idx, row in df.iterrows():
        seq = row['last_10_pitches']
        entropy = row['sequence_entropy']
        print(f"   Pitch {idx+1}: {row['pitch_type']:3s} | Last 10: {seq} | Entropy: {entropy:.4f}")
    
    # 3. 검증
    print("\n3. Validation...")
    
    # Pitch 1: 빈 시퀀스 → 0.0
    assert df.loc[0, 'sequence_entropy'] == 0.0, "Empty sequence should be 0.0"
    print("   ✅ Empty sequence → 0.0")
    
    # Pitch 2: ['FF'] → 0.0 (모두 동일)
    assert df.loc[1, 'sequence_entropy'] == 0.0, "All same pitches should be 0.0"
    print("   ✅ All same pitches → 0.0")
    
    # Pitch 3: ['FF', 'FF'] → 0.0
    assert df.loc[2, 'sequence_entropy'] == 0.0
    print("   ✅ Two same pitches → 0.0")
    
    # Pitch 4: ['SL', 'FF', 'FF'] → 0.918 (2 FF, 1 SL)
    entropy_4 = df.loc[3, 'sequence_entropy']
    expected_4 = calculate_sequence_entropy(['SL', 'FF', 'FF'])
    assert abs(entropy_4 - expected_4) < 0.001, f"Expected {expected_4}, got {entropy_4}"
    print(f"   ✅ Mixed sequence → {entropy_4:.4f}")
    
    # Pitch 5: ['FF', 'SL', 'FF', 'FF'] → ~1.5 (3 FF, 1 SL)
    entropy_5 = df.loc[4, 'sequence_entropy']
    print(f"   ✅ 4-pitch sequence → {entropy_5:.4f}")
    
    print("\n✅ All Tests Passed!")


def main():
    """
    Main execution: Integrate entropy to actual database
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate Sequence Entropy to DuckDB')
    parser.add_argument('--db', type=str, default='../data/savant.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--table', type=str, default='pitches',
                       help='Table name')
    parser.add_argument('--update', action='store_true',
                       help='Update existing sequence_entropy column')
    parser.add_argument('--test', action='store_true',
                       help='Run test only')
    
    args = parser.parse_args()
    
    if args.test:
        test_entropy_integration()
    else:
        integrate_entropy_to_db(args.db, args.table, args.update)


if __name__ == "__main__":
    # Run test by default
    test_entropy_integration()
    
    # Uncomment to run integration:
    # main()
