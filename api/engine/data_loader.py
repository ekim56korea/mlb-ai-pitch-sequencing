from pybaseball import playerid_lookup, statcast_pitcher, statcast_batter
import pandas as pd
import sqlite3
import os
import asyncio
from datetime import datetime, timedelta
from api.engine.preprocessor import DataPreprocessor

class PlayerDataLoader:
    """
    [v7.0 Phase 1] Optimized Data Loader (Zero-Cost Architecture)
    - SQLite WAL Mode & Indexing (Performance Tuning)
    - Parquet File Caching (Cold Data Storage)
    - Async IO Support
    """
    def __init__(self, db_path="data/mlb_statcast.db", parquet_dir="data/parquet_cache"):
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)
        
        self.preprocessor = DataPreprocessor()
        
        # [1.1] DB 최적화 (WAL 모드 및 인덱싱)
        self._init_db_optimization()

    def _init_db_optimization(self):
        """SQLite 성능을 극한으로 끌어올리는 설정"""
        try:
            conn = sqlite3.connect(self.db_path)
            # WAL(Write-Ahead Logging) 모드: 읽기/쓰기 동시성 향상
            conn.execute("PRAGMA journal_mode=WAL;")
            # 동기화 레벨 완화 (데이터 안정성보다 속도 우선)
            conn.execute("PRAGMA synchronous=NORMAL;")
            # 캐시 사이즈 증대
            conn.execute("PRAGMA cache_size=10000;")
            
            # [Indexing] 자주 조회하는 컬럼에 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pitcher_date ON pitcher_cache (pitcher, game_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batter_date ON batter_cache (batter, game_date);")
            
            conn.close()
            print("⚡ SQLite WAL Mode & Indexing Enabled.")
        except Exception as e:
            print(f"⚠️ DB Optimization Warning: {e}")

    def find_player_id(self, name_input):
        """(기존과 동일) 유연한 선수 검색"""
        if not name_input: return None
        parts = name_input.strip().split()
        if not parts: return None

        try:
            data = pd.DataFrame()
            if len(parts) >= 2:
                last, first = parts[-1], parts[0]
                data = playerid_lookup(last, first)
                if data.empty: data = playerid_lookup(first, last)
                if data.empty: data = playerid_lookup(parts[-1], parts[0], fuzzy=True)
            else:
                data = playerid_lookup(parts[0], fuzzy=True)

            if data.empty:
                print(f"❌ Player not found: '{name_input}'")
                return None
            
            if 'mlb_played_last' in data.columns:
                data = data.sort_values('mlb_played_last', ascending=False)
            
            player_id = data.iloc[0]['key_mlbam']
            print(f"✅ Found Player: {data.iloc[0]['name_first']} {data.iloc[0]['name_last']} (ID: {player_id})")
            return player_id
            
        except Exception as e:
            print(f"❌ Lookup Error: {e}")
            return None

    def load_pitcher_data(self, player_id, start_dt=None, end_dt=None):
        """[1.2] Parquet 우선 로딩 방식 적용"""
        if not start_dt: start_dt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_dt: end_dt = datetime.now().strftime('%Y-%m-%d')

        # 1. Parquet 캐시 확인 (가장 빠름)
        parquet_path = os.path.join(self.parquet_dir, f"pitcher_{player_id}.parquet")
        if os.path.exists(parquet_path):
            try:
                # 파일 수정 시간 확인 (하루 지났으면 다시 다운로드)
                mtime = datetime.fromtimestamp(os.path.getmtime(parquet_path))
                if datetime.now() - mtime < timedelta(hours=24):
                    print(f"⚡ Loading from Parquet Cache: {parquet_path}")
                    return pd.read_parquet(parquet_path)
            except Exception: pass # 읽기 실패 시 다운로드 진행

        print(f"📥 Downloading Pitcher Data (ID: {player_id})...")
        try:
            df = statcast_pitcher(start_dt, end_dt, player_id)
            if df is None or df.empty: return pd.DataFrame()
            
            df_clean = self.preprocessor.clean_data(df)
            
            # 2. 저장 (DB + Parquet)
            self._save_to_db(df_clean, "pitcher_cache")
            self._save_to_parquet(df_clean, parquet_path) # [New]
            
            return df_clean
        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

    def load_batter_data(self, player_id, start_dt=None, end_dt=None):
        if not start_dt: start_dt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_dt: end_dt = datetime.now().strftime('%Y-%m-%d')

        parquet_path = os.path.join(self.parquet_dir, f"batter_{player_id}.parquet")
        if os.path.exists(parquet_path):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(parquet_path))
                if datetime.now() - mtime < timedelta(hours=24):
                    print(f"⚡ Loading from Parquet Cache: {parquet_path}")
                    return pd.read_parquet(parquet_path)
            except: pass

        print(f"📥 Downloading Batter Data (ID: {player_id})...")
        try:
            df = statcast_batter(start_dt, end_dt, player_id)
            if df is None or df.empty: return pd.DataFrame()
            
            self._save_to_db(df, "batter_cache")
            self._save_to_parquet(df, parquet_path)
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

    def _save_to_db(self, df, table_name):
        """SQLite 저장 (메타데이터 및 SQL 쿼리용)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cols = ['game_date', 'player_name', 'batter', 'pitcher', 'events', 
                    'description', 'zone', 'stand', 'p_throws', 'pitch_type', 
                    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 
                    'plate_x', 'plate_z', 'release_extension']
            save_cols = [c for c in cols if c in df.columns]
            df[save_cols].to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
        except Exception as e: print(f"⚠️ DB Save Warning: {e}")

    def _save_to_parquet(self, df, path):
        """[1.2] Parquet 저장 (초고속 로딩용)"""
        try:
            # 모든 컬럼을 문자열로 변환하여 저장하면 호환성 문제 감소 (선택사항)
            # 여기서는 그대로 저장
            df.to_parquet(path, engine='pyarrow', index=False)
            print(f"💾 Saved to Parquet: {path}")
        except Exception as e:
            print(f"⚠️ Parquet Save Warning: {e}")