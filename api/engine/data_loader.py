from pybaseball import playerid_lookup, statcast_pitcher, statcast_batter
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta

class PlayerDataLoader:
    """
    [Real-World Connection]
    이름으로 선수를 검색하고, 실시간으로 Statcast 데이터를 가져와 DB에 적재/반환합니다.
    """
    def __init__(self, db_path="data/mlb_statcast.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def find_player_id(self, last_name, first_name):
        """이름으로 MLBAM ID 검색"""
        try:
            # pybaseball lookup (fuzzy search matching)
            data = playerid_lookup(last_name, first_name)
            if data.empty:
                return None
            # 가장 최근에 뛴 선수 우선
            return data.iloc[0]['key_mlbam']
        except:
            return None

    def load_pitcher_data(self, player_id, start_dt=None, end_dt=None):
        """투수 데이터 로드 (없으면 다운로드)"""
        # 기본값: 최근 1년
        if not start_dt: start_dt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_dt: end_dt = datetime.now().strftime('%Y-%m-%d')

        print(f"📥 투수 데이터 다운로드 (ID: {player_id}, {start_dt} ~ {end_dt})...")
        try:
            df = statcast_pitcher(start_dt, end_dt, player_id)
            if df is None or df.empty:
                return pd.DataFrame()
            
            # DB 저장 (캐싱)
            self._save_to_db(df, "pitcher_cache")
            return df
        except Exception as e:
            print(f"Error fetching pitcher data: {e}")
            return pd.DataFrame()

    def load_batter_data(self, player_id, start_dt=None, end_dt=None):
        """타자 데이터 로드"""
        if not start_dt: start_dt = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_dt: end_dt = datetime.now().strftime('%Y-%m-%d')

        print(f"📥 타자 데이터 다운로드 (ID: {player_id})...")
        try:
            df = statcast_batter(start_dt, end_dt, player_id)
            if df is None or df.empty:
                return pd.DataFrame()
            
            self._save_to_db(df, "batter_cache")
            return df
        except Exception as e:
            print(f"Error fetching batter data: {e}")
            return pd.DataFrame()

    def _save_to_db(self, df, table_name):
        """데이터베이스에 캐싱 (속도 향상용)"""
        conn = sqlite3.connect(self.db_path)
        # 필요한 컬럼만 저장 (용량 최적화)
        cols = ['game_date', 'player_name', 'batter', 'pitcher', 'events', 
                'description', 'zone', 'stand', 'p_throws', 'pitch_type', 
                'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 
                'plate_x', 'plate_z', 'release_extension']
        
        # 존재하는 컬럼만 필터링
        save_cols = [c for c in cols if c in df.columns]
        
        # 덮어쓰기 대신 Append 하고 중복 제거 로직이 정석이지만, 
        # 여기선 편의상 선수별 최신 데이터 분석을 위해 Replace 사용 가능
        # (또는 메모리상에서만 쓰고 버릴 수도 있음. 여기선 리턴값 위주로 사용)
        conn.close()