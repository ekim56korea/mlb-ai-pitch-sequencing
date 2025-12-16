import duckdb
import os
import time
import glob

DATA_DIR = "savant_data"
DB_FILE = "savant.duckdb"

def migrate_data():
    # 데이터 폴더 확인
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: '{DATA_DIR}' directory not found!")
        return

    # CSV 파일들이 있는지 확인
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{DATA_DIR}'!")
        return

    print(f"found {len(csv_files)} CSV files.")

    # 기존 DB 삭제 (초기화)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    print(f"🚀 Building Database from {len(csv_files)} years of data...")
    start_time = time.time()

    con = duckdb.connect(DB_FILE)

    # 🌟 DuckDB의 마법: read_csv_auto 안에 와일드카드(*)를 쓰면 알아서 합쳐줍니다.
    # union_by_name=True: 연도별로 컬럼 순서가 달라도 이름 기준으로 맞춰줍니다 (매우 중요!)
    print("📥 Importing all CSVs into DuckDB...")
    
    query = f"""
        CREATE TABLE pitches AS 
        SELECT * FROM read_csv_auto('{DATA_DIR}/*.csv', union_by_name=True)
    """
    con.execute(query)

    print("⚡ Creating Indexes...")
    con.execute("CREATE INDEX idx_pitcher ON pitches(player_name)")
    con.execute("CREATE INDEX idx_pitch_type ON pitches(pitch_type)")
    con.execute("CREATE INDEX idx_stand ON pitches(stand)")
    con.execute("CREATE INDEX idx_date ON pitches(game_date)") # 날짜 검색용 추가

    # 총 데이터 개수 확인
    count = con.execute("SELECT count(*) FROM pitches").fetchone()[0]
    con.close()
    
    elapsed = time.time() - start_time
    print(f"✅ Migration Complete! Total Pitches: {count:,}")
    print(f"⏱ Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    migrate_data()