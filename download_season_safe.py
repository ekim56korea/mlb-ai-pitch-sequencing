import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
import time

def download_season_safe():
    # 2024 시즌 시작일
    start_date = datetime(2023, 3, 20)
    end_date = datetime.now()
    
    # 5일 단위로 쪼개서 다운로드 (안전하게)
    delta = timedelta(days=5)
    current_date = start_date
    
    all_data = []
    
    print(f"⚾️ 2024 시즌 데이터를 안전하게 나누어 받습니다 ({start_date.strftime('%Y-%m-%d')} ~ 현재)...")
    print("⏳ 진행 상황을 터미널에 표시합니다. 중간에 멈추지 마세요.")

    while current_date <= end_date:
        # 구간 설정 (시작일 ~ +5일)
        chunk_end = current_date + delta
        if chunk_end > end_date:
            chunk_end = end_date
            
        str_start = current_date.strftime('%Y-%m-%d')
        str_end = chunk_end.strftime('%Y-%m-%d')
        
        print(f"📥 다운로드 중: {str_start} ~ {str_end} ...", end=" ")
        
        try:
            # 해당 구간 데이터 요청
            df = statcast(start_dt=str_start, end_dt=str_end, verbose=False)
            
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"✅ 성공 ({len(df)}개)")
            else:
                print("⚠️ 데이터 없음 (Pass)")
                
        except Exception as e:
            # 에러가 나도 멈추지 않고 다음 구간으로 넘어감 (중요!)
            print(f"❌ 실패 (Error: {e}) -> 건너뜁니다.")
        
        # 서버 부하 방지를 위해 1초 휴식
        time.sleep(1)
        
        # 다음 구간으로 이동
        current_date = chunk_end + timedelta(days=1)

    # ─── 데이터 합치기 및 저장 ───
    if all_data:
        print("\n🔄 다운로드 받은 조각들을 하나로 합치는 중...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 날짜순 정렬
        if 'game_date' in final_df.columns:
            final_df = final_df.sort_values('game_date', ascending=False)

        output_file = 'savant_data.csv'
        final_df.to_csv(output_file, index=False)
        
        print(f"🎉 전체 다운로드 완료! 총 {len(final_df):,}개의 데이터가 저장되었습니다.")
        print(f"💾 파일 위치: {output_file}")
    else:
        print("❌ 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    download_season_safe()