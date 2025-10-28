import pandas as pd
from pathlib import Path
from fastmcp.server import FastMCP, Context
from typing import List, Dict, Any, Optional
import numpy as np

# 전역 데이터 저장
DF: Optional[pd.DataFrame] = None

# MCP 서버 초기화 (AI에게 새 도구의 기능을 명확히 지시)
mcp = FastMCP(
    "StoreAnalyzerServer",
    instructions="""
    신한카드 가맹점 및 상권 분석 서비스입니다.
    
    사용자가 가맹점명(예: '윤스*')을 입력하면, `get_store_and_district_analysis` 함수를 사용하여 
    1. 해당 가맹점의 최신 상세 정보 (store_info)
    2. 해당 가맹점이 속한 상권의 전체 통계 (district_analysis)
    를 한 번에 검색하고 분석합니다.
    """
)

# 데이터 로드 함수 (원본과 동일)
def _load_df():
    global DF
    try:
        DF = pd.read_csv("./data/분기 합침.csv")
        # '기준년월'을 정수형으로 변환 (최신 월 찾기 위함)
        DF['기준년월'] = pd.to_numeric(DF['기준년월'], errors='coerce')
        print("데이터 로드 완료.")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        DF = None
    return DF

# 서버 시작 시 데이터 로드
_load_df()

@mcp.tool()
def get_store_and_district_analysis(merchant_name: str) -> Dict[str, Any]:
    """
    가맹점명을 입력받아, [1. 해당 가맹점의 최신 정보]와 [2. 해당 가맹점이 속한 상권의 전체 통계]를 반환합니다.
    
    매개변수:
     - merchant_name: 검색할 가맹점명 (예: '윤스*', '동대*')
    
    반환값:
     - 'store_info' (dict): 가맹점의 최신 상세 정보
     - 'district_analysis' (dict): 상권의 전체 통계 또는 에러 메시지
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    
    # --- 1. 가맹점 정보 검색 (사장님의 코드 수정) ---
    # '*'를 제거하고 .str.contains를 사용하여 부분 일치 검색
    search_term = merchant_name.replace('*', '')
    result_df = DF[DF['가맹점명'].astype(str).str.contains(search_term, na=False)]
    
    if result_df.empty:
        return {
            "store_info": {"error": f"'{merchant_name}' 가맹점을 찾을 수 없습니다."},
            "district_analysis": {"error": "가맹점을 찾을 수 없어 상권 분석을 수행할 수 없습니다."}
        }
    
    # '기준년월'이 가장 최신인 행을 'store_info'로 선택
    store_info = result_df.sort_values(by='기준년월', ascending=False).to_dict()
    
    # --- 2. (핵심) 상권 전체 데이터 분석 ---
    target_district = store_info.get('상권')
    
    if pd.isna(target_district):
        return {
            "store_info": store_info,
            "district_analysis": {"error": "이 가맹점은 배정된 상권이 없습니다."}
        }

    # '상권'이 일치하는 모든 가맹점 데이터를 DF에서 필터링
    # (주의: 이 상권 데이터에는 24개월치가 모두 포함됨)
    district_df = DF[DF['상권'] == target_district].copy()
    
    # 상권 분석 시, 가맹점별 '최신 데이터'만 사용하는 것이 정확함
    # '가맹점구분번호'별로 '기준년월'이 가장 최신인 행만 남김
    district_latest_df = district_df.sort_values('기준년월').drop_duplicates(subset=['가맹점구분번호'], keep='last')

    print(f"[{target_district}] 상권 분석 중... (총 {len(district_latest_df)}개 가맹점 대상)")

    # 상권 전체의 평균 통계 계산 (프롬프트 1번에 필요한 모든 항목)
    # (숫자 변환 및 -999999.9 처리)
    def clean_mean(series):
        series_numeric = pd.to_numeric(series, errors='coerce')
        series_cleaned = series_numeric.replace(-999999.9, np.nan)
        return series_cleaned.mean()

    district_analysis = {
        "상권명": target_district,
        "전체_가맹점_수": len(district_latest_df),
        "평균_거주이용고객비율": clean_mean(district_latest_df['거주이용고객비율']),
        "평균_직장이용고객비율": clean_mean(district_latest_df['직장이용고객비율']),
        "평균_유동인구이용고객비율": clean_mean(district_latest_df['유동인구이용고객비율']),
        "평균_남성20대이하고객비중": clean_mean(district_latest_df['남성20대이하고객비중']),
        "평균_남성30대고객비중": clean_mean(district_latest_df['남성30대고객비중']),
        "평균_남성40대고객비중": clean_mean(district_latest_df['남성40대고객비중']),
        "평균_남성50대고객비중": clean_mean(district_latest_df['남성50대고객비중']),
        "평균_남성60대이상고객비중": clean_mean(district_latest_df['남성60대이상고객비중']),
        "평균_여성20대이하고객비중": clean_mean(district_latest_df['여성20대이하고객비중']),
        "평균_여성30대고객비중": clean_mean(district_latest_df['여성30대고객비중']),
        "평균_여성40대고객비중": clean_mean(district_latest_df['여성40대고객비중']),
        "평균_여성50대고객비중": clean_mean(district_latest_df['여성50대고객비중']),
        "평균_여성60대이상고객비중": clean_mean(district_latest_df['여성60대이상고객비중']),
    }
    
    # 3. 두 정보(지점, 상권)를 합쳐서 반환
    return {
        "store_info": store_info,
        "district_analysis": district_analysis
    }

if __name__ == "__main__":
    mcp.run()