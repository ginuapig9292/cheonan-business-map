import os
import re
from pathlib import Path

import pandas as pd

# =========================================================
# 국토교통부 국토지리정보원_공간정보공동활용_읍면동 CSV
# → 천안 법정동/읍면동 경계 GeoJSON 변환 코드
#
# 입력 파일:
#   data/boundary/emd.csv
# 또는
#   data/boundary/국토교통부 국토지리정보원_공간정보공동활용_읍면동_20230915.csv
#
# 출력 파일:
#   result/dong_boundary.geojson
#
# 설치:
#   pip install pandas geopandas shapely pyogrio pyproj
#
# 실행:
#   python make_boundary_from_ngii_csv.py
# =========================================================

DATA_DIR = Path("data/boundary")
RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)

OUT = RESULT_DIR / "dong_boundary.geojson"

try:
    import geopandas as gpd
    from shapely import wkb
except ImportError:
    raise SystemExit(
        "geopandas 또는 shapely가 없습니다.\n"
        "터미널에서 아래 명령어 실행 후 다시 시도하세요.\n"
        "pip install geopandas shapely pyogrio pyproj"
    )


def read_csv_safely(path):
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if len(df.columns) > 1:
                print(f"[성공] {path}")
                print(f"인코딩: {enc}")
                print("컬럼:", df.columns.tolist())
                return df
        except Exception:
            pass
    raise ValueError(f"CSV를 읽지 못했습니다: {path}")


# 1. CSV 파일 찾기
candidates = [
    DATA_DIR / "emd.csv",
    DATA_DIR / "국토교통부 국토지리정보원_공간정보공동활용_읍면동_20230915.csv",
]

csv_files = [p for p in candidates if p.exists()]
if not csv_files:
    csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "data/boundary 폴더 안에 읍면동 CSV 파일이 없습니다.\n"
        "CSV 파일을 data/boundary/emd.csv 로 넣어주세요."
    )

csv_path = csv_files[0]
df = read_csv_safely(csv_path)

# 2. 컬럼명 확인
required_cols = ["읍면동코드", "읍면동명", "공간정보"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {df.columns.tolist()}")

# 3. 천안 필터
# 천안시 법정동 코드:
#   44131 = 동남구
#   44133 = 서북구
# 일부 파일의 객체시군구코드는 44130으로 들어올 수 있음
df["읍면동코드문자"] = df["읍면동코드"].astype(str)
df["객체시군구코드문자"] = df.get("객체시군구코드", "").astype(str)

cheonan = df[
    df["읍면동코드문자"].str.startswith(("44131", "44133"))
    | df["객체시군구코드문자"].str.startswith("44130")
].copy()

print("천안 필터 후 행 수:", len(cheonan))
print("천안 동명 목록:")
print(sorted(cheonan["읍면동명"].dropna().astype(str).unique().tolist()))

if len(cheonan) == 0:
    raise ValueError("천안 경계를 찾지 못했습니다. 읍면동코드 컬럼 값을 확인하세요.")

# 4. WKB HEX → geometry 변환
def parse_wkb_hex(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        return wkb.loads(bytes.fromhex(text))
    except Exception:
        return None

cheonan["geometry"] = cheonan["공간정보"].apply(parse_wkb_hex)
cheonan = cheonan.dropna(subset=["geometry"]).copy()

print("geometry 변환 성공 행 수:", len(cheonan))

# 5. GeoDataFrame 생성
gdf = gpd.GeoDataFrame(
    cheonan[["읍면동코드", "읍면동명", "geometry"]].copy(),
    geometry="geometry",
    crs="EPSG:4326"
)

# 6. 같은 동명이 여러 개면 일단 이름 기준 병합
# 예: 쌍용동처럼 중복명이 있는 경우 MultiPolygon으로 합쳐짐
gdf = gdf.rename(columns={"읍면동명": "ADM_NM", "읍면동코드": "ADM_CD"})
gdf["ADM_NM"] = gdf["ADM_NM"].astype(str).str.strip()

# 너무 자세한 경계는 웹에서 느려지므로 단순화
# 값이 클수록 가벼워지지만 경계가 거칠어짐
gdf["geometry"] = gdf["geometry"].simplify(0.00025, preserve_topology=True)

# HTML에서 이름 매칭이 쉽도록 name도 추가
gdf["name"] = gdf["ADM_NM"]

# 7. 저장
gdf.to_file(OUT, driver="GeoJSON", encoding="utf-8")

print("저장 완료:", OUT)
print("최종 경계 수:", len(gdf))
print("최종 동명 예시:", gdf["ADM_NM"].head(30).tolist())
