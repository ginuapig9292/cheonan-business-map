import os
import json
import math
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"
RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

STORE_CSV = os.path.join(DATA_DIR, "store.csv")
APARTMENT_CSV = os.path.join(DATA_DIR, "apartment.csv")
STATION_CSV = os.path.join(DATA_DIR, "station.csv")
UNIVERSITY_CSV = os.path.join(DATA_DIR, "university.csv")
OUTPUT_JSON = os.path.join(RESULT_DIR, "analysis_result.json")

def read_csv_safely(path):
    if not os.path.exists(path):
        print(f"[없음] {path}")
        return None
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    seps = [",", "\t", "|", ";"]
    last = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False, on_bad_lines="skip")
                if len(df.columns) > 1 and len(df) > 0:
                    print(f"\n[성공] {path} / encoding={enc} / sep={repr(sep)} / rows={len(df)}")
                    print("컬럼:", df.columns.tolist())
                    return df
            except Exception as e:
                last = e
    print(f"[주의] {path} 읽기 실패:", last)
    return None

def promote_header_if_needed(df, expected_keywords):
    """
    일부 공공데이터 CSV는 첫 줄에 '안내 문구'가 들어가 있어
    실제 컬럼명(세대수, 동리 등)이 헤더가 아니라 첫 데이터 행으로 밀려 있다.
    이 경우 첫 행을 진짜 헤더로 승격시킨다. (apartment.csv 세대수=0 버그 수정)
    """
    if df is None or len(df) == 0:
        return df
    cols = [str(c) for c in df.columns]
    has_keyword_in_header = any(any(k in c for k in expected_keywords) for c in cols)
    looks_like_disclaimer = any(("Unnamed" in c) for c in cols) or any(len(c) > 40 for c in cols)
    if has_keyword_in_header or not looks_like_disclaimer:
        return df
    first_row = df.iloc[0].astype(str).tolist()
    if any(any(k in v for k in expected_keywords) for v in first_row):
        new_df = df.iloc[1:].copy()
        new_df.columns = [str(v).strip() for v in first_row]
        new_df = new_df.reset_index(drop=True)
        print(f"[보정] 첫 줄이 안내문구라 첫 데이터 행을 헤더로 승격했습니다. 새 컬럼: {new_df.columns.tolist()[:8]} ...")
        return new_df
    return df

def find_col(df, candidates):
    if df is None:
        return None
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    norm = {str(c).replace(" ", "").strip(): c for c in cols}
    for cand in candidates:
        k = cand.replace(" ", "").strip()
        if k in norm:
            return norm[k]
    for col in cols:
        col_s = str(col).replace(" ", "")
        for cand in candidates:
            if cand.replace(" ", "") in col_s:
                return col
    return None

def clean_num(v):
    if pd.isna(v):
        return 0
    text = str(v).replace(",", "")
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0
    # 세대수 컬럼에 "2020-01-01" 같은 날짜가 들어오는 사고 방지: 가장 큰 숫자 사용
    return max(int(x) for x in nums)

def normalize(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min()) * 100

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except:
        return 999
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def nearest_info(row, targets):
    if not targets:
        return pd.Series([999, "없음"])
    best_d, best_n = 999, "없음"
    for t in targets:
        d = haversine(row["중심위도"], row["중심경도"], t["lat"], t["lng"])
        if d < best_d:
            best_d, best_n = d, t["name"]
    return pd.Series([best_d, best_n])

def extract_dongs_from_text(text):
    if pd.isna(text):
        return []
    text = str(text)
    # 주소에서 동/읍/면 추출. 예: "충남 천안시 서북구 두정동 123" -> 두정동
    return re.findall(r"([가-힣0-9]+(?:동|읍|면))", text)

def pick_matching_dong(row, candidate_cols, dong_list):
    # 모든 후보 컬럼에서 동명 추출 후, store 기준 동명과 맞는 것을 반환
    values = []
    for c in candidate_cols:
        if c in row.index:
            values.append(row[c])
    for v in values:
        # 1) 값 자체가 동명인 경우
        if not pd.isna(v):
            raw = str(v).strip()
            if raw in dong_list:
                return raw
        # 2) 주소/문장 속에서 동명 추출
        for d in extract_dongs_from_text(v):
            if d in dong_list:
                return d
    return None

# 1. store
store = read_csv_safely(STORE_CSV)
if store is None:
    raise FileNotFoundError("data/store.csv가 필요합니다.")

sigungu_col = find_col(store, ["시군구명", "시군구"])
legal_col = find_col(store, ["법정동명", "법정동"])
admin_col = find_col(store, ["행정동명", "행정동", "읍면동명"])
unit_col = legal_col or admin_col
name_col = find_col(store, ["상호명", "상호", "사업장명", "업소명"])
big_col = find_col(store, ["상권업종대분류명", "업종대분류명", "대분류"])
mid_col = find_col(store, ["상권업종중분류명", "업종중분류명", "중분류"])
small_col = find_col(store, ["상권업종소분류명", "업종소분류명", "소분류"])
lat_col = find_col(store, ["위도", "lat", "latitude", "Y좌표", "Y"])
lng_col = find_col(store, ["경도", "lng", "lon", "longitude", "X좌표", "X"])

for label, col in {"시군구": sigungu_col, "동명": unit_col, "상호": name_col, "위도": lat_col, "경도": lng_col}.items():
    if col is None:
        raise ValueError(f"store.csv에서 {label} 컬럼을 찾지 못했습니다.")

store = store[store[sigungu_col].astype(str).str.contains("천안", na=False)].copy()
store[lat_col] = pd.to_numeric(store[lat_col], errors="coerce")
store[lng_col] = pd.to_numeric(store[lng_col], errors="coerce")
store = store.dropna(subset=[lat_col, lng_col])
store["분석동명"] = store[unit_col].astype(str).str.strip()
print("\n분석 기준:", unit_col)

def classify_store(row):
    text = " ".join([str(row.get(big_col, "")), str(row.get(mid_col, "")), str(row.get(small_col, "")), str(row.get(name_col, ""))])
    cafe = ["카페","커피","다방","디저트","베이커리","제과","제빵","아이스크림","빙수","도넛","마카롱","케이크","브런치"]
    food = ["음식","한식","중식","일식","양식","분식","식당","국밥","고기","갈비","치킨","피자","버거","족발","보쌈","냉면","찜닭","초밥","라멘","돈까스","샤브","회","찌개","탕","김밥","떡볶이","순대"]
    activity = ["노래","PC","피시","볼링","당구","스크린","골프","헬스","요가","필라테스","게임","만화","키즈","방탈출","놀이터","체험","공방"]
    if any(k in text for k in cafe): return "카페"
    if any(k in text for k in food): return "음식점"
    if any(k in text for k in activity): return "액티비티"
    return "기타"

store["분석카테고리"] = store.apply(classify_store, axis=1)
target = store[store["분석카테고리"].isin(["카페", "음식점", "액티비티"])].copy()

summary = target.pivot_table(index=[sigungu_col, "분석동명"], columns="분석카테고리", values=name_col, aggfunc="count", fill_value=0).reset_index()
for c in ["카페", "음식점", "액티비티"]:
    if c not in summary.columns:
        summary[c] = 0
summary["총업소수"] = summary["카페"] + summary["음식점"] + summary["액티비티"]

centers = target.groupby([sigungu_col, "분석동명"]).agg(중심위도=(lat_col, "mean"), 중심경도=(lng_col, "mean")).reset_index()
summary = summary.merge(centers, on=[sigungu_col, "분석동명"], how="left")

# 같은 동명이 동남구/서북구로 나뉘어 집계되는 문제 보정 (예: 쌍용동)
# 동명 기준으로 업소 수를 합치고, 업소가 더 많은 구를 대표 구로, 중심좌표는 업소수 가중평균으로 통합
def consolidate_same_dong(df):
    if df["분석동명"].duplicated().any():
        rows = []
        for dong, g in df.groupby("분석동명"):
            if len(g) == 1:
                rows.append(g.iloc[0].to_dict())
                continue
            total = g["총업소수"].sum()
            # 대표 구: 업소가 가장 많은 행 기준
            main = g.loc[g["총업소수"].idxmax()]
            w = g["총업소수"].replace(0, 1)  # 0이면 1로 두어 가중치 부여
            lat = (g["중심위도"] * w).sum() / w.sum()
            lng = (g["중심경도"] * w).sum() / w.sum()
            rows.append({
                sigungu_col: main[sigungu_col],
                "분석동명": dong,
                "카페": int(g["카페"].sum()),
                "음식점": int(g["음식점"].sum()),
                "액티비티": int(g["액티비티"].sum()),
                "총업소수": int(total),
                "중심위도": lat,
                "중심경도": lng,
            })
        merged = pd.DataFrame(rows)
        # 컬럼 순서 정렬
        for c in df.columns:
            if c not in merged.columns:
                merged[c] = df.groupby("분석동명")[c].first().reindex(merged["분석동명"]).values
        print(f"[보정] 동명 중복 통합: {df['분석동명'].duplicated().sum()}건 합침 (예: 쌍용동)")
        return merged[df.columns.tolist()].reset_index(drop=True)
    return df

summary = consolidate_same_dong(summary)
dong_list = sorted(summary["분석동명"].astype(str).unique().tolist())

# 2. apartment robust matching
summary["아파트세대수"] = 0
summary["아파트단지수"] = 0
apt = read_csv_safely(APARTMENT_CSV)
if apt is not None:
    # apartment.csv는 첫 줄에 K-apt 안내문구가 들어가 세대수/동리 컬럼이 헤더로 안 잡힘 -> 보정
    apt = promote_header_if_needed(apt, ["세대수", "동리", "단지명", "시군구"])
    # 천안 필터
    possible_text_cols = [c for c in apt.columns if apt[c].dtype == "object"]
    if possible_text_cols:
        mask = pd.Series(False, index=apt.index)
        for c in possible_text_cols:
            mask = mask | apt[c].astype(str).str.contains("천안|동남구|서북구", na=False)
        # 천안이 포함된 행이 있으면 필터 적용, 없으면 전체 사용
        if mask.any():
            apt = apt[mask].copy()

    house_col = find_col(apt, ["세대수","총세대수","총 세대수","세대","가구수","공동주택세대수","세대수(세대)","세대수(호)","분양세대수","총호수","호수","세대수계"])
    if house_col:
        apt["__세대수"] = apt[house_col].apply(clean_num)
        print("아파트 세대수 컬럼:", house_col)
    else:
        apt["__세대수"] = 0
        print("[주의] 아파트 세대수 컬럼을 못 찾음. 단지수만 주거수요에 반영합니다.")

    preferred_cols = []
    for cand in ["법정동명","법정동","동리","행정동명","행정동","동","읍면동","읍면동명","읍면","주소","도로명주소","법정동주소","소재지","소재지도로명주소","소재지지번주소","위치","지번주소"]:
        col = find_col(apt, [cand])
        if col and col not in preferred_cols:
            preferred_cols.append(col)
    # 못 찾으면 모든 문자 컬럼에서 탐색
    candidate_cols = preferred_cols + [c for c in possible_text_cols if c not in preferred_cols]
    print("아파트 동명 매칭 후보 컬럼:", candidate_cols[:10])

    apt["__동"] = apt.apply(lambda r: pick_matching_dong(r, candidate_cols, dong_list), axis=1)
    valid = apt.dropna(subset=["__동"]).copy()

    print("아파트 매칭 행 수:", len(valid))
    print("아파트 매칭 동 목록:", sorted(valid["__동"].astype(str).unique().tolist())[:80])

    if len(valid) > 0:
        apt_summary = valid.groupby("__동").agg(
            아파트세대수=("__세대수", "sum"),
            아파트단지수=("__세대수", "count")
        ).reset_index()
        apt_summary.columns = ["분석동명", "아파트세대수", "아파트단지수"]

        summary = summary.drop(columns=["아파트세대수", "아파트단지수"])
        summary = summary.merge(apt_summary, on="분석동명", how="left")
        summary["아파트세대수"] = summary["아파트세대수"].fillna(0)
        summary["아파트단지수"] = summary["아파트단지수"].fillna(0)

# 3. station/university
stations = [
    {"name":"천안역","lat":36.8101,"lng":127.1469},
    {"name":"두정역","lat":36.8337,"lng":127.1489},
    {"name":"봉명역","lat":36.8017,"lng":127.1357},
    {"name":"쌍용역","lat":36.7937,"lng":127.1214},
    {"name":"천안아산역","lat":36.7946,"lng":127.1045},
]
terminals = [{"name":"천안종합버스터미널","lat":36.8194,"lng":127.1573}]
universities = [
    {"name":"공주대 천안캠퍼스","lat":36.8506,"lng":127.1527},
    {"name":"단국대 천안캠퍼스","lat":36.8334,"lng":127.1656},
    {"name":"백석대","lat":36.8407,"lng":127.1831},
    {"name":"상명대 천안캠퍼스","lat":36.8330,"lng":127.1774},
    {"name":"호서대 천안캠퍼스","lat":36.8235,"lng":127.1790},
]

st = read_csv_safely(STATION_CSV)
if st is not None:
    st_name = find_col(st, ["역명","역사명","정거장명","전철역명","시설명"])
    st_lat = find_col(st, ["위도","lat","latitude"])
    st_lng = find_col(st, ["경도","lng","lon","longitude"])
    if st_name and st_lat and st_lng:
        temp = st[st[st_name].astype(str).str.contains("천안|두정|봉명|쌍용|아산|부성", na=False)].copy()
        for _, r in temp.iterrows():
            try: stations.append({"name":str(r[st_name]),"lat":float(r[st_lat]),"lng":float(r[st_lng])})
            except: pass

univ = read_csv_safely(UNIVERSITY_CSV)
if univ is not None:
    univ_name = find_col(univ, ["학교명","대학명","학교","기관명"])
    univ_addr = find_col(univ, ["주소","도로명주소","소재지","학교주소","위치"])
    univ_lat = find_col(univ, ["위도","lat","latitude"])
    univ_lng = find_col(univ, ["경도","lng","lon","longitude"])
    if univ_name and univ_lat and univ_lng:
        mask_name = univ[univ_name].astype(str).str.contains("공주|단국|백석|상명|호서|천안", na=False)
        mask_addr = univ[univ_addr].astype(str).str.contains("천안", na=False) if univ_addr else False
        temp = univ[mask_name | mask_addr].copy()
        for _, r in temp.iterrows():
            try: universities.append({"name":str(r[univ_name]),"lat":float(r[univ_lat]),"lng":float(r[univ_lng])})
            except: pass

summary[["가까운역거리","가까운역명"]] = summary.apply(lambda r: nearest_info(r, stations), axis=1)
summary[["가까운터미널거리","가까운터미널명"]] = summary.apply(lambda r: nearest_info(r, terminals), axis=1)
summary[["가까운대학거리","가까운대학명"]] = summary.apply(lambda r: nearest_info(r, universities), axis=1)

summary["역접근성점수"] = (100 - summary["가까운역거리"]*12).clip(0,100)
summary["터미널접근성점수"] = (100 - summary["가까운터미널거리"]*15).clip(0,100)
summary["대학접근성점수"] = (100 - summary["가까운대학거리"]*12).clip(0,100)
summary["접근성점수"] = (summary["역접근성점수"]*0.45 + summary["터미널접근성점수"]*0.20 + summary["대학접근성점수"]*0.35).round(1)

summary["상권활성도"] = normalize(summary["총업소수"])
summary["포화도"] = normalize(summary["카페"] + summary["음식점"])
summary["카페포화도"] = normalize(summary["카페"])
summary["음식점포화도"] = normalize(summary["음식점"])
summary["액티비티포화도"] = normalize(summary["액티비티"])

# 세대수가 없거나 전부 0이면 단지수 기반으로 주거수요 산출
if summary["아파트세대수"].max() > 0:
    summary["주거수요점수"] = normalize(summary["아파트세대수"])
else:
    summary["주거수요점수"] = normalize(summary["아파트단지수"])
    print("[알림] 세대수 합계가 0이라 아파트단지수 기준으로 주거수요점수를 계산합니다.")

summary["학생수요점수"] = summary["대학접근성점수"]

development_map = {"부성동":94,"부대동":90,"성성동":96,"두정동":84,"불당동":86,"신부동":80,"백석동":72,"성정동":68,"신방동":65,"용곡동":66,"쌍용동":70}
summary["개발호재점수"] = summary["분석동명"].astype(str).map(development_map).fillna(55)

summary["카페창업점수"] = (summary["접근성점수"]*0.20 + summary["주거수요점수"]*0.30 + summary["학생수요점수"]*0.15 + summary["개발호재점수"]*0.15 + (100-summary["카페포화도"])*0.20).clip(0,100).round(1)
summary["음식점창업점수"] = (summary["접근성점수"]*0.25 + summary["주거수요점수"]*0.20 + summary["학생수요점수"]*0.15 + summary["개발호재점수"]*0.15 + summary["상권활성도"]*0.10 + (100-summary["음식점포화도"])*0.15).clip(0,100).round(1)
summary["액티비티창업점수"] = (summary["주거수요점수"]*0.30 + summary["학생수요점수"]*0.25 + summary["접근성점수"]*0.20 + summary["개발호재점수"]*0.10 + (100-summary["액티비티포화도"])*0.15).clip(0,100).round(1)
summary["창업추천점수"] = summary[["카페창업점수","음식점창업점수","액티비티창업점수"]].max(axis=1).round(1)
summary["업종다양성"] = normalize((summary["카페"]>0).astype(int)+(summary["음식점"]>0).astype(int)+(summary["액티비티"]>0).astype(int))
summary["방문추천점수"] = (summary["상권활성도"]*0.35 + summary["접근성점수"]*0.30 + summary["업종다양성"]*0.20 + summary["개발호재점수"]*0.15).clip(0,100).round(1)
summary["미래성장점수"] = (summary["개발호재점수"]*0.35 + summary["주거수요점수"]*0.25 + summary["접근성점수"]*0.20 + summary["학생수요점수"]*0.10 + (100-summary["포화도"])*0.10).clip(0,100).round(1)

def best_type(row):
    d = {"카페 창업 후보":row["카페창업점수"], "음식점 창업 후보":row["음식점창업점수"], "생활·액티비티 창업 후보":row["액티비티창업점수"]}
    return max(d, key=d.get)

def make_insight(row):
    apt_text = f"공동주택 {int(row['아파트세대수']):,}세대·{int(row['아파트단지수'])}개 단지" if row["아파트단지수"] > 0 else "공동주택 CSV 매칭값 부족"
    return f"{row[sigungu_col]} {row['분석동명']}은 법정동 기준으로 분리 분석한 지역입니다. 소상공인 데이터 기준 카페 {int(row['카페'])}개, 음식점 {int(row['음식점'])}개, 생활·액티비티 업소 {int(row['액티비티'])}개가 집계되었습니다. {apt_text}가 반영되어 주거수요 점수는 {row['주거수요점수']:.1f}점입니다. 가장 가까운 역은 {row['가까운역명']}({row['가까운역거리']:.2f}km), 가장 가까운 대학은 {row['가까운대학명']}({row['가까운대학거리']:.2f}km)입니다. 업종별 창업 점수는 카페 {row['카페창업점수']:.1f}점, 음식점 {row['음식점창업점수']:.1f}점, 생활·액티비티 {row['액티비티창업점수']:.1f}점이며, 이 지역은 '{best_type(row)}'로 해석할 수 있습니다."

summary["추천유형"] = summary.apply(best_type, axis=1)
summary["인사이트"] = summary.apply(make_insight, axis=1)

records = []
for _, row in summary.iterrows():
    records.append({
        "name": str(row["분석동명"]), "gu": str(row[sigungu_col]), "type": str(row["추천유형"]),
        "lat": float(row["중심위도"]) if not pd.isna(row["중심위도"]) else 36.815,
        "lng": float(row["중심경도"]) if not pd.isna(row["중심경도"]) else 127.145,
        "cafe": int(row["카페"]), "food": int(row["음식점"]), "activity_count": int(row["액티비티"]),
        "apt": int(row["아파트세대수"]), "apartment_complex_count": int(row["아파트단지수"]),
        "comp": float(round(row["포화도"],1)), "access": float(round(row["접근성점수"],1)), "news": float(round(row["개발호재점수"],1)),
        "startup": float(row["창업추천점수"]), "visit": float(row["방문추천점수"]), "growth": float(row["미래성장점수"]), "saturation": float(round(row["포화도"],1)),
        "cafe_startup": float(row["카페창업점수"]), "food_startup": float(row["음식점창업점수"]), "activity_startup": float(row["액티비티창업점수"]),
        "housing_score": float(round(row["주거수요점수"],1)), "student_score": float(round(row["학생수요점수"],1)),
        "nearest_station": str(row["가까운역명"]), "nearest_station_km": float(round(row["가까운역거리"],2)),
        "nearest_terminal": str(row["가까운터미널명"]), "nearest_terminal_km": float(round(row["가까운터미널거리"],2)),
        "nearest_university": str(row["가까운대학명"]), "nearest_university_km": float(round(row["가까운대학거리"],2)),
        "color": "#64748b", "insight": row["인사이트"], "news_query": f"천안 {row['분석동명']} 부동산 개발 상권"
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("\n분석 완료:", OUTPUT_JSON)
print("행 수:", len(records))
print("\n주거수요 상위 10개 동")
print(summary.sort_values(["아파트세대수","아파트단지수"], ascending=False)[[sigungu_col, "분석동명", "아파트세대수", "아파트단지수", "주거수요점수"]].head(10))
