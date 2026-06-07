import os
import re
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # 개발 환경: 포트/파일 열기 방식에 상관없이 동작하도록 모든 origin 허용
    # (운영 배포 시에는 특정 도메인으로 제한하세요)
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


def clean_html(text):
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"<[^>]*>", "", text)
    return text


@app.get("/api/news")
def get_news(
    query: str = Query("천안 부성역"),
    display: int = Query(6, ge=1, le=20),
):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {
            "error": ".env 파일에 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 없습니다."
        }

    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": query,
        "display": display,
        "sort": "date",
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code != 200:
        return {
            "error": "네이버 뉴스 API 호출 실패",
            "status_code": response.status_code,
            "message": response.text,
        }

    data = response.json()

    items = []
    for item in data.get("items", []):
        items.append({
            "title": clean_html(item.get("title", "")),
            "description": clean_html(item.get("description", "")),
            "link": item.get("link", ""),
            "originallink": item.get("originallink", ""),
            "pubDate": item.get("pubDate", ""),
        })

    return {
        "query": query,
        "items": items,
    }
