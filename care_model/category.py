import json
import os
import random
from datetime import datetime, timedelta

LOG_DIR = "logs"

CATEGORIES = {
    "음식",
    "운동",
    "가족",
    "여행",
    "취미",
    "장소",
    "건강",
    "감정"
}

def pick_category():
    recent_dates = [(datetime.today() - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 7)]
    used_categories = set()

    # 로그 파일 이름에서 카테고리 추출
    for filename in os.listdir(LOG_DIR):
        if filename.endswith(".txt"):
            category, date_str = filename.replace(".txt", "").rsplit("_", 1)
            if date_str in recent_dates:
                used_categories.add(category)

    unused_categories = [cat for cat in CATEGORIES if cat not in used_categories]

    if not unused_categories:
        return random.choice(list(CATEGORIES))

    return random.choice(unused_categories)