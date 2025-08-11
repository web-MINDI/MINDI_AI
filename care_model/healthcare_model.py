from .remind_prompt import *
from .daily_prompt import *
from .category import *
from .utils import *
import openai
import httpx
import os

# GPT API 키 설정 (환경변수 권장)
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# 음성파일 폴더
AUDIO_DIR = "data/"

# 대화 이력 저장 폴더
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# GPT 응답 생성 함수
def ask_gpt(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def get_latest_txt_file():
    txt_files = [
        f for f in os.listdir(LOG_DIR)
        if f.endswith(".txt")
    ]
    if not txt_files:
        return None

    # 수정시간 기준 정렬
    txt_files.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, txt_files[0])

def fetch_last_log_from_backend(user_id):
    url = f"http://localhost:8000/care/last-log"
    headers = {"Authorization": f"Bearer {user_id}"}  # 실제 서비스에서는 JWT 등 사용
    try:
        response = httpx.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("text", "")
    except Exception as e:
        print(f"[오류] 백엔드에서 대화 기록 불러오기 실패: {e}")
    return ""

# def main(user_id=None):
#     print("\n💬 음성 기반 치매 예방 회상 대화 시작")
#     # 이전 대화 기록을 백엔드에서 불러옴
#     yesterday_text = fetch_last_log_from_backend(user_id) if user_id else None

#     if yesterday_text:
#         prompt = get_remind_prompt(yesterday_text)
#         for i in range (1,4):
#             remind(prompt)

#     # 일상 대화: 항상 진행
#     MAX_INDEX = 23  # 최대 파일 수 예상 범위 (필요 시 조정)
#     category = pick_category()

#     for i in range(4, MAX_INDEX + 1):
#         messages = daily(category)

#     # 대화 저장
#     full_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
#     today = datetime.today().strftime("%Y%m%d")
#     filename = f"{category}_{today}.txt"
#     path = "logs/"+filename
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(full_text)

# if __name__ == "__main__":
#     # 예시: user_id를 환경변수나 인자로 받아서 전달
#     import os
#     user_id = os.getenv("USER_ID", None)
#     main(user_id)
