import os
from .utils import *
import openai

def get_remind_prompt(recent_text, age, date):
    return f"""
당신은 {age}살 사용자와 따뜻한 대화를 나누는 AI입니다.  
다음은 사용자가 최근에 나눈 대화 내용입니다.  
이 내용을 바탕으로 사용자가 최근 어떤 일을 했는지 스스로 떠올릴 수 있도록  
짧고 간단한 회상 질문을 한 문장 생성해주세요.

- 너무 길거나 복잡한 문장은 피하고, 간결하고 따뜻한 말투로 작성해주세요.
- 질문은 반드시 최근 했던 활동, 장소, 만난 사람, 감정 등을 회상할 수 있게 만들어주세요.
- 예시: "저번에 마트에 다녀오셨다고 하셨는데, 어떤걸 샀었죠?"

최근 대화: {date}일 전

--- 최근 대화 내용 ---
{recent_text}
-----------------------

회상 질문:

"""
def ask_gpt(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def remind(prompt):
    filename = f"answer.wav"
    filepath = os.path.join("data/", filename)

    user_text = single_wav_to_text(filepath).strip()
    print(f"사용자(STT): {user_text}")

    messages_remind = [{"role": "user", "system": prompt},
                       {"role": "user", "content": user_text}]
    reply = ask_gpt(messages_remind)
    print(f"GPT: {reply}")
    messages_remind.append({"role": "assistant", "content": reply})

    return messages_remind
