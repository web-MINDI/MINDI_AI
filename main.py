from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
import traceback
import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
from care_model.utils import single_wav_to_text
from care_model.healthcare_model import ask_gpt
from care_model.remind_prompt import get_remind_prompt

# 진단 모델 import 추가
import sys
sys.path.append('diagnosis_model')
from diagnosis_model.ko_model import ko_language_model, ko_acoustic_model

# 리포트 모델 import 추가
sys.path.append('report_model')
from report_model.health_report import report_by_score
from report_model.make_care_report import score_model, evaluate_fluency, evaluate_emotionality, evaluate_sentence_complexity, extract_result, evaluate_memory_fidelity, evaluate_coherence
from report_model.main import generate_weekly_report, collect_weekly_results

app = FastAPI()

UPLOAD_DIR = "care_model/data/"
DIAGNOSIS_UPLOAD_DIR = "diagnosis_model/ko_wav/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DIAGNOSIS_UPLOAD_DIR, exist_ok=True)

# 개인화된 인사말 요청을 위한 스키마
class ConversationData(BaseModel):
    user_question: str
    ai_reply: str
    conversation_date: str
    created_at: str

class PersonalizedGreetingRequest(BaseModel):
    user_id: int
    recent_conversations: List[ConversationData]
    age: int

class PersonalizedGreetingResponse(BaseModel):
    greeting_text: str

# 진단 관련 스키마 추가
class DiagnosisRequest(BaseModel):
    session_id: str
    user_id: int
    user_age: int
    user_education: str

class DiagnosisResponse(BaseModel):
    session_id: str
    user_id: int
    total_score: float
    language_score: float
    acoustic_score: float
    check_score: float
    dementia_result: int  # 0: 정상, 1: 치매
    risk_level: str
    detailed_analysis: str
    threshold: int
    language_score_gpt: float
    language_score_BERT: float
    acoustic_score_vit: float
    acoustic_score_lgbm: float
    
# 리포트 관련 스키마 추가
class DiagnosisReportRequest(BaseModel):
    user_id: int
    acoustic_score_vit: float
    acoustic_score_lgbm: float
    language_score_BERT: float
    language_score_gpt: float
    user_name: str

class DiagnosisReportResponse(BaseModel):
    evaluate_good_list: List[str]
    evaluate_bad_list: List[str]
    result_good_list: List[str]
    result_bad_list: List[str]

class CareReportRequest(BaseModel):
    user_id: int
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    user_email: str
    user_name: str
    weekly_conversations: List[Dict[str, Any]]  # 백엔드에서 전달받은 주간 대화 데이터

class CareReportResponse(BaseModel):
    report_html: str
    report_text: str
    weekly_data: List[Dict]
    overall_comment: str
    care_recommendations: List[str]

class EmailRequest(BaseModel):
    to_email: str
    subject: str
    html_content: str
    report_type: str  # 'diagnosis' or 'care'
    user_id: int

@app.post("/stt-and-reply")
async def stt_and_reply(file: UploadFile = File(...), messages: str = Form(...)):
    # 1. 파일 저장
    ext = file.filename.split('.')[-1] if file.filename and '.' in file.filename else 'wav'
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 2. 음성 → 텍스트 변환
    try:
        user_text = single_wav_to_text(file_path).strip()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"STT 변환 실패: {e}")

    # 3. messages 파싱 및 user 메시지 추가
    try:
        messages_list = json.loads(messages)
        messages_list.append({"role": "user", "content": user_text})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"messages 파싱 실패: {e}")

    # 4. AI 답변 생성
    try:
        ai_reply = ask_gpt(messages_list)
        print(ai_reply)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"AI 답변 생성 실패: {e}")

    # 5. 결과 반환
    return {"reply": ai_reply, "user_text": user_text}

@app.post("/diagnosis")
async def diagnose_single_answer(
    file: UploadFile = File(...),
    question_id: str = Form(...),
    user_id: int = Form(...)
):
    """단일 질문에 대한 진단 처리 - 파일만 저장"""
    
    try:
        # 1. 파일 저장 (ko_model.py 방식과 동일하게)
        filename = f"{user_id}_ans{question_id}.wav"
        file_path = os.path.join(DIAGNOSIS_UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 2. 성공 응답 반환
        result = {
            "question_id": question_id,
            "file_path": file_path,
            "status": "saved",
            "message": f"답변 {question_id} 저장 완료"
        }
        
        return result
        
    except Exception as e:
        print(f"진단 파일 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

@app.post("/diagnosis/final", response_model=DiagnosisResponse)
async def diagnose_final(request: DiagnosisRequest):
    """전체 진단 결과 분석 - ko_model.py 방식 적용"""
    
    try:
        # ko_model.py와 동일한 임계값 테이블
        threshold_table = [
            {"age_min": 0, "age_max": 59, "thresholds": {"ele": 24, "middle": 26, "high": 28, "univ": 30}},
            {"age_min": 60, "age_max": 69, "thresholds": {"ele": 23, "middle": 25, "high": 27, "univ": 28}},
            {"age_min": 70, "age_max": 79, "thresholds": {"ele": 21, "middle": 24, "high": 25, "univ": 27}},
            {"age_min": 80, "age_max": 200, "thresholds": {"ele": 18, "middle": 24, "high": 22, "univ": 24}},
        ]
        
        # ko_model.py 방식으로 모든 답변을 한번에 STT 처리
        answers = []
        for i in range(1, 22):
            answer_path = f"{DIAGNOSIS_UPLOAD_DIR}{request.user_id}_ans{i}.wav"
            answer = single_wav_to_text(answer_path) if os.path.exists(answer_path) else ""
            answers.append(answer)
        
        # ko_model.py와 동일한 음향 모델 경로 설정
        acoustic_path = []
        for i in range(18, 22):  # ans18.wav ~ ans21.wav만 사용 (ko_model.py와 동일)
            path = f"{DIAGNOSIS_UPLOAD_DIR}{request.user_id}_ans{i}.wav"
            if os.path.exists(path):
                acoustic_path.append(path)
        
        # ko_model.py와 동일한 방식으로 점수 계산
        try:
            language_score, check_score, language_score_gpt, language_score_BERT = ko_language_model(
                answers[0], answers[1], answers[2], answers[3], answers[4], answers[6], answers[7], answers[8], answers[9], answers[10], answers[11], answers[12], answers[13], answers[14], answers[15], answers[16], answers[17], answers[18], answers[19], answers[20]
            )
        except Exception as e:
            print(f"언어 모델 오류: {traceback.format_exc()}")
            language_score, check_score, language_score_gpt, language_score_BERT = 0, 0, 0, 0
        
        # 음향 모델 점수 계산
        try:
            if acoustic_path:
                acoustic_score, acoustic_score_lgbm, acoustic_score_vit = ko_acoustic_model(acoustic_path)
            else:
                acoustic_score, acoustic_score_lgbm, acoustic_score_vit = 0, 0, 0
        except Exception as e:
            print(f"음향 모델 오류: {e}")
            acoustic_score, acoustic_score_lgbm, acoustic_score_vit = 0, 0, 0
        
        # 총점 계산 (ko_model.py와 동일)
        total_score = language_score + acoustic_score + check_score
        
        # 임계값 결정
        threshold = None
        for row in threshold_table:
            if row["age_min"] <= request.user_age <= row["age_max"]:
                threshold = row["thresholds"].get(request.user_education, 25)
                break
        
        # ko_model.py와 동일한 진단 로직
        dementia_result = 0  # 기본값: 정상
        risk_level = ""
        
        if threshold:
            if total_score <= (threshold - 1):
                dementia_result = 1
                risk_level = "severe"
            elif (threshold - 1) < total_score <= (threshold + 1):
                risk_level = "mild"
            else:
                risk_level = "normal"
        
        # 상세 분석 (ko_model.py 출력 형식과 유사)
        detailed_analysis = f"""
        설문 점수: {check_score}/24
        언어모델 점수: {language_score}/6
        음향모델 점수: {acoustic_score}/6
        총점: {total_score}
        임계값: {threshold} (이상이면 정상)
        위험도: {risk_level}
        """

        print(f"{detailed_analysis}")
        
        return DiagnosisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            total_score=total_score,
            language_score=language_score,
            acoustic_score=acoustic_score,
            check_score=check_score,
            dementia_result=dementia_result,
            risk_level=risk_level,
            detailed_analysis=detailed_analysis,
            threshold=threshold,
            language_score_gpt=language_score_gpt,
            language_score_BERT=language_score_BERT,
            acoustic_score_vit=acoustic_score_vit,
            acoustic_score_lgbm=acoustic_score_lgbm
        )
        
    except Exception as e:
        print(f"최종 진단 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"최종 진단 처리 실패: {e}")

@app.post("/personalized-greeting", response_model=PersonalizedGreetingResponse)
async def generate_personalized_greeting(request: PersonalizedGreetingRequest):
    """최근 대화 기록을 바탕으로 개인화된 인사말 생성"""
    
    try:
        # 최근 대화가 없는 경우 기본 인사말
        if not request.recent_conversations:
            return PersonalizedGreetingResponse(
                greeting_text="안녕하세요! 민디입니다. 오늘 하루는 어떠셨나요?"
            )
        
        # 최근 대화 내용 분석을 위한 프롬프트 구성
        conversation_summary = ""
        conversation_date = datetime.today() - datetime.strptime(request.recent_conversations[0].conversation_date, "%Y-%m-%d")
        
        for i, conv in enumerate(request.recent_conversations, 1):
            conversation_summary += f"\n대화 {i}:\n"
            conversation_summary += f"사용자: {conv.user_question}\n"
            conversation_summary += f"민디: {conv.ai_reply}\n"
        
        # GPT에게 개인화된 인사말 생성 요청
        system_prompt = get_remind_prompt(conversation_summary, request.age, conversation_date.days)

        user_prompt = f"""최근 대화 날짜: {conversation_date.days}일 전

최근 대화 내용:
{conversation_summary}

위의 대화 내용을 바탕으로 자연스럽고 개인화된 인사말을 생성해주세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # GPT로 개인화된 인사말 생성
        greeting_text = ask_gpt(messages)
        
        # 응답에서 불필요한 따옴표나 줄바꿈 제거
        greeting_text = greeting_text.strip().strip('"').strip("'")
        
        print(f"생성된 개인화 인사말: {greeting_text}")
        
        return PersonalizedGreetingResponse(greeting_text=greeting_text)
        
    except Exception as e:
        print(f"개인화 인사말 생성 오류: {e}")
        # 오류 발생 시 기본 인사말 반환
        return PersonalizedGreetingResponse(
            greeting_text="안녕하세요! 민디입니다. 오늘 하루는 어떠셨나요?"
        )

# 리포트 관련 엔드포인트 추가
@app.post("/diagnosis/generate-diagnosis-report", response_model=DiagnosisReportResponse)
async def generate_diagnosis_report(request: DiagnosisReportRequest):
    """진단 결과를 바탕으로 리포트 생성"""
    try:
        # 기존 health_report.py의 report_by_score 함수 활용
        evaluate_good_list, evaluate_bad_list, result_good_list, result_bad_list = report_by_score(
            request.acoustic_score_vit,
            request.acoustic_score_lgbm,
            request.language_score_BERT,
            request.language_score_gpt
        )
        
        return DiagnosisReportResponse(
            evaluate_good_list=evaluate_good_list,
            evaluate_bad_list=evaluate_bad_list,
            result_good_list=result_good_list,
            result_bad_list=result_bad_list,
        )
        
    except Exception as e:
        print(f"진단 리포트 생성 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"진단 리포트 생성 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 
