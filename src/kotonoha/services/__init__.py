"""상주 모델 서버들.

각각 별도 프로세스/컨테이너로 띄운다. 매 턴 모델을 로드하면 지연 예산이
무너진다(§3, §12).

    uvicorn kotonoha.services.asr_server:app        --host 0.0.0.0 --port 8001
    uvicorn kotonoha.services.asr_verify_server:app --host 0.0.0.0 --port 8002
    # :8003 은 llama.cpp server (scripts/run_llm.sh)
    uvicorn kotonoha.services.tts_server:app        --host 0.0.0.0 --port 8004
"""
