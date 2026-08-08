"""The resident model servers.

Each runs as its own process/container. Loading a model per turn destroys the
latency budget (§3, §12).

    uvicorn kotonoha.services._asr_server:app        --host 127.0.0.1 --port 8001 --loop uvloop
    uvicorn kotonoha.services._asr_verify_server:app --host 127.0.0.1 --port 8002 --loop uvloop
    uvicorn kotonoha.services._llm_server:app        --host 127.0.0.1 --port 8003 --loop uvloop
    uvicorn kotonoha.services._tts_server:app        --host 127.0.0.1 --port 8004 --loop uvloop
"""
