"""The resident model servers.

Each runs as its own process/container. Loading a model per turn destroys the
latency budget (§3, §12).

    uvicorn kotonoha.services.asr_server:app        --host 0.0.0.0 --port 8001 --loop uvloop
    uvicorn kotonoha.services.asr_verify_server:app --host 0.0.0.0 --port 8002 --loop uvloop
    # :8003 is the llama.cpp server (scripts/run_llm.sh)
    uvicorn kotonoha.services.tts_server:app        --host 0.0.0.0 --port 8004 --loop uvloop
"""
