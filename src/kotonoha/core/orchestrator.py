"""오케스트레이터 — 상태 기계, 언어 라우팅, 품질 게이트, 절 스트리밍, 실패 처리.

한 턴의 흐름:

    EOU → (잡음 제거) → shm publish → ASR(N-best 5 + LID)
        → LID 결정/폴백 → 조건부 교차 검증
        → 정정+번역 단일 패스 스트리밍 → 절 단위로 TTS → 재생

§10 을 그대로 옮겼다. 통역기는 틀린 답보다 멈추는 게 더 치명적이므로,
모든 단계가 타임아웃과 폴백을 갖고 어떤 경로로 빠지든 IDLE 로 돌아온다.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np

from ..clients import (
    AsrClient,
    AsrVerifyClient,
    LlmClient,
    ServiceError,
    ServiceTimeout,
    StreamStats,
    TtsClient,
)
from ..config import Settings
from ..logging_setup import get_logger
from ..metrics import TurnLog, TurnMetrics
from ..prompts import build_asr_context, build_translate_messages
from ..shmring import AudioRing
from ..store import Store
from .clauses import ClauseStreamer
from .events import EventBus
from .lid import decide_language, route_targets
from .quality import cer, is_divergent, should_cross_verify
from .state import Machine, State
from .zh import TraditionalizeTW, looks_simplified

log = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        capture,
        segmenter,
        playback,
        denoiser=None,
        session_id: str | None = None,
    ):
        self.s = settings
        self.capture = capture
        self.seg = segmenter
        self.playback = playback
        self.denoiser = denoiser
        self.session_id = session_id or uuid.uuid4().hex[:12]

        self.bus = EventBus()
        self.machine = Machine(on_change=self._on_state_change)

        self.ring = AudioRing.create(
            name=settings.shm.name,
            slots=settings.shm.slots,
            slot_seconds=settings.shm.slot_seconds,
            sample_rate=settings.shm.sample_rate,
        )

        self.asr = AsrClient(settings.services.asr, settings.asr)
        self.verify = AsrVerifyClient(settings.services.asr_verify, settings.asr_verify)
        self.llm = LlmClient(settings.services.llm, settings.llm)
        self.tts = TtsClient(settings.services.tts, settings.tts)

        self.store = Store(settings.resolve(settings.store.path))
        self.turnlog = TurnLog(settings.resolve(settings.logging.turn_log_path), settings.budget_ms)
        self.zh = TraditionalizeTW(settings.zh.opencc_config, self.store.zh_rules())

        self._task: asyncio.Task | None = None
        self._running = False
        self._busy = asyncio.Lock()
        self.last_lang: str | None = self.store.last_language(self.session_id)

        self.store.start_session(
            self.session_id,
            settings.session.routing,
            {"llm_profile": settings.llm.profile, "asr_backend": settings.asr.backend},
        )

    # ── 수명주기 ────────────────────────────────────────────────────────
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.capture.loop = loop
        self.capture.start()
        self.playback.start(loop)
        self._running = True
        self._task = asyncio.create_task(self._frame_loop(), name="frame-loop")
        asyncio.create_task(self._probe_services(), name="probe")
        log.info("orchestrator.started", session=self.session_id, mode=self.s.session.mode)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.capture.stop()
        self.playback.stop()
        for c in (self.asr, self.verify, self.llm, self.tts):
            await c.aclose()
        self.ring.close()
        self.store.close()
        log.info("orchestrator.stopped")

    async def _probe_services(self) -> None:
        for c in (self.asr, self.verify, self.llm, self.tts):
            h = await c.health()
            self.bus.emit("service", name=c.name, ok=bool(h.get("ok")), detail=h)

    # ── push-to-talk (§4: 초기 구현) ────────────────────────────────────
    def ptt_down(self) -> None:
        if self.machine.state is not State.IDLE:
            return
        ev = self.seg.force_start()
        if ev.kind == "speech_start":
            self.machine.to(State.LISTENING, "ptt")

    def ptt_up(self) -> None:
        if self.machine.state is not State.LISTENING:
            return
        ev = self.seg.force_end()
        if ev.utterance is not None:
            asyncio.create_task(self._on_utterance(ev.utterance))
        else:
            self.machine.to(State.IDLE, "ptt_empty")

    # ── 프레임 루프 ─────────────────────────────────────────────────────
    async def _frame_loop(self) -> None:
        auto = self.s.session.mode == "auto"
        level_tick = 0
        while self._running:
            frame = await self.capture.frames.get()

            # SPEAKING 중에는 캡처 자체가 게이팅되지만, 큐에 남은 잔여 프레임도 버린다.
            if self.machine.state is State.SPEAKING:
                continue

            if self.machine.state is State.PROCESSING:
                continue

            if not auto and self.machine.state is State.IDLE:
                # PTT 모드: 전이는 키 입력이 결정한다. VAD 는 돌리지 않고
                # 프리롤 링만 채워둔다 (§5.1 은 PTT 에서도 그대로 적용된다).
                self.seg.prime_preroll(frame.pcm)
                continue

            ev = self.seg.feed(frame.pcm)

            level_tick += 1
            if level_tick % 4 == 0:
                self.bus.emit("level", prob=round(ev.prob, 3), rms=_rms(frame.pcm))

            if ev.kind == "speech_start" and self.machine.state is State.IDLE:
                self.machine.to(State.LISTENING, "vad")
            elif ev.kind == "speech_end" and ev.utterance is not None:
                await self._on_utterance(ev.utterance)

    async def _on_utterance(self, utt) -> None:
        if self._busy.locked():
            log.warning("turn.dropped_busy")
            return
        async with self._busy:
            try:
                await self._process(utt)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — 어떤 실패든 IDLE 로 복귀해야 한다
                log.exception("turn.crashed", error=repr(e))
                self.bus.emit("error", where="turn", message=str(e))
            finally:
                self._to_idle("turn_end")

    # ── 한 턴 ───────────────────────────────────────────────────────────
    async def _process(self, utt) -> None:
        m = TurnMetrics()
        m.mark("eou")
        if self.machine.state is State.LISTENING:
            self.machine.to(State.PROCESSING, "eou")

        self.bus.emit(
            "eou",
            turn_id=m.turn_id,
            seconds=round(utt.pcm.size / utt.sample_rate, 2),
            preroll_ms=round(utt.preroll_ms),
            ended_by=utt.ended_by,
        )

        pcm = self._denoise(utt, m)
        m.audio_seconds = round(pcm.size / self.s.shm.sample_rate, 3)

        if pcm.size == 0:
            m.outcome = "empty_asr"
            self._finish(m, None, None, None)
            return

        ref = self.ring.publish(pcm)

        # ── 1차 ASR ────────────────────────────────────────────────────
        history = self.store.recent_turns(self.session_id, self.s.context.history_turns)
        expect_tw = "zh-TW" in self.s.session.languages
        ctx = build_asr_context(history, self.store.all_glossary()[:40], expect_tw)

        try:
            asr = await self.asr.transcribe(ref, context=ctx, language_hint=None)
        except (ServiceTimeout, ServiceError) as e:
            log.error("asr.failed", error=repr(e))
            self.bus.emit("error", where="asr", message=str(e))
            m.outcome = "empty_asr"
            self._finish(m, None, None, None)
            return
        m.mark("asr_done")
        m.asr_avg_logprob = round(asr.best_avg_logprob, 4)

        if asr.is_empty:
            # §10: 빈 결과는 무음 처리. 재생 없이 IDLE.
            m.outcome = "empty_asr"
            self.bus.emit("asr", text="", empty=True)
            self._finish(m, None, None, None)
            return

        # ── LID 결정 (§5, §10) ─────────────────────────────────────────
        dec = decide_language(
            asr.language,
            asr.language_confidence,
            m.audio_seconds or 0.0,
            self.s.asr.lid,
            self.last_lang,
            self.s.session.languages,
        )
        m.lang_detected, m.lang_source, m.lid_confidence = dec.lang, dec.source, dec.confidence
        self.last_lang = dec.lang
        self.bus.emit(
            "lang", lang=dec.lang, source=dec.source, confidence=dec.confidence, note=dec.note
        )

        n_best = [self._maybe_tw(t, dec.lang, "asr") for t in asr.texts]
        self.bus.emit("asr", text=n_best[0], n_best=n_best, avg_logprob=m.asr_avg_logprob)

        # ── 조건부 교차 검증 (§5.5) ────────────────────────────────────
        verify_text: str | None = None
        if self.s.asr_verify.enabled:
            fire, why = should_cross_verify(
                asr.best_avg_logprob,
                self.s.asr.avg_logprob_threshold,
                n_best,
                m.audio_seconds or 0.0,
            )
            if fire:
                m.cross_verify_fired = True
                self.bus.emit("verify", state="running", reason=why)
                try:
                    v = await self.verify.transcribe(ref, language=dec.lang)
                    verify_text = self._maybe_tw(v.text, dec.lang, "asr")
                    m.cross_verify_divergent = is_divergent(
                        n_best[0], verify_text, self.s.asr_verify.divergence_cer
                    )
                    self.bus.emit(
                        "verify",
                        state="done",
                        text=verify_text,
                        divergent=m.cross_verify_divergent,
                        cer=round(cer(n_best[0], verify_text), 3),
                    )
                except (ServiceTimeout, ServiceError) as e:
                    # 검증 실패는 치명적이지 않다. 1차 결과로 그냥 간다.
                    log.warning("verify.failed", error=repr(e))
                    self.bus.emit("verify", state="failed", message=str(e))

        # ── 타깃 라우팅 ────────────────────────────────────────────────
        targets = route_targets(dec.lang, self.s.session)
        if not targets:
            m.outcome = "ok"
            m.notes["no_target"] = True
            self._finish(m, n_best[0], None, None)
            return
        m.target_lang = ",".join(targets)
        m.llm_profile = self.s.llm.profile

        translation: str | None = None
        for i, tgt in enumerate(targets):
            translation = await self._translate_and_speak(
                m, n_best, verify_text, dec.lang, tgt, history, first=(i == 0)
            )

        self._finish(m, n_best[0], translation, dec.lang)

    # ── 번역 + TTS ──────────────────────────────────────────────────────
    async def _translate_and_speak(
        self,
        m: TurnMetrics,
        n_best: list[str],
        verify_text: str | None,
        src_lang: str,
        tgt_lang: str,
        history,
        first: bool,
    ) -> str | None:
        glossary = self.store.glossary_for(
            src_lang, tgt_lang, n_best[0], self.s.context.glossary_max_terms
        )
        messages = build_translate_messages(
            n_best=n_best,
            source_lang=src_lang,
            target_lang=tgt_lang,
            history=history,
            glossary=glossary,
            verify_hypothesis=verify_text,
            verify_divergent=bool(m.cross_verify_divergent),
        )

        clause_q: asyncio.Queue[str | None] = asyncio.Queue()
        first_clause = asyncio.Event()
        streamer = ClauseStreamer()
        stats = StreamStats()

        self.playback.begin_turn()
        watcher = asyncio.create_task(self._watch_first_audio(m))
        speaker = asyncio.create_task(self._tts_worker(clause_q, tgt_lang, m))
        pump = asyncio.create_task(
            self._pump_llm(messages, streamer, clause_q, first_clause, m, stats, tgt_lang)
        )

        # §10 LLM 타임아웃 3초 — '첫 절'까지의 시간으로 잰다.
        waiter = asyncio.create_task(first_clause.wait())
        done, _ = await asyncio.wait(
            {waiter, pump}, timeout=self.s.llm.timeout_s, return_when=asyncio.FIRST_COMPLETED
        )

        if waiter not in done and not first_clause.is_set():
            waiter.cancel()
            pump.cancel()
            await clause_q.put(None)
            await asyncio.gather(speaker, watcher, return_exceptions=True)
            m.outcome = "llm_timeout"
            self.playback.flush()
            self.bus.emit("error", where="llm", message=f"first clause > {self.s.llm.timeout_s}s")
            self.bus.emit("translation", text=None, timeout=True)
            return None

        waiter.cancel()
        try:
            await pump
        except asyncio.CancelledError:
            pass
        except (ServiceTimeout, ServiceError) as e:
            log.error("llm.failed", error=repr(e))
            self.bus.emit("error", where="llm", message=str(e))
            m.outcome = "llm_timeout"

        await clause_q.put(None)
        await asyncio.gather(speaker, return_exceptions=True)

        # 큐 소진까지 기다린다. 여기까지가 SPEAKING.
        drained = await self.playback.wait_drained(timeout=60.0)
        if not drained:
            log.warning("playback.drain_timeout")
        m.mark("queue_drained")
        watcher.cancel()

        m.output_tokens = stats.tokens
        m.tok_per_s = stats.tok_per_s
        text = self._maybe_tw(streamer.translation, tgt_lang, "translation")
        self.bus.emit("translation", text=text, lang=tgt_lang, tok_per_s=stats.tok_per_s)
        return text

    async def _pump_llm(
        self,
        messages,
        streamer: ClauseStreamer,
        clause_q: asyncio.Queue,
        first_clause: asyncio.Event,
        m: TurnMetrics,
        stats: StreamStats,
        tgt_lang: str,
    ) -> None:
        async for delta in self.llm.stream_chat(messages, stats):
            for c in streamer.push(delta):
                if not first_clause.is_set():
                    m.mark("first_clause")
                    first_clause.set()
                    if self.machine.state is State.PROCESSING:
                        self.machine.to(State.SPEAKING, "first_clause")
                await clause_q.put(self._maybe_tw(c, tgt_lang, "translation"))
            self.bus.emit("translation_delta", text=streamer.translation)
            if streamer.stopped:
                break
        for c in streamer.flush():
            if not first_clause.is_set():
                m.mark("first_clause")
                first_clause.set()
                if self.machine.state is State.PROCESSING:
                    self.machine.to(State.SPEAKING, "first_clause")
            await clause_q.put(self._maybe_tw(c, tgt_lang, "translation"))

    async def _tts_worker(self, clause_q: asyncio.Queue, lang: str, m: TurnMetrics) -> None:
        while True:
            clause = await clause_q.get()
            if clause is None:
                return
            if not clause.strip():
                continue
            self.bus.emit("clause", text=clause)
            try:
                async for chunk in self.tts.synthesize(clause, lang):
                    self.playback.enqueue(chunk, self.s.tts.sample_rate)
            except (ServiceTimeout, ServiceError) as e:
                # §10 TTS 실패 — 서비스 내부 MeloTTS 폴백까지 실패한 경우.
                log.error("tts.failed", error=repr(e), clause=clause[:40])
                m.outcome = "tts_failed"
                self.bus.emit("error", where="tts", message=str(e))

    async def _watch_first_audio(self, m: TurnMetrics) -> None:
        await self.playback.first_packet.wait()
        m.mark("first_audio")
        self.bus.emit("first_audio", ms=m.rel_ms("first_audio"))

    # ── 잡음 제거 ───────────────────────────────────────────────────────
    def _denoise(self, utt, m: TurnMetrics) -> np.ndarray:
        """DFN3 은 48kHz 발화 단위로 돌린다.

        스트리밍 프레임 단위 처리는 libdf 의 프레임 API 가 필요하고 상태 관리가
        까다롭다. 발화 단위 처리의 실제 소요를 여기서 재서 로그에 남기고,
        §6 프런트엔드 예산 100ms 를 넘기면 그때 스트리밍으로 옮긴다.
        """
        if self.denoiser is None or self.denoiser.name == "none":
            return utt.pcm

        t0 = time.perf_counter()
        try:
            raw48 = self.capture.tail48(utt.pcm.size)
            if raw48.size < self.denoiser.rate // 10:  # 0.1초도 안 되면 원본 사용
                return utt.pcm
            from ..audio.resample import resample_once

            clean48 = self.denoiser(raw48)
            out = resample_once(clean48, self.denoiser.rate, self.s.shm.sample_rate)
        except Exception as e:  # noqa: BLE001
            log.warning("denoise.failed", error=repr(e))
            return utt.pcm
        dt = (time.perf_counter() - t0) * 1000
        m.notes["denoise_ms"] = round(dt, 1)
        if dt > self.s.budget_ms.frontend:
            log.warning("denoise.over_budget", ms=round(dt, 1), budget=self.s.budget_ms.frontend)
        return out

    # ── 번체 후처리 (§5) ────────────────────────────────────────────────
    def _maybe_tw(self, text: str, lang: str, stage: str) -> str:
        if lang != "zh-TW" or stage not in self.s.zh.apply_to or not text:
            return text
        out = self.zh(text)
        if looks_simplified(out):
            log.warning("zh.simplified_leak", stage=stage, sample=out[:40])
        return out

    # ── 마무리 ──────────────────────────────────────────────────────────
    def _finish(
        self, m: TurnMetrics, source_text: str | None, translation: str | None, src_lang: str | None
    ) -> None:
        rec = self.turnlog.write(m)
        self.store.add_turn(
            turn_id=m.turn_id,
            session_id=self.session_id,
            src_lang=src_lang or m.lang_detected,
            tgt_lang=m.target_lang,
            source_text=source_text,
            translation=translation,
            lang_source=m.lang_source,
            lid_confidence=m.lid_confidence,
            asr_avg_logprob=m.asr_avg_logprob,
            cross_verified=m.cross_verify_fired,
            audio_seconds=m.audio_seconds,
            outcome=m.outcome,
        )
        log.info("turn", **rec)
        self.bus.emit("turn", **rec)
        if rec.get("over_budget_ms"):
            # §6: 예산 초과 시 어느 단계가 원인인지 특정해 보고한다.
            self.bus.emit("budget", over=rec["over_budget_ms"], stages=rec["stages_ms"])

    def _to_idle(self, reason: str) -> None:
        self.machine.force_idle(reason)

    def _on_state_change(self, prev: State, cur: State, reason: str) -> None:
        # 반이중 게이팅은 오직 여기서만 (§4)
        if cur is State.SPEAKING:
            self.capture.close_gate()
        elif cur is State.IDLE:
            self.capture.open_gate()
            self.seg.reset()
        log.info("state", **{"from": prev.value, "to": cur.value, "reason": reason})
        self.bus.emit("state", state=cur.value, prev=prev.value, reason=reason)


def _rms(x: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12)), 5)
