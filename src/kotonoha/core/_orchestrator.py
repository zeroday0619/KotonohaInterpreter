"""Orchestrator — state machine, language routing, quality gate, clause
streaming, failure handling.

One turn:

    EOU -> (denoise) -> shm publish -> ASR (N-best 5 + LID)
        -> LID decision/fallback -> conditional cross-verification
        -> single-pass correct+translate streaming -> clause-wise TTS -> playback

§10 is transposed here directly. For an interpreter, stopping is worse than
being wrong, so every stage has a timeout and a fallback and every path leads
back to IDLE.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, ClassVar

import numpy as np

from kotonoha._config import Settings
from kotonoha._logging_setup import get_logger
from kotonoha._metrics import TurnLog, TurnMetrics
from kotonoha._shmring import AudioRing
from kotonoha._transport import AudioPayload
from kotonoha._typing import override
from kotonoha.clients._base import ServiceError, ServiceTimeout
from kotonoha.clients._build import ServiceGroup, build_service_group
from kotonoha.clients._llm import GenerationStatistics
from kotonoha.clients._router import FailoverClient
from kotonoha.core._clauses import ClauseStreamer
from kotonoha.core._events import EventBus
from kotonoha.core._lid import decide_language, decide_typed_language, route_targets
from kotonoha.core._quality import character_error_rate, is_divergent, should_cross_verify
from kotonoha.core._state import Machine, State
from kotonoha.core._zh import TraditionalChineseConverter, looks_simplified
from kotonoha.prompts._asr_prompt import build_asr_context
from kotonoha.prompts._translate import build_translate_messages
from kotonoha.store._db import Store

log = get_logger(__name__)


class Orchestrator:
    __slots__: ClassVar[tuple[str, ...]] = (
        "__dict__",
        "_busy",
        "_frame_task",
        "_running",
        "asr_verifier",
        "capture",
        "denoiser",
        "event_bus",
        "language_model",
        "last_language",
        "machine",
        "playback",
        "primary_asr",
        "ring",
        "segmenter",
        "services",
        "session_id",
        "settings",
        "store",
        "text_to_speech",
        "traditionalizer",
        "turn_log",
    )
    settings: Settings
    capture: Any
    segmenter: Any
    playback: Any
    denoiser: Any | None
    session_id: str
    event_bus: EventBus
    machine: Machine
    ring: AudioRing
    services: ServiceGroup
    primary_asr: FailoverClient
    asr_verifier: FailoverClient
    language_model: FailoverClient
    text_to_speech: FailoverClient
    store: Store
    turn_log: TurnLog
    traditionalizer: TraditionalChineseConverter
    last_language: str | None
    _frame_task: asyncio.Task[None] | None
    _running: bool
    _busy: asyncio.Lock

    @override
    def __init__(
        self,
        /,
        settings: Settings,
        capture: Any,
        segmenter: Any,
        playback: Any,
        denoiser: Any = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.capture = capture
        self.segmenter = segmenter
        self.playback = playback
        self.denoiser = denoiser
        self.session_id = session_id or uuid.uuid4().hex[:12]

        self.event_bus = EventBus()
        self.machine = Machine(on_change=self._on_state_change)

        self.ring = AudioRing.create(
            name=settings.shm.name,
            slots=settings.shm.slots,
            slot_seconds=settings.shm.slot_seconds,
            sample_rate=settings.shm.sample_rate,
        )

        # Roles are routed to the A6000 or to the on-board service according to
        # perf_mode, each with automatic failover (clients/router.py).
        self.services = build_service_group(settings, on_change=self._on_placement_change)
        self.primary_asr = self.services.asr
        self.asr_verifier = self.services.asr_verify
        self.language_model = self.services.llm
        self.text_to_speech = self.services.tts

        self.store = Store(settings.resolve(settings.store.path))
        self.turn_log = TurnLog(
            settings.resolve(settings.logging.turn_log_path),
            settings.budget_ms,
        )
        self.traditionalizer = TraditionalChineseConverter(
            settings.zh.opencc_config,
            self.store.zh_rules(),
        )

        self._frame_task = None
        self._running = False
        self._busy = asyncio.Lock()
        self.last_language = self.store.last_language(self.session_id)

        self.store.start_session(
            self.session_id,
            settings.session.routing,
            {
                "llm_profile": settings.llm.profile,
                "asr_backend": settings.asr.backend,
                "perf_mode": settings.perf_mode,
                "placement": settings.resolved_placement(),
            },
        )

    # -- lifecycle -------------------------------------------------------
    async def start(
        self,
        /,
    ) -> None:
        loop = asyncio.get_running_loop()
        self.capture.loop = loop
        await asyncio.to_thread(self.capture.start)
        await asyncio.to_thread(self.playback.start, loop)
        self._running = True
        self._frame_task = asyncio.create_task(self._frame_loop(), name="frame-loop")
        self.services.start_probes()
        asyncio.create_task(self._probe_services(), name="probe")
        log.info(
            "orchestrator.started",
            session=self.session_id,
            mode=self.settings.session.mode,
            perf_mode=self.settings.perf_mode,
            placement=self.services.placement,
        )
        if self.settings.audio_leaves_device:
            # Worth stating plainly: in this mode the utterance audio is sent to
            # another machine. §1 asked for the device to be self-contained.
            log.warning("privacy.audio_offbox", placement=self.services.placement)
            self.event_bus.emit(
                "privacy",
                audio_leaves_device=True,
                placement=self.services.placement,
            )

    async def stop(
        self,
        /,
    ) -> None:
        self._running = False
        if self._frame_task:
            self._frame_task.cancel()
            try:
                await self._frame_task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.capture.stop)
        await asyncio.to_thread(self.playback.stop)
        await self.services.aclose()
        await asyncio.to_thread(self.ring.close)
        await asyncio.to_thread(self.store.close)
        log.info("orchestrator.stopped")

    async def _probe_services(
        self,
        /,
    ) -> None:
        for role in self.services.all():
            health = await role.active.health()
            self.event_bus.emit(
                "service",
                name=role.name,
                ok=bool(health.get("ok")),
                side=role.side,
                degraded=role.degraded,
                detail=health,
            )

    def _on_placement_change(
        self,
        /,
        role: str,
        side: str,
        reason: str,
    ) -> None:
        """A role moved between the A6000 and the on-board service."""
        self.event_bus.emit("placement", role=role, side=side, reason=reason)

    # -- push-to-talk (§4: the initial implementation) ---------------------
    def ptt_down(
        self,
        /,
    ) -> None:
        if self.machine.state is not State.IDLE:
            return
        event = self.segmenter.force_start()
        if event.kind == "speech_start":
            self.machine.to(State.LISTENING, "ptt")

    def ptt_up(
        self,
        /,
    ) -> None:
        if self.machine.state is not State.LISTENING:
            return
        event = self.segmenter.force_end()
        if event.utterance is not None:
            asyncio.create_task(self._on_utterance(event.utterance))
        else:
            self.machine.to(State.IDLE, "ptt_empty")

    # -- frame loop ------------------------------------------------------
    async def _frame_loop(
        self,
        /,
    ) -> None:
        level_tick = 0
        while self._running:
            frame = await self.capture.frames.get()

            # Capture is already gated during SPEAKING, but drop any frame that
            # was still sitting in the queue as well.
            if self.machine.state is State.SPEAKING:
                continue

            if self.settings.session.mode == "text":
                continue

            if self.machine.state is State.PROCESSING:
                continue

            if self.settings.session.mode != "auto" and self.machine.state is State.IDLE:
                # PTT mode: the key decides the transition. Skip the VAD and
                # just keep the preroll ring filled (§5.1 applies to PTT too).
                self.segmenter.prime_preroll(frame.pcm)
                continue

            event = self.segmenter.feed(frame.pcm)

            level_tick += 1
            if level_tick % 4 == 0:
                self.event_bus.emit(
                    "level",
                    prob=round(event.probability, 3),
                    rms=_rms(frame.pcm),
                )

            if event.kind == "speech_start" and self.machine.state is State.IDLE:
                self.machine.to(State.LISTENING, "vad")
            elif event.kind == "speech_end" and event.utterance is not None:
                await self._on_utterance(event.utterance)

    async def _on_utterance(
        self,
        /,
        utterance: Any,
    ) -> None:
        if self._busy.locked():
            log.warning("turn.dropped_busy")
            return
        async with self._busy:
            try:
                await self._process(utterance)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - every failure must reach IDLE
                log.exception("turn.crashed", error=repr(error))
                self.event_bus.emit("error", where="turn", message=str(error))
            finally:
                self._to_idle("turn_end")

    # -- one turn --------------------------------------------------------
    async def _process(
        self,
        /,
        utterance: Any,
    ) -> None:
        metrics = TurnMetrics()
        metrics.mark("eou")
        metrics.perf_mode = self.settings.perf_mode
        metrics.placement = dict(self.services.placement)
        failover_baseline = self._failover_total()
        if self.machine.state is State.LISTENING:
            self.machine.to(State.PROCESSING, "eou")

        self.event_bus.emit(
            "eou",
            turn_id=metrics.turn_id,
            seconds=round(utterance.pcm.size / utterance.sample_rate, 2),
            preroll_ms=round(utterance.preroll_ms),
            ended_by=utterance.ended_by,
        )

        pcm = await asyncio.to_thread(self._denoise, utterance, metrics)
        metrics.audio_seconds = round(pcm.size / self.settings.shm.sample_rate, 3)

        if pcm.size == 0:
            metrics.outcome = "empty_asr"
            await self._finish(metrics, None, None, None, failover_baseline)
            return

        # Both carriers are prepared: the shm reference for an on-board service
        # and the PCM itself for one on the A6000. The client picks (transport.py).
        payload = AudioPayload(
            pcm=pcm,
            audio_reference=self.ring.publish(pcm),
            sample_rate=self.settings.shm.sample_rate,
        )

        # -- primary ASR ------------------------------------------------
        history = await asyncio.to_thread(
            self.store.recent_turns,
            self.session_id,
            self.settings.context.history_turns,
        )
        expect_tw = "zh-TW" in self.settings.session.languages
        glossary = await asyncio.to_thread(self.store.all_glossary)
        context = build_asr_context(history, glossary[:40], expect_tw)

        try:
            asr_result = await self.primary_asr.run(
                lambda client: client.transcribe(
                    payload,
                    context=context,
                    language_hint=None,
                )
            )
        except (ServiceTimeout, ServiceError) as error:
            log.error("asr.failed", error=repr(error))
            self.event_bus.emit("error", where="asr", message=str(error))
            metrics.outcome = "empty_asr"
            await self._finish(metrics, None, None, None, failover_baseline)
            return
        metrics.mark("asr_done")
        metrics.asr_avg_logprob = round(asr_result.best_avg_logprob, 4)

        if asr_result.is_empty:
            # §10: an empty result is treated as silence. Back to IDLE, nothing played.
            metrics.outcome = "empty_asr"
            self.event_bus.emit("asr", text="", empty=True)
            await self._finish(metrics, None, None, None, failover_baseline)
            return

        # -- LID decision (§5, §10) ---------------------------------------
        decision = decide_language(
            asr_result.language,
            asr_result.language_confidence,
            metrics.audio_seconds or 0.0,
            self.settings.asr.lid,
            self.last_language,
            self.settings.session.languages,
        )
        metrics.lang_detected = decision.language
        metrics.lang_source = decision.source
        metrics.lid_confidence = decision.confidence
        self.last_language = decision.language
        self.event_bus.emit(
            "lang",
            lang=decision.language,
            source=decision.source,
            confidence=decision.confidence,
            note=decision.note,
        )

        n_best = [
            self._apply_traditional_chinese(hypothesis, decision.language, "asr")
            for hypothesis in asr_result.texts
        ]
        self.event_bus.emit(
            "asr",
            text=n_best[0],
            n_best=n_best,
            avg_logprob=metrics.asr_avg_logprob,
        )

        # -- conditional cross-verification (§5.5) -------------------------
        verify_text: str | None = None
        if self.settings.asr_verify.enabled:
            if self.settings.asr_verify.mode == "always":
                # Affordable on the A6000, so take the accuracy for free.
                verify_required, verify_reason = True, "mode=always"
            else:
                verify_required, verify_reason = should_cross_verify(
                    asr_result.best_avg_logprob,
                    self.settings.asr.avg_logprob_threshold,
                    n_best,
                    metrics.audio_seconds or 0.0,
                )
            if verify_required:
                metrics.cross_verify_fired = True
                self.event_bus.emit("verify", state="running", reason=verify_reason)
                try:
                    verification_result = await self.asr_verifier.run(
                        lambda client: client.transcribe(
                            payload,
                            language=decision.language,
                        )
                    )
                    verify_text = self._apply_traditional_chinese(
                        verification_result.text,
                        decision.language,
                        "asr",
                    )
                    metrics.cross_verify_divergent = is_divergent(
                        n_best[0], verify_text, self.settings.asr_verify.divergence_cer
                    )
                    self.event_bus.emit(
                        "verify",
                        state="done",
                        text=verify_text,
                        divergent=metrics.cross_verify_divergent,
                        cer=round(character_error_rate(n_best[0], verify_text), 3),
                    )
                except (ServiceTimeout, ServiceError) as error:
                    # A failed verification is not fatal. Carry on with the primary result.
                    log.warning("verify.failed", error=repr(error))
                    self.event_bus.emit("verify", state="failed", message=str(error))

        await self._route_and_translate(
            metrics,
            n_best,
            verify_text,
            decision.language,
            history,
            failover_baseline,
        )

    # -- routing shared by the spoken and typed paths ----------------------
    async def _route_and_translate(
        self,
        /,
        metrics: TurnMetrics,
        n_best: list[str],
        verify_text: str | None,
        source_language: str,
        history: Any,
        failover_baseline: int,
    ) -> None:
        targets = route_targets(source_language, self.settings.session)
        if not targets:
            metrics.outcome = "ok"
            metrics.notes["no_target"] = True
            await self._finish(metrics, n_best[0], None, None, failover_baseline)
            return
        metrics.target_lang = ",".join(targets)
        metrics.llm_profile = self.settings.llm.profile

        translation: str | None = None
        for target_index, target_language in enumerate(targets):
            translation = await self._translate_and_speak(
                metrics,
                n_best,
                verify_text,
                source_language,
                target_language,
                history,
                first=(target_index == 0),
            )

        await self._finish(
            metrics,
            n_best[0],
            translation,
            source_language,
            failover_baseline,
        )

    # -- typed input --------------------------------------------------------
    def text_mode_active(
        self,
        /,
    ) -> bool:
        return self.settings.session.mode == "text"

    def set_text_mode(
        self,
        /,
        active: bool,
        previous: str = "push_to_talk",
    ) -> str:
        """Switch the keyboard in or out as the input source.

        The microphone is closed while typing. In automatic mode the VAD would
        otherwise segment room noise into a turn while the operator is still
        composing, and in push-to-talk the space bar belongs to the text field.
        """
        self.settings.session.mode = "text" if active else previous
        if active:
            self.capture.close_gate()
        elif self.machine.state is not State.SPEAKING:
            self.capture.open_gate()
        self.segmenter.reset()
        log.info("session.input_mode", mode=self.settings.session.mode)
        return self.settings.session.mode

    async def submit_text(
        self,
        /,
        text: str,
        source_language: str | None = None,
    ) -> bool:
        """Run one turn from typed text, skipping capture, ASR and verification.

        Returns False when the submission was refused, which happens for blank
        input and while another turn is still running.
        """
        text = text.strip()
        if not text:
            return False
        if self._busy.locked() or self.machine.state is not State.IDLE:
            log.warning("text.dropped_busy", state=self.machine.state.value)
            self.event_bus.emit("error", where="text", message="busy")
            return False
        async with self._busy:
            await self._process_text(text, source_language)
        return True

    async def _process_text(
        self,
        /,
        text: str,
        source_language: str | None,
    ) -> None:
        metrics = TurnMetrics()
        metrics.input_mode = "text"
        # There is no end of utterance, so the clock starts at submission. ASR is
        # skipped rather than measured, which the marks show as a zero-length stage.
        metrics.mark("eou")
        metrics.mark("asr_done")
        metrics.perf_mode = self.settings.perf_mode
        metrics.placement = dict(self.services.placement)
        failover_baseline = self._failover_total()
        self.machine.to(State.PROCESSING, "text")

        self.event_bus.emit("text_submitted", turn_id=metrics.turn_id, text=text)

        decision = decide_typed_language(
            text,
            source_language or self.settings.session.text_source_language,
            self.last_language,
            self.settings.session.languages,
        )
        metrics.lang_detected = decision.language
        metrics.lang_source = decision.source
        metrics.lid_confidence = decision.confidence
        self.last_language = decision.language
        self.event_bus.emit(
            "lang",
            lang=decision.language,
            source=decision.source,
            confidence=decision.confidence,
            note=decision.note,
        )

        # Typed Simplified input is converted for the same reason ASR output is:
        # this device produces Taiwanese Traditional (§5).
        source = self._apply_traditional_chinese(text, decision.language, "asr")
        self.event_bus.emit("asr", text=source, n_best=[source], avg_logprob=None)

        history = await asyncio.to_thread(
            self.store.recent_turns,
            self.session_id,
            self.settings.context.history_turns,
        )
        await self._route_and_translate(
            metrics,
            [source],
            None,
            decision.language,
            history,
            failover_baseline,
        )

    # -- translation + TTS -------------------------------------------------
    async def _translate_and_speak(
        self,
        /,
        metrics: TurnMetrics,
        n_best: list[str],
        verify_text: str | None,
        source_language: str,
        target_language: str,
        history: Any,
        first: bool,
    ) -> str | None:
        glossary = await asyncio.to_thread(
            self.store.glossary_for,
            source_language,
            target_language,
            n_best[0],
            self.settings.context.glossary_max_terms,
        )
        messages = build_translate_messages(
            n_best,
            source_language=source_language,
            target_language=target_language,
            history=history,
            glossary=glossary,
            verify_hypothesis=verify_text,
            verify_divergent=bool(metrics.cross_verify_divergent),
        )

        clause_queue: asyncio.Queue[str | None] = asyncio.Queue()
        first_clause = asyncio.Event()
        streamer = ClauseStreamer()
        generation_statistics = GenerationStatistics()

        self.playback.begin_turn()
        first_audio_task = asyncio.create_task(self._watch_first_audio(metrics))
        speaker_task = asyncio.create_task(
            self._tts_worker(clause_queue, target_language, metrics)
        )
        language_model_task = asyncio.create_task(
            self._stream_language_model(
                messages,
                streamer,
                clause_queue,
                first_clause,
                metrics,
                generation_statistics,
                target_language,
            )
        )

        # §10 LLM timeout of 3 s, measured as time-to-first-clause.
        first_clause_task = asyncio.create_task(first_clause.wait())
        completed, _pending = await asyncio.wait(
            {first_clause_task, language_model_task},
            timeout=self.settings.llm.timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if first_clause_task not in completed and not first_clause.is_set():
            first_clause_task.cancel()
            language_model_task.cancel()
            await clause_queue.put(None)
            await asyncio.gather(
                speaker_task,
                first_audio_task,
                return_exceptions=True,
            )
            metrics.outcome = "llm_timeout"
            self.playback.flush()
            self.event_bus.emit(
                "error",
                where="llm",
                message=f"first clause > {self.settings.llm.timeout_s}s",
            )
            self.event_bus.emit("translation", text=None, timeout=True)
            return None

        first_clause_task.cancel()
        try:
            await language_model_task
        except asyncio.CancelledError:
            pass
        except (ServiceTimeout, ServiceError) as error:
            log.error("llm.failed", error=repr(error))
            self.event_bus.emit("error", where="llm", message=str(error))
            metrics.outcome = "llm_timeout"

        await clause_queue.put(None)
        await asyncio.gather(speaker_task, return_exceptions=True)

        # Wait for the queue to drain. SPEAKING lasts until here.
        drained = await self.playback.wait_drained(timeout=60.0)
        if not drained:
            log.warning("playback.drain_timeout")
        metrics.mark("queue_drained")
        first_audio_task.cancel()

        metrics.output_tokens = generation_statistics.token_count
        metrics.tok_per_s = generation_statistics.tokens_per_second
        metrics.placement = dict(self.services.placement)
        text = self._apply_traditional_chinese(
            streamer.translation,
            target_language,
            "translation",
        )
        self.event_bus.emit(
            "translation",
            text=text,
            lang=target_language,
            tok_per_s=generation_statistics.tokens_per_second,
        )
        return text

    async def _stream_language_model(
        self,
        /,
        messages: Any,
        streamer: ClauseStreamer,
        clause_queue: asyncio.Queue[str | None],
        first_clause: asyncio.Event,
        metrics: TurnMetrics,
        generation_statistics: GenerationStatistics,
        target_language: str,
    ) -> None:
        async for delta in self.language_model.stream(
            lambda client: client.stream_chat(messages, generation_statistics)
        ):
            for clause in streamer.push(delta):
                if not first_clause.is_set():
                    metrics.mark("first_clause")
                    first_clause.set()
                    if self.machine.state is State.PROCESSING:
                        self.machine.to(State.SPEAKING, "first_clause")
                await clause_queue.put(
                    self._apply_traditional_chinese(clause, target_language, "translation")
                )
            self.event_bus.emit("translation_delta", text=streamer.translation)
            if streamer.stopped:
                break
        for clause in streamer.flush():
            if not first_clause.is_set():
                metrics.mark("first_clause")
                first_clause.set()
                if self.machine.state is State.PROCESSING:
                    self.machine.to(State.SPEAKING, "first_clause")
            await clause_queue.put(
                self._apply_traditional_chinese(clause, target_language, "translation")
            )

    async def _tts_worker(
        self,
        /,
        clause_queue: asyncio.Queue[str | None],
        language: str,
        metrics: TurnMetrics,
    ) -> None:
        while True:
            clause = await clause_queue.get()
            if clause is None:
                return
            if not clause.strip():
                continue
            self.event_bus.emit("clause", text=clause)
            try:
                # Bind the clause explicitly: the router may re-invoke this
                # factory on the fallback, and by then the loop has moved on.
                stream_factory = lambda client, text=clause: client.synthesize(  # noqa: E731
                    text,
                    language,
                )
                async for chunk in self.text_to_speech.stream(stream_factory):
                    self.playback.enqueue(chunk, self.settings.tts.sample_rate)
            except (ServiceTimeout, ServiceError) as error:
                # The router may retry a remote request against the resident
                # onboard vLLM-Omni service before this failure reaches the turn.
                log.error("tts.failed", error=repr(error), clause=clause[:40])
                metrics.outcome = "tts_failed"
                self.event_bus.emit("error", where="tts", message=str(error))

    async def _watch_first_audio(
        self,
        /,
        metrics: TurnMetrics,
    ) -> None:
        await self.playback.first_packet.wait()
        metrics.mark("first_audio")
        self.event_bus.emit("first_audio", ms=metrics.rel_ms("first_audio"))

    # -- noise suppression -------------------------------------------------
    def _denoise(
        self,
        /,
        utterance: Any,
        metrics: TurnMetrics,
    ) -> np.ndarray:
        """Run DFN3 over the whole 48 kHz utterance.

        Per-frame streaming needs libdf's frame API and careful state handling.
        Utterance-level processing remains in place until target-device measurements
        show that it exceeds the 100 ms frontend budget in §6.
        """
        if self.denoiser is None or self.denoiser.name == "none":
            return utterance.pcm

        start_time = time.perf_counter()
        try:
            raw_audio = self.capture.tail48(utterance.pcm.size)
            if raw_audio.size < self.denoiser.rate // 10:
                return utterance.pcm
            from kotonoha.audio._resample import resample_once

            clean_audio = self.denoiser(raw_audio)
            output = resample_once(
                clean_audio,
                self.denoiser.rate,
                self.settings.shm.sample_rate,
            )
        except Exception as error:  # noqa: BLE001
            log.warning("denoise.failed", error=repr(error))
            return utterance.pcm
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics.notes["denoise_ms"] = round(elapsed_ms, 1)
        if elapsed_ms > self.settings.budget_ms.frontend:
            log.warning(
                "denoise.over_budget",
                ms=round(elapsed_ms, 1),
                budget=self.settings.budget_ms.frontend,
            )
        return output

    # -- Traditional Chinese post-processing (§5) -------------------------
    def _apply_traditional_chinese(
        self,
        /,
        text: str,
        language: str,
        stage: str,
    ) -> str:
        if language != "zh-TW" or stage not in self.settings.zh.apply_to or not text:
            return text
        output = self.traditionalizer(text)
        if looks_simplified(output):
            log.warning("zh.simplified_leak", stage=stage, sample=output[:40])
        return output

    # -- wrap-up -----------------------------------------------------------
    def _failover_total(
        self,
        /,
    ) -> int:
        return sum(client.failover_count for client in self.services.all())

    async def _finish(
        self,
        /,
        metrics: TurnMetrics,
        source_text: str | None,
        translation: str | None,
        source_language: str | None,
        failover_baseline: int = 0,
    ) -> None:
        # Report failovers for *this* turn, not the session total.
        metrics.failovers = max(0, self._failover_total() - failover_baseline)
        metrics.placement = dict(self.services.placement)
        record = await self.turn_log.write(metrics)
        stored_at = await asyncio.to_thread(
            self.store.add_turn,
            turn_id=metrics.turn_id,
            session_id=self.session_id,
            src_lang=source_language or metrics.lang_detected,
            tgt_lang=metrics.target_lang,
            source_text=source_text,
            translation=translation,
            lang_source=metrics.lang_source,
            lid_confidence=metrics.lid_confidence,
            asr_avg_logprob=metrics.asr_avg_logprob,
            cross_verified=metrics.cross_verify_fired,
            audio_seconds=metrics.audio_seconds,
            outcome=metrics.outcome,
        )
        log.info("turn", **record)
        self.event_bus.emit("turn", **record)
        if source_text or translation:
            # The panel appends from this rather than re-querying: the row was
            # just written, and a query per turn would run inside the latency budget.
            self.event_bus.emit(
                "history",
                turn_id=metrics.turn_id,
                ts=stored_at,
                src_lang=source_language or metrics.lang_detected,
                tgt_lang=metrics.target_lang,
                source_text=source_text,
                translation=translation,
                outcome=metrics.outcome,
            )
        if record.get("over_budget_ms"):
            # §6: when the budget is blown, report which stage caused it.
            self.event_bus.emit(
                "budget",
                over=record["over_budget_ms"],
                stages=record["stages_ms"],
            )

    def _to_idle(
        self,
        /,
        reason: str,
    ) -> None:
        self.machine.force_idle(reason)

    def _on_state_change(
        self,
        /,
        previous: State,
        current: State,
        reason: str,
    ) -> None:
        # Half-duplex gating happens here and nowhere else (§4).
        if current is State.SPEAKING:
            self.capture.close_gate()
        elif current is State.IDLE:
            self.capture.open_gate()
            self.segmenter.reset()
        log.info(
            "state",
            **{"from": previous.value, "to": current.value, "reason": reason},
        )
        self.event_bus.emit(
            "state",
            state=current.value,
            prev=previous.value,
            reason=reason,
        )


def _rms(
    samples: np.ndarray,
    /,
) -> float:
    return round(
        float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12)),
        5,
    )
