"""Traditional Chinese (Taiwan) message catalog.

Taiwanese vocabulary throughout, matching the glossary policy applied to translation
output: 軟體, 網路, 資訊, 設定.
"""

MESSAGES: dict[str, str] = {
    # -- CLI: application and options -----------------------------------
    "cli.app.help": "逐步口譯用四語離線語音口譯系統",
    "cli.opt.config": "YAML 設定檔路徑",
    "cli.opt.lang": "介面語言：auto、en、ko、ja、zh-TW",
    # -- CLI: commands ---------------------------------------------------
    "cli.run.help": "啟動終端機介面",
    "cli.replay.help": "不使用麥克風，以 WAV 檔執行整條流程",
    "cli.replay.arg.wav": "16 位元 PCM WAV 檔",
    "cli.replay.opt.seconds": "執行時間（秒）",
    "cli.devices.help": "列出音訊裝置",
    "cli.serve.help": "啟動模型服務",
    "cli.serve.arg.service": "要啟動的服務",
    "cli.serve.opt.host": "繫結位址",
    "cli.serve.opt.port": "連接埠。未指定時使用各服務的預設值",
    "cli.glossary.help": "管理詞彙表",
    "cli.glossary.import.help": "從 YAML 匯入詞彙表與繁體轉換規則",
    "cli.glossary.import.arg.path": "詞彙表 YAML 檔",
    "cli.glossary.list.help": "列出已登錄的詞彙",
    "cli.doctor.help": "檢查環境、角色配置與服務狀態",
    "cli.netcheck.help": "量測到外部伺服器的延遲與頻寬",
    "cli.netcheck.opt.samples": "每個角色的量測次數",
    "cli.netcheck.opt.seconds": "探測語句長度（秒）",
    "cli.config.help": "在終端機介面中編輯設定",
    # -- CLI: output -----------------------------------------------------
    "cli.replay.turn_log": "輪次記錄：{path}",
    "cli.devices.default": "預設輸入/輸出：",
    "cli.glossary.imported": "已將 {terms} 筆詞彙與 {rules} 筆規則寫入 {path}",
    "cli.doctor.services": "服務：",
    "cli.doctor.audio_offbox": "！此模式下語音會離開本機",
    "cli.netcheck.remote_disabled": (
        "remote.enabled 為 false。請改用 config/performance.yaml，或啟用後重試。"
    ),
    "cli.netcheck.no_remote_roles": "perf_mode={mode} 下沒有任何角色送往遠端。",
    "cli.netcheck.probe": "probe       {seconds} 秒語句、{encoding}、{size} 位元組",
    "cli.netcheck.failed": "連線失敗：{roles}。這些角色會退回本機執行。",
    "cli.netcheck.overhead": "每輪連線額外耗時估計  {ms} ms",
    "cli.netcheck.budget": "語句結束到首次發聲的預算  {ms} ms",
    "cli.netcheck.over_budget": (
        "  ！連線佔用超過預算的 25%，建議評估 hybrid 模式。"
    ),
    "cli.netcheck.within_budget": (
        "  連線本身在預算內，其餘為模型推論時間。"
    ),
    # -- TUI: chrome -----------------------------------------------------
    "tui.title": "Kotonoha 口譯",
    "tui.subtitle": "工作階段 {session}",
    "tui.pane.source": "原文 (ASR)",
    "tui.pane.target": "譯文",
    "tui.panel.latency": "延遲 (ms)            實測 / 預算",
    "tui.panel.services": "服務",
    "tui.panel.errors": "近期錯誤",
    "tui.mic.open": "開啟",
    "tui.mic.shut": "關閉",
    "tui.audio_offbox": " 語音外送",
    "tui.stage.asr": "ASR (+驗證)",
    "tui.stage.llm": "LLM 首個子句",
    "tui.stage.tts": "TTS 首個封包",
    "tui.stage.total": "EOU→首次發聲",
    "tui.over_budget": "超出：",
    # -- TUI: key bindings -----------------------------------------------
    "tui.key.talk": "說話（切換）",
    "tui.key.mode": "PTT/自動",
    "tui.key.routing": "路由",
    "tui.key.clear": "清除",
    "tui.key.quit": "結束",
    # -- TUI: turn events -------------------------------------------------
    "tui.eou": "[{seconds} 秒、前置 {preroll}ms、{reason}]",
    "tui.asr.empty": "（靜音，未播放即返回）",
    "tui.verify.running": "交叉驗證中（{reason}）",
    "tui.llm.timeout": "（LLM 逾時，僅顯示原文，略過 TTS）",
    "tui.placement.moved": "{role} → {side}（{reason}）",
    # -- Configuration editor: chrome ------------------------------------
    "cfg.title": "Kotonoha 設定",
    "cfg.subtitle": "變更會寫入 {path}",
    "cfg.key.save": "儲存",
    "cfg.key.reload": "重新載入",
    "cfg.key.menu": "類別",
    "cfg.key.quit": "結束",
    "cfg.categories": "類別",
    "cfg.saved": "已將 {count} 項設定儲存至 {path}",
    "cfg.no_changes": "沒有需要儲存的變更",
    "cfg.invalid": "已拒絕，設定將不合法：{error}",
    "cfg.reloaded": "已從磁碟重新載入",
    "cfg.restart_required": "需重新啟動口譯系統才會生效",
    "cfg.effective": "生效值",
    "cfg.modified": "已變更",
    # -- Configuration editor: sections ----------------------------------
    "cfg.section.interface": "介面",
    "cfg.section.session": "工作階段",
    "cfg.section.audio": "音訊裝置",
    "cfg.section.frontend": "音訊前處理",
    "cfg.section.models": "模型",
    "cfg.section.remote": "外部伺服器",
    # -- Configuration editor: field descriptions ------------------------
    "cfg.f.ui.language": "介面語言。auto 依系統地區設定。",
    "cfg.f.session.mode": "push_to_talk 需按鍵，auto 由 VAD 切分語句。",
    "cfg.f.session.routing": "pair 在兩種語言間往返，fixed 一律譯往同一語言。",
    "cfg.f.audio.input_device": "麥克風編號或名稱。留空使用系統預設。",
    "cfg.f.audio.output_device": "喇叭編號或名稱。留空使用系統預設。",
    "cfg.f.frontend.denoise.enabled": "DeepFilterNet3 雜訊抑制。",
    "cfg.f.frontend.vad.backend": "裝置上使用 silero_onnx，energy 僅供開發機替代。",
    "cfg.f.frontend.vad.threshold": "判定語音開始的機率，0 到 1。",
    "cfg.f.frontend.vad.preroll_ms": (
        "語音開始前保留的音訊。低於 200ms 會截掉第一個音節。"
    ),
    "cfg.f.frontend.vad.silence_ms": "判定語句結束所需的靜音長度。",
    "cfg.f.asr.backend": "transformers 已確認，vllm 待 Spike 1 結果。",
    "cfg.f.asr.n_best": "每句回傳的假設數量，修正流程會全部使用。",
    "cfg.f.asr_verify.mode": "conditional 依信心值判斷，always 每輪都執行。",
    "cfg.f.llm.profile": "moe 為 30B 混合專家模型，dense 為 14B。",
    "cfg.f.tts.backend": "qwen3 取決於 Spike 2 結果，melo 為備援。",
    "cfg.f.perf_mode": (
        "onboard 全部在本機執行。hybrid 只移出 LLM，語音留在裝置內。"
        "remote 則移出所有模型。"
    ),
    "cfg.f.remote.enabled": "設為 false 時，無論 perf_mode 為何，所有角色都在本機執行。",
    "cfg.f.remote.services.llm": "外部伺服器的翻譯服務 URL。",
    "cfg.f.remote.services.asr": "外部伺服器的 ASR 服務 URL。",
    "cfg.f.remote.services.asr_verify": "外部伺服器的驗證服務 URL。",
    "cfg.f.remote.services.tts": "外部伺服器的語音合成服務 URL。",
    "cfg.f.remote.audio_encoding": "s16le 的傳輸位元組數為 f32le 的一半。",
    "cfg.f.remote.failover_after": "角色退回本機前容許的連續傳輸失敗次數。",
}
