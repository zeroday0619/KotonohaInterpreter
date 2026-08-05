# Localization

## Scope

Kotonoha uses English gettext message identifiers for operator-facing CLI and TUI text.
Korean, Japanese, and Traditional Chinese for Taiwan are maintained in `.po` catalogs.
Developer logs, structured log fields, identifiers, configuration paths, commands, and
model identifiers remain English.

Repository documentation, comments, and docstrings remain English. They follow the same
terminology as the English interface but are not extracted into gettext catalogs.

## Translation Profile

| Variable | Value |
|---|---|
| Source language | English |
| Target languages | Korean, Japanese, Traditional Chinese for Taiwan |
| Target locales | `ko-KR`, `ja-JP`, `zh-TW` |
| Product | Consecutive offline speech interpreter |
| Domain | Speech processing, model inference, and edge deployment |
| Input type | UI resource (`.po`) |
| Log policy | Do not translate |
| Proper nouns | Preserve established product and model names |

## English Source Style

- Write labels as concise noun phrases and actions as concise verb phrases.
- Write errors as complete sentences that state the cause and available recovery action.
- Use `source` for recognized input and `translation` for translated output.
- Use `interpretation` for the end-to-end product workflow. Do not substitute
  `translation` when speech input or speech output is part of the workflow.
- Use `turn` for one completed source-and-translation exchange.
- Use `onboard` for execution on the Jetson and `remote` for execution on the external
  server.
- Avoid idioms, phrasal abbreviations, ambiguous pronouns, and English-specific wordplay.
- Do not embed terminal layout with repeated spaces unless the alignment is intentional.
  Add an `i18n:` translator comment when leading or trailing whitespace is required.

## Locale Style

| Locale | Required style |
|---|---|
| `ko-KR` | Use concise labels. Use `합니다` style for complete instructions and errors. Avoid translated English word order and ambiguous paired particles such as `을(를)`. |
| `ja-JP` | Use concise labels. Use `です` and `ます` style for complete instructions and errors. Use Japanese punctuation and established technical terminology. |
| `zh-TW` | Use concise neutral Taiwanese technical language. Use Traditional characters and Taiwanese vocabulary. Do not use Mainland Chinese software terminology. |

Preserve the source punctuation contract for fragments that are concatenated at runtime.
Use locale-native punctuation for complete standalone sentences.

## Terminology

| English | Korean | Japanese | Traditional Chinese (Taiwan) |
|---|---|---|---|
| source | 원문 | 原文 | 原文 |
| source language | 원문 언어 | 原言語 | 來源語言 |
| translated output | 번역문 | 訳文 | 譯文 |
| translation process | 번역 | 翻訳 | 翻譯 |
| interpretation | 통역 | 通訳 | 口譯 |
| turn | 턴 | ターン | 輪次 |
| history | 기록 | 履歴 | 紀錄 |
| glossary | 용어집 | 用語集 | 詞彙表 |
| verification ASR | 검증 ASR | 検証 ASR | 驗證 ASR |
| fallback | 대체 | 代替 | 備援 |
| placement | 배치 | 配置 | 配置 |
| latency | 지연 | 遅延 | 延遲 |
| throughput | 처리량 | スループット | 處理量 |
| configuration | 설정 | 設定 | 設定 |
| software | 소프트웨어 | ソフトウェア | 軟體 |
| information | 정보 | 情報 | 資訊 |
| mouse | 마우스 | マウス | 滑鼠 |
| video | 영상 | 動画 | 影片 |

Apply one term consistently within a screen. Preserve enum values when the interface
describes configuration choices.

## Protected Content

Do not translate or rewrite:

- Placeholders such as `{path}`, `{count}`, and `{error}`
- Identifiers and dotted settings such as `perf_mode` and `remote.enabled`
- Commands, options, environment variables, paths, URLs, and MIME types
- Product and component names such as Kotonoha, vLLM, Transformers, and DeepFilterNet3
- Protocol and format names such as JSON, JSONL, YAML, WAV, and PCM
- Model-role abbreviations such as ASR, LLM, TTS, VAD, PTT, and EOU
- Enum values such as `onboard`, `hybrid`, `remote`, `conditional`, and `always`
- Model and backend values such as `vllm_omni`, `silero_onnx`, `s16le`, and `f32le`

Preserve placeholder names, repetition counts, conversions, and format specifications.
Preserve leading whitespace, trailing whitespace, tabs, and explicit newlines exactly.

## Workflow

1. Write or revise the English source string.
2. Add an `i18n:` translator comment when context or whitespace is not evident.
3. Extract and merge the catalogs:

   ```bash
   uv run python scripts/i18n.py extract
   uv run python scripts/i18n.py update
   ```

4. Translate every non-obsolete entry in each `.po` file.
5. Remove `fuzzy` only after semantic and formatting review.
6. Compile and validate the catalogs:

   ```bash
   uv run python scripts/i18n.py compile
   uv run python scripts/i18n.py check
   uv run pytest -q tests/kotonoha/test_i18n.py
   ```

7. Inspect `--help` and the integrated TUI in all four locales.

## Typer Help

Typer constructs help metadata while importing the CLI module. `KOTONOHA_LANG` and the
configured `ui.language` therefore control localized help. The `--lang` option controls
command output after argument parsing and cannot change help metadata for the same
invocation.

`src/kotonoha/_typer_i18n.py` localizes usage prefixes, section headings, required and
default markers, error labels, completion option descriptions, and the help option.
Command groups and commands must use the localized classes from that module. A Typer
upgrade requires CLI regression tests and manual `--help` inspection in all four locales.

## Validation

The catalog checker rejects:

- Missing, obsolete, untranslated, or fuzzy entries
- Placeholder name, count, conversion, or format-specification changes
- Rich, HTML, or XML markup-tag changes
- Leading or trailing whitespace changes
- Tab, newline, or carriage-return count changes
- Changes to protected technical tokens
- Locale-prohibited vocabulary, including Mainland Chinese software terminology in
  `zh-TW`
- Missing compiled catalogs

Semantic review remains mandatory. Automated checks cannot detect mistranslation,
unnatural phrasing, incorrect register, or locale-specific terminology errors.
