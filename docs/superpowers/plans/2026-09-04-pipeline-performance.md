# Ускорение пайплайна whisp — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перенести диаризацию и alignment на MPS и считать диаризацию параллельно с ASR, сократив обработку часовой записи с ~3 часов до ~15 минут без изменения результата.

**Architecture:** `whisp.sh` перестаёт вызывать `whisperx` CLI и вызывает собственный драйвер — пакет `whisp`, использующий whisperx как библиотеку. Драйвер декодирует аудио один раз, запускает диаризацию на MPS в отдельном потоке параллельно с ASR на CPU, затем делает alignment на MPS, сливает результаты и пишет `.txt` тем же writer'ом, что и CLI.

**Tech Stack:** Python 3.11+, whisperx 3.8.6, faster-whisper/CTranslate2 (CPU), pyannote.audio 4.0.7 + torch 2.8 MPS, pytest, bash.

## Global Constraints

- Спека: `docs/superpowers/specs/2026-09-04-pipeline-performance-design.md`.
- **Побайтовая совместимость вывода.** MD5 транскрипта эталонной записи обязан остаться `c1d06fca712248ef5a48d22f7304a807`. Любой шаг, меняющий параметры ASR, нарушает этот критерий и запрещён.
- **Драйвер обязан воспроизвести параметры whisperx CLI дословно.** Значения зафиксированы в Task 3 и получены из `.venv/lib/python3.13/site-packages/whisperx/__main__.py` и `transcribe.py`. Особые случаи, на которых легко ошибиться:
  - `--threads 0` (наш случай) означает **`faster_whisper_threads = 4`**, а не «все ядра»; `torch.set_num_threads()` при этом **не вызывается**.
  - `temperature_increment_on_fallback` по умолчанию `0.2`, поэтому `temperatures` — кортеж `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`, а не `[0]`.
  - `condition_on_previous_text` жёстко `False` независимо от флага CLI.
- Контракт `whisp.sh` не меняется: те же аргументы, `.env`, проверка `HF_TOKEN`, путь вывода рядом с исходником, звук в конце.
- Рядом с исходником по-прежнему появляются только `<имя>.txt` и `<имя>-summary.txt`. Никаких новых файлов в папке пользователя.
- Транскрипт важнее всего остального: падение диаризации или саммари не должно ронять прогон.
- Целевая платформа — macOS на Apple Silicon. На машинах без MPS всё обязано работать с откатом на CPU.
- Язык кода и комментариев — английский, как в существующем `whisp.sh`. Пользовательские тексты README — русский и английский, оба файла.

---

## Файловая структура

| Файл | Ответственность |
|---|---|
| `whisp/__init__.py` | Пустой маркер пакета |
| `whisp/devices.py` | Выбор устройства для torch-этапов, число потоков CTranslate2. Чистые функции |
| `whisp/timing.py` | Замер этапов и форматирование строк лога |
| `whisp/stages.py` | Обёртки над whisperx: загрузка моделей, ASR, alignment, диаризация. Держит дословные параметры CLI |
| `whisp/pipeline.py` | Оркестрация: декод, параллельная диаризация, слияние, запись. Этапы инъектируются |
| `whisp/__main__.py` | Разбор аргументов, чтение переменных окружения, вызов `pipeline.run` |
| `whisp-lib.sh` | `run_with_timeout` — вынесен из `whisp.sh`, чтобы поддавался тестам |
| `whisp.sh` | Меняется вызов whisperx на драйвер, добавляется `--no-diarize`, чинится отчёт об ошибке саммари |
| `pyproject.toml` | `packages`, dev-зависимость pytest, конфиг pytest |
| `tests/` | Юнит-тесты и опциональный интеграционный тест |
| `README.md`, `README.en.md` | Новые переменные окружения, флаг, обновлённый раздел про ограничения |

---

### Task 1: Скелет тестов и выбор устройства

**Files:**
- Create: `whisp/__init__.py`
- Create: `whisp/devices.py`
- Create: `tests/__init__.py`
- Create: `tests/test_devices.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: ничего
- Produces: `whisp.devices.select_device(requested: str | None) -> str`, `whisp.devices.asr_threads(requested: int) -> int`

- [ ] **Step 1: Добавить pytest и пакет в pyproject.toml**

Заменить блок `[tool.setuptools]` и добавить два новых блока:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools]
packages = ["whisp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: needs cached models and real audio; opt-in"]
```

Удалить строки комментария выше `[tool.setuptools]`, которые утверждают, что проект не поставляет импортируемый пакет, и `py-modules = []`.

- [ ] **Step 2: Установить dev-зависимости**

Run:
```bash
./.venv/bin/pip install "pytest>=8"
./.venv/bin/pip install -e . --no-deps
```
Expected: `./.venv/bin/pytest --version` печатает версию.

Два шага, а не `pip install -e ".[dev]"`, намеренно: `--no-deps` не даёт pip
переразрешать зависимости и случайно тронуть уже установленные torch и
whisperx, переустановка которых занимает минуты и рискует сменить версии.

- [ ] **Step 3: Написать падающий тест**

Создать `whisp/__init__.py` пустым и `tests/__init__.py` пустым, затем `tests/test_devices.py`:

```python
from unittest import mock

from whisp import devices


def test_select_device_honours_explicit_request():
    assert devices.select_device("cpu") == "cpu"
    assert devices.select_device("mps") == "mps"


def test_select_device_auto_prefers_mps_when_available():
    with mock.patch.object(devices.torch.backends.mps, "is_available", return_value=True):
        assert devices.select_device("auto") == "mps"
        assert devices.select_device(None) == "mps"


def test_select_device_auto_falls_back_to_cpu():
    with mock.patch.object(devices.torch.backends.mps, "is_available", return_value=False):
        assert devices.select_device("auto") == "cpu"


def test_asr_threads_replicates_whisperx_cli_default():
    # whisperx CLI: `--threads 0` means 4 CTranslate2 threads, not "all cores".
    assert devices.asr_threads(0) == 4
    assert devices.asr_threads(6) == 6
```

- [ ] **Step 4: Запустить тест и убедиться, что он падает**

Run: `./.venv/bin/pytest tests/test_devices.py -v`
Expected: FAIL — `ModuleNotFoundError` или `AttributeError: module 'whisp.devices' has no attribute 'select_device'`

- [ ] **Step 5: Написать минимальную реализацию**

`whisp/devices.py`:

```python
"""Device selection for the torch-backed stages.

CTranslate2 (the ASR engine) has no Metal backend, so ASR always runs on CPU.
Diarization and alignment are plain torch and do run on MPS, which is why the
pipeline picks devices per stage instead of using one global device.
"""

import torch


def select_device(requested: str | None = None) -> str:
    """Device for diarization and alignment.

    "auto" (the default) picks MPS when the machine has it and falls back to
    CPU otherwise, so the pipeline keeps working on Intel Macs and in CI.
    """
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def asr_threads(requested: int = 0) -> int:
    """CTranslate2 thread count, replicating the whisperx CLI.

    whisperx treats `--threads 0` as "leave the default", which is 4 -- not
    "use every core". Reproducing that exactly is what keeps transcripts
    byte-identical to the pre-optimisation output.
    """
    return requested if requested > 0 else 4
```

- [ ] **Step 6: Запустить тесты и убедиться, что они проходят**

Run: `./.venv/bin/pytest tests/test_devices.py -v`
Expected: 4 passed

- [ ] **Step 7: Коммит**

```bash
git add pyproject.toml whisp/__init__.py whisp/devices.py tests/__init__.py tests/test_devices.py
git commit -m "Add whisp package skeleton and per-stage device selection

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Замер этапов

**Files:**
- Create: `whisp/timing.py`
- Create: `tests/test_timing.py`

**Interfaces:**
- Consumes: ничего
- Produces: `whisp.timing.StageLog(audio_seconds: float, stream)` с методами `record(name: str, seconds: float) -> str`, `stage(name: str)` (контекстный менеджер), `total(seconds: float) -> str`, и свойством `stages: dict[str, float]`

- [ ] **Step 1: Написать падающий тест**

`tests/test_timing.py`:

```python
import io

from whisp import timing


def test_record_formats_line_with_rtf():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=1200.0, stream=out)
    line = log.record("diarize", 86.1)
    assert line == "whisp: diarize 86.1s RTF 0.072"
    assert out.getvalue() == line + "\n"


def test_record_without_audio_duration_omits_rtf():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=0.0, stream=out)
    assert log.record("decode", 8.0) == "whisp: decode 8.0s"


def test_stages_are_accumulated():
    log = timing.StageLog(audio_seconds=100.0, stream=io.StringIO())
    log.record("asr", 21.4)
    log.record("align", 3.1)
    assert log.stages == {"asr": 21.4, "align": 3.1}


def test_stage_context_manager_records_elapsed():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=10.0, stream=out)
    with log.stage("asr"):
        pass
    assert "asr" in log.stages
    assert log.stages["asr"] >= 0.0
    assert out.getvalue().startswith("whisp: asr ")


def test_total_line():
    out = io.StringIO()
    log = timing.StageLog(audio_seconds=4840.0, stream=out)
    assert log.total(1215.0) == "whisp: TOTAL 1215.0s RTF 0.251"
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `./.venv/bin/pytest tests/test_timing.py -v`
Expected: FAIL — `AttributeError: module 'whisp.timing' has no attribute 'StageLog'`

- [ ] **Step 3: Написать минимальную реализацию**

`whisp/timing.py`:

```python
"""Per-stage timing.

The automation log is the only place stage durations are recorded, so the
lines are flushed as they happen: whisperx block-buffers its own output when
redirected to a file, which made the log lag reality by minutes.
"""

import contextlib
import sys
import time


class StageLog:
    def __init__(self, audio_seconds: float, stream=None):
        self.audio_seconds = audio_seconds
        self.stream = stream if stream is not None else sys.stderr
        self.stages: dict[str, float] = {}

    def _format(self, label: str, seconds: float) -> str:
        line = f"whisp: {label} {seconds:.1f}s"
        if self.audio_seconds > 0:
            line += f" RTF {seconds / self.audio_seconds:.3f}"
        return line

    def _emit(self, line: str) -> str:
        print(line, file=self.stream, flush=True)
        return line

    def record(self, name: str, seconds: float) -> str:
        self.stages[name] = seconds
        return self._emit(self._format(name, seconds))

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - started)

    def total(self, seconds: float) -> str:
        return self._emit(self._format("TOTAL", seconds))
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `./.venv/bin/pytest tests/test_timing.py -v`
Expected: 5 passed

- [ ] **Step 5: Коммит**

```bash
git add whisp/timing.py tests/test_timing.py
git commit -m "Add per-stage timing with flushed log lines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Обёртки этапов над whisperx

**Files:**
- Create: `whisp/stages.py`
- Create: `tests/test_stages.py`

**Interfaces:**
- Consumes: ничего
- Produces:
  - `whisp.stages.ASR_OPTIONS: dict`, `whisp.stages.VAD_OPTIONS: dict`
  - `load_asr(model_name: str, compute_type: str, language: str, threads: int, hf_token: str)` -> whisperx pipeline
  - `transcribe(asr_model, audio, batch_size: int, language: str) -> dict`
  - `align_segments(segments: list[dict], audio, language: str, device: str) -> dict`
  - `diarize(audio, device: str, hf_token: str, batch_size: int, model_name: str)` -> `pandas.DataFrame`

Эти значения — не «разумные дефолты», а дословная копия того, что собирает `whisperx/transcribe.py` при вызове из `whisp.sh`. Менять их нельзя: от этого зависит побайтовое совпадение транскрипта.

- [ ] **Step 1: Написать падающий тест**

Тест сверяет наши константы не с литералом, а с argparse-дефолтами
установленного whisperx: сравнение с литералом ловило бы только правки в
нашем же файле, а настоящий риск — апгрейд whisperx, меняющий его дефолт.
Тогда побайтовая совместимость сломалась бы молча. Модели тест не грузит.

`tests/test_stages.py`:

```python
"""The ASR options must match what the installed whisperx CLI would build.

Comparing against a hardcoded literal would only catch edits to our own
constants. The real risk is a whisperx upgrade changing one of its argparse
defaults: the transcript would silently stop being byte-identical and
nothing would say so. So the expected values are read out of the installed
whisperx instead of being written down here.
"""

import ast
import pathlib

import numpy as np
import whisperx

from whisp import stages


def whisperx_cli_defaults() -> dict:
    """argparse defaults declared by the installed whisperx CLI.

    Parsed statically: whisperx builds its parser inside cli(), which also
    runs the whole pipeline, so it cannot be imported and inspected.
    """
    source = pathlib.Path(whisperx.__file__).with_name("__main__.py").read_text()
    defaults: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        flag = next(
            (
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ),
            None,
        )
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag[2:]] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass  # e.g. --device, whose default is a torch call
    return defaults


def test_extraction_finds_the_arguments_we_depend_on():
    """Guards the guard: a whisperx refactor could make the parse silently
    return nothing, and every assertion below would pass on empty data."""
    found = whisperx_cli_defaults()
    for name in (
        "beam_size",
        "patience",
        "length_penalty",
        "temperature",
        "temperature_increment_on_fallback",
        "compression_ratio_threshold",
        "logprob_threshold",
        "no_speech_threshold",
        "suppress_tokens",
        "chunk_size",
        "vad_onset",
        "vad_offset",
        "diarize_model",
    ):
        assert name in found, f"whisperx no longer declares --{name}"


def test_asr_options_match_the_installed_whisperx_defaults():
    found = whisperx_cli_defaults()
    assert stages.ASR_OPTIONS["beam_size"] == found["beam_size"]
    assert stages.ASR_OPTIONS["patience"] == found["patience"]
    assert stages.ASR_OPTIONS["length_penalty"] == found["length_penalty"]
    assert (
        stages.ASR_OPTIONS["compression_ratio_threshold"]
        == found["compression_ratio_threshold"]
    )
    assert stages.ASR_OPTIONS["log_prob_threshold"] == found["logprob_threshold"]
    assert stages.ASR_OPTIONS["no_speech_threshold"] == found["no_speech_threshold"]
    assert stages.ASR_OPTIONS["initial_prompt"] == found["initial_prompt"]
    assert stages.ASR_OPTIONS["hotwords"] == found["hotwords"]
    assert stages.ASR_OPTIONS["suppress_tokens"] == [
        int(x) for x in found["suppress_tokens"].split(",")
    ]


def test_temperatures_expand_the_fallback_increment():
    """whisperx turns --temperature 0 plus --temperature_increment_on_fallback
    0.2 into a six-value tuple, not [0]. Getting this wrong changes decoding
    and therefore the transcript."""
    found = whisperx_cli_defaults()
    expected = tuple(
        np.arange(
            found["temperature"],
            1.0 + 1e-6,
            found["temperature_increment_on_fallback"],
        )
    )
    assert stages.ASR_OPTIONS["temperatures"] == expected
    assert len(expected) == 6


def test_condition_on_previous_text_is_forced_off():
    """whisperx hardcodes False in transcribe.py regardless of the CLI flag."""
    assert stages.ASR_OPTIONS["condition_on_previous_text"] is False


def test_suppress_numerals_defaults_to_false():
    """A store_true flag, so it declares no argparse default at all."""
    assert "suppress_numerals" not in whisperx_cli_defaults()
    assert stages.ASR_OPTIONS["suppress_numerals"] is False


def test_vad_options_match_the_installed_whisperx_defaults():
    found = whisperx_cli_defaults()
    assert stages.VAD_OPTIONS == {
        "chunk_size": found["chunk_size"],
        "vad_onset": found["vad_onset"],
        "vad_offset": found["vad_offset"],
    }


def test_default_diarization_model_matches_cli():
    assert stages.DIARIZE_MODEL == whisperx_cli_defaults()["diarize_model"]
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `./.venv/bin/pytest tests/test_stages.py -v`
Expected: FAIL — `AttributeError: module 'whisp.stages' has no attribute 'ASR_OPTIONS'`

- [ ] **Step 3: Написать минимальную реализацию**

`whisp/stages.py`:

```python
"""Thin wrappers around whisperx, one per pipeline stage.

Every option below is copied verbatim from what whisperx's own CLI assembles
for the arguments whisp.sh passes (see whisperx/transcribe.py and
whisperx/__main__.py). They are reproduced here rather than re-derived
because the transcript must stay byte-identical to the pre-optimisation
output; the acceptance test is an md5 comparison.
"""

import numpy as np
import whisperx
from whisperx.audio import SAMPLE_RATE, load_audio

DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

# whisperx CLI: --temperature 0 with --temperature_increment_on_fallback 0.2
ASR_OPTIONS = {
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "temperatures": tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    # whisperx hardcodes this to False regardless of the CLI flag.
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "hotwords": None,
    "suppress_tokens": [-1],
    "suppress_numerals": False,
}

VAD_OPTIONS = {
    "chunk_size": 30,
    "vad_onset": 0.500,
    "vad_offset": 0.363,
}


def decode(audio_path: str):
    """Decode once; every stage reuses this array."""
    return load_audio(audio_path)


def duration_seconds(audio) -> float:
    return len(audio) / SAMPLE_RATE


def load_asr(model_name: str, compute_type: str, language: str, threads: int, hf_token: str):
    return whisperx.load_model(
        model_name,
        device="cpu",  # CTranslate2 has no Metal backend
        device_index=0,
        compute_type=compute_type,
        language=language,
        asr_options=dict(ASR_OPTIONS),
        vad_method="pyannote",
        vad_options=dict(VAD_OPTIONS),
        task="transcribe",
        local_files_only=False,
        threads=threads,
        use_auth_token=hf_token,
    )


def transcribe(asr_model, audio, batch_size: int, language: str) -> dict:
    return asr_model.transcribe(audio, batch_size=batch_size, language=language)


def align_segments(segments, audio, language: str, device: str) -> dict:
    model, metadata = whisperx.load_align_model(language_code=language, device=device)
    try:
        return whisperx.align(
            segments,
            model,
            metadata,
            audio,
            device,
            interpolate_method="nearest",
            return_char_alignments=False,
        )
    finally:
        del model


def diarize(audio, device: str, hf_token: str, batch_size: int, model_name: str = DIARIZE_MODEL):
    pipeline = whisperx.DiarizationPipeline(
        model_name=model_name, token=hf_token, device=device
    )
    # Measured on M1 Pro / 16 GB: batch 64 gives RTF 0.067 against 0.082 at the
    # shipped default of 32. Batch 128 regresses to 0.115 -- it runs out of
    # unified memory -- so 64 is the ceiling, not a starting point.
    pipeline.model.segmentation_batch_size = batch_size
    pipeline.model.embedding_batch_size = batch_size
    try:
        return pipeline(audio)
    finally:
        del pipeline
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `./.venv/bin/pytest tests/test_stages.py -v`
Expected: 7 passed

- [ ] **Step 5: Посмотреть, что именно извлеклось из whisperx**

Run:
```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0, 'tests')
from test_stages import whisperx_cli_defaults
d = whisperx_cli_defaults()
for k in ('beam_size','temperature','temperature_increment_on_fallback','vad_offset','threads'):
    print(k, '=', repr(d.get(k)))
"
```
Expected: `beam_size = 5`, `temperature = 0`, `temperature_increment_on_fallback = 0.2`, `vad_offset = 0.363`, `threads = 0`.

- [ ] **Step 6: Коммит**

```bash
git add whisp/stages.py tests/test_stages.py
git commit -m "Add stage wrappers reproducing whisperx CLI parameters verbatim

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Оркестрация с параллельной диаризацией

**Files:**
- Create: `whisp/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `whisp.timing.StageLog`
- Produces: `whisp.pipeline.Stages` (dataclass с полями `decode`, `duration`, `load_asr`, `transcribe`, `align`, `diarize`, `assign`, `write`), `whisp.pipeline.run(...) -> pathlib.Path`

Инъекция этапов существует ради тестируемости: логика параллелизма, слияния и деградации проверяется на подделках за доли секунды, без загрузки моделей.

- [ ] **Step 1: Написать падающий тест**

`tests/test_pipeline.py`:

```python
import io
import threading

import pytest

from whisp import pipeline
from whisp.timing import StageLog


def make_stages(**overrides):
    """Fake stages: fast, deterministic, no models."""
    calls = []

    def record(name, result=None):
        def fn(*args, **kwargs):
            calls.append(name)
            return result
        return fn

    stages = pipeline.Stages(
        decode=record("decode", "AUDIO"),
        duration=lambda audio: 100.0,
        load_asr=record("load_asr", "ASR_MODEL"),
        transcribe=record("transcribe", {"segments": [{"text": "hi"}]}),
        align=record("align", {"segments": [{"text": "hi", "words": []}]}),
        diarize=record("diarize", "DIARIZE_DF"),
        assign=record("assign", {"segments": [{"text": "hi", "speaker": "SPEAKER_00"}]}),
        write=record("write"),
    )
    for key, value in overrides.items():
        setattr(stages, key, value)
    stages.calls = calls
    return stages


def run(stages, **kwargs):
    params = dict(
        audio_path="/tmp/x.wav",
        output_dir="/tmp",
        language="ru",
        model_name="turbo",
        compute_type="int8",
        device="mps",
        hf_token="tok",
        batch_size=8,
        diarize_batch_size=64,
        asr_thread_count=4,
        diarize_enabled=True,
        parallel=True,
        stages=stages,
        log=StageLog(audio_seconds=100.0, stream=io.StringIO()),
    )
    params.update(kwargs)
    return pipeline.run(**params)


def test_diarization_runs_concurrently_with_asr():
    """The whole point of the change: diarization must not wait for ASR."""
    diarize_started = threading.Event()
    asr_may_finish = threading.Event()

    def slow_diarize(*args, **kwargs):
        diarize_started.set()
        return "DIARIZE_DF"

    def transcribe_waiting_for_diarize(*args, **kwargs):
        # Fails the test rather than hanging forever if the stages are serial.
        assert diarize_started.wait(timeout=5), "diarization did not start during ASR"
        asr_may_finish.set()
        return {"segments": [{"text": "hi"}]}

    stages = make_stages(diarize=slow_diarize, transcribe=transcribe_waiting_for_diarize)
    run(stages)
    assert asr_may_finish.is_set()


def test_serial_mode_still_produces_a_transcript():
    stages = make_stages()
    run(stages, parallel=False)
    assert "diarize" in stages.calls
    assert "assign" in stages.calls
    assert "write" in stages.calls


def test_diarization_failure_does_not_lose_the_transcript():
    def failing_diarize(*args, **kwargs):
        raise RuntimeError("MPS out of memory")

    stages = make_stages(diarize=failing_diarize)
    run(stages)
    assert "write" in stages.calls
    assert "assign" not in stages.calls


def test_no_diarize_skips_diarization_and_alignment():
    stages = make_stages()
    run(stages, diarize_enabled=False)
    assert "diarize" not in stages.calls
    assert "align" not in stages.calls
    assert "write" in stages.calls


def test_returns_transcript_path():
    stages = make_stages()
    result = run(stages, audio_path="/tmp/meeting.webm", output_dir="/out")
    assert str(result) == "/out/meeting.txt"
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `./.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL — `AttributeError: module 'whisp.pipeline' has no attribute 'Stages'`

- [ ] **Step 3: Написать минимальную реализацию**

`whisp/pipeline.py`:

```python
"""Pipeline orchestration.

Diarization does not depend on the transcript, and it runs on the GPU while
ASR runs on the CPU, so the two are started together and the wall time is
max(ASR, diarization) instead of their sum. On an 80-minute recording that
takes diarization off the critical path entirely: it finishes around minute 6
while ASR runs to minute 17.

Stages are injected so the concurrency, merge and degradation logic can be
tested without loading any model.
"""

import dataclasses
import pathlib
import threading
import time
from typing import Any, Callable

from whisp import stages as default_stages
from whisp.timing import StageLog


@dataclasses.dataclass
class Stages:
    decode: Callable[..., Any]
    duration: Callable[..., float]
    load_asr: Callable[..., Any]
    transcribe: Callable[..., dict]
    align: Callable[..., dict]
    diarize: Callable[..., Any]
    assign: Callable[..., dict]
    write: Callable[..., Any]


def default() -> Stages:
    import whisperx
    from whisperx.utils import get_writer

    def write(result, audio_path, output_dir):
        writer = get_writer("txt", output_dir)
        writer(
            result,
            audio_path,
            {"highlight_words": False, "max_line_count": None, "max_line_width": None},
        )

    return Stages(
        decode=default_stages.decode,
        duration=default_stages.duration_seconds,
        load_asr=default_stages.load_asr,
        transcribe=default_stages.transcribe,
        align=default_stages.align_segments,
        diarize=default_stages.diarize,
        assign=whisperx.assign_word_speakers,
        write=write,
    )


def run(
    audio_path: str,
    output_dir: str,
    language: str,
    model_name: str,
    compute_type: str,
    device: str,
    hf_token: str,
    batch_size: int,
    diarize_batch_size: int,
    asr_thread_count: int,
    diarize_enabled: bool,
    parallel: bool,
    stages: Stages | None = None,
    log: StageLog | None = None,
) -> pathlib.Path:
    stages = stages or default()
    started = time.monotonic()

    audio = stages.decode(audio_path)
    audio_seconds = stages.duration(audio)
    log = log or StageLog(audio_seconds=audio_seconds)
    log.record("decode", time.monotonic() - started)

    diarization: dict[str, Any] = {}

    def diarize_worker():
        try:
            worker_started = time.monotonic()
            diarization["df"] = stages.diarize(
                audio, device, hf_token, diarize_batch_size
            )
            log.record("diarize", time.monotonic() - worker_started)
        except Exception as exc:  # transcript matters more than speaker labels
            diarization["error"] = exc

    worker = None
    if diarize_enabled:
        if parallel:
            worker = threading.Thread(target=diarize_worker, name="whisp-diarize")
            worker.start()
        else:
            diarize_worker()

    with log.stage("asr"):
        asr_model = stages.load_asr(
            model_name, compute_type, language, asr_thread_count, hf_token
        )
        result = stages.transcribe(asr_model, audio, batch_size, language)
    del asr_model

    if worker is not None:
        worker.join()

    if diarize_enabled and "df" in diarization:
        with log.stage("align"):
            result = stages.align(result["segments"], audio, language, device)
        result = stages.assign(diarization["df"], result)
    elif diarize_enabled:
        print(
            f"whisp: diarization failed ({diarization.get('error')}); "
            "writing the transcript without speaker labels",
            flush=True,
        )

    stages.write(result, audio_path, output_dir)
    log.total(time.monotonic() - started)

    stem = pathlib.Path(audio_path).stem
    return pathlib.Path(output_dir) / f"{stem}.txt"
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `./.venv/bin/pytest tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Запустить весь набор**

Run: `./.venv/bin/pytest -v`
Expected: 20 passed

- [ ] **Step 6: Коммит**

```bash
git add whisp/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline orchestration with diarization running alongside ASR

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: CLI драйвера

**Files:**
- Create: `whisp/__main__.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `whisp.pipeline.run`, `whisp.devices.select_device`, `whisp.devices.asr_threads`
- Produces: `whisp.__main__.parse_args(argv: list[str], env: dict) -> argparse.Namespace`, `whisp.__main__.main(argv: list[str] | None) -> int`

- [ ] **Step 1: Написать падающий тест**

`tests/test_cli.py`:

```python
import pytest

from whisp import __main__ as cli


def test_defaults_come_from_environment():
    args = cli.parse_args(
        ["audio.webm"],
        env={
            "WHISP_LANG": "en",
            "WHISP_MODEL": "large-v3",
            "WHISP_COMPUTE_TYPE": "float32",
            "WHISP_BATCH_SIZE": "16",
            "WHISP_DIARIZE_BATCH_SIZE": "32",
            "WHISP_PARALLEL": "0",
            "WHISP_DEVICE": "cpu",
            "WHISP_ASR_THREADS": "6",
        },
    )
    assert args.language == "en"
    assert args.model == "large-v3"
    assert args.compute_type == "float32"
    assert args.batch_size == 16
    assert args.diarize_batch_size == 32
    assert args.parallel is False
    assert args.device == "cpu"
    assert args.threads == 6


def test_defaults_without_environment_match_current_behaviour():
    args = cli.parse_args(["audio.webm"], env={})
    assert args.language == "ru"
    assert args.model == "turbo"
    assert args.compute_type == "int8"
    assert args.batch_size == 8
    assert args.diarize_batch_size == 64
    assert args.parallel is True
    assert args.device == "auto"
    assert args.threads == 0
    assert args.diarize is True


def test_no_diarize_flag():
    args = cli.parse_args(["audio.webm", "--no-diarize"], env={})
    assert args.diarize is False


def test_output_dir_defaults_to_the_source_directory():
    args = cli.parse_args(["/recordings/meeting.webm"], env={})
    assert args.output_dir == "/recordings"
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `./.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — `AttributeError: module 'whisp.__main__' has no attribute 'parse_args'`

- [ ] **Step 3: Написать минимальную реализацию**

`whisp/__main__.py`:

```python
"""CLI for the whisp driver.

Replaces the `whisperx` CLI call in whisp.sh. whisperx's own CLI takes a
single --device for every stage, which forces diarization onto the CPU
because CTranslate2 cannot use Metal; this driver picks a device per stage
and overlaps the two that do not depend on each other.
"""

import argparse
import os
import pathlib
import sys

from whisp import devices, pipeline


def parse_args(argv: list[str], env: dict | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env

    parser = argparse.ArgumentParser(prog="whisp", description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--language", default=env.get("WHISP_LANG", "ru"))
    parser.add_argument("--model", default=env.get("WHISP_MODEL", "turbo"))
    parser.add_argument("--compute-type", default=env.get("WHISP_COMPUTE_TYPE", "int8"))
    parser.add_argument("--device", default=env.get("WHISP_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=int(env.get("WHISP_BATCH_SIZE", "8")))
    parser.add_argument(
        "--diarize-batch-size",
        type=int,
        default=int(env.get("WHISP_DIARIZE_BATCH_SIZE", "64")),
    )
    parser.add_argument("--threads", type=int, default=int(env.get("WHISP_ASR_THREADS", "0")))
    parser.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.set_defaults(diarize=True, parallel=env.get("WHISP_PARALLEL", "1") != "0")

    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(pathlib.Path(args.audio).resolve().parent)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    hf_token = os.environ.get("HF_TOKEN", "")
    if args.diarize and not hf_token:
        print("whisp: HF_TOKEN is not set; diarization needs it", file=sys.stderr)
        return 1

    device = devices.select_device(args.device)
    if device == "cpu" and args.device in (None, "auto"):
        print("whisp: MPS unavailable, running diarization and alignment on CPU", flush=True)

    pipeline.run(
        audio_path=args.audio,
        output_dir=args.output_dir,
        language=args.language,
        model_name=args.model,
        compute_type=args.compute_type,
        device=device,
        hf_token=hf_token,
        batch_size=args.batch_size,
        diarize_batch_size=args.diarize_batch_size,
        asr_thread_count=devices.asr_threads(args.threads),
        diarize_enabled=args.diarize,
        parallel=args.parallel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `./.venv/bin/pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Коммит**

```bash
git add whisp/__main__.py tests/test_cli.py
git commit -m "Add whisp driver CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Отчёт об ошибке саммари

Сейчас `whisp.sh` печатает «failed or timed out after 300s» и при падении за секунду. За четыре последних прогона саммари не сгенерировалось ни разу из-за протухшего OAuth, и по логу это выглядело как таймаут. Чтобы это стало тестируемым, `run_with_timeout` выносится в отдельный файл.

**Files:**
- Create: `whisp-lib.sh`
- Create: `tests/test_whisp_lib.sh`
- Modify: `whisp.sh`

**Interfaces:**
- Consumes: ничего
- Produces: `run_with_timeout <timeout_secs> <out_file> <in_file> <flag_file> <cmd...>` — при срабатывании сторожа пишет `timeout` в `flag_file`, иначе оставляет его пустым

- [ ] **Step 1: Написать падающий тест**

`tests/test_whisp_lib.sh`:

```bash
#!/bin/bash
# Run: bash tests/test_whisp_lib.sh
set -uo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/whisp-lib.sh"
# shellcheck source=/dev/null
source "$LIB"

fail=0
check() {
    if [ "$2" = "$3" ]; then
        echo "ok   - $1"
    else
        echo "FAIL - $1 (expected '$3', got '$2')"
        fail=1
    fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
: >"$tmp/in"

# A command that fails immediately must not be reported as a timeout.
run_with_timeout 30 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'echo boom >&2; exit 3'
check "fast failure returns the command status" "$?" "3"
check "fast failure leaves the flag empty" "$(cat "$tmp/flag")" ""
check "fast failure captures output" "$(cat "$tmp/out")" "boom"

# A command that outlives the timeout must be reported as a timeout.
run_with_timeout 1 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'sleep 30'
check "timeout is flagged" "$(cat "$tmp/flag")" "timeout"

# A command that succeeds must return 0 and leave the flag empty.
run_with_timeout 30 "$tmp/out" "$tmp/in" "$tmp/flag" sh -c 'echo fine'
check "success returns 0" "$?" "0"
check "success leaves the flag empty" "$(cat "$tmp/flag")" ""

exit "$fail"
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `bash tests/test_whisp_lib.sh`
Expected: FAIL — `whisp-lib.sh: No such file or directory`

- [ ] **Step 3: Создать whisp-lib.sh**

```bash
#!/bin/bash
# Shared helpers for whisp.sh. Kept in its own file so tests can source it.

# Run "$5..." with stdin from $3, output to $2, killing it after $1 seconds and
# writing "timeout" into $4 if the watchdog is what stopped it. Hand-rolled
# because no timeout/gtimeout binary exists on a stock Mac. Stdin must be
# passed in and redirected right here (not inherited from the caller) -- bash
# drops an inherited stdin redirect to /dev/null for a backgrounded (&)
# command unless the redirect is written on that exact command line.
#
# The flag file exists so callers can tell a timeout from a fast failure. They
# used to be indistinguishable, and an instant auth error was reported to the
# log as "timed out after 300s" for weeks.
run_with_timeout() {
    local timeout_secs="$1" out_file="$2" in_file="$3" flag_file="$4"
    shift 4
    : >"$flag_file"
    "$@" <"$in_file" >"$out_file" 2>&1 &
    local cmd_pid=$!
    ( sleep "$timeout_secs"
      if kill -TERM "$cmd_pid" 2>/dev/null; then printf 'timeout' >"$flag_file"; fi ) &
    local watcher_pid=$!
    local status=0
    wait "$cmd_pid" || status=$?
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
    return "$status"
}
```

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `bash tests/test_whisp_lib.sh`
Expected: 6 строк `ok`, код возврата 0

- [ ] **Step 5: Подключить библиотеку в whisp.sh и развести сообщения**

В `whisp.sh` удалить определение `run_with_timeout` (вместе с его комментарием) и вместо него сразу после строки с `SCRIPT_DIR` добавить:

```bash
# shellcheck source=whisp-lib.sh
source "$SCRIPT_DIR/whisp-lib.sh"
```

Рядом с `MODEL`/`LANG_CODE`/`COMPUTE_TYPE` добавить:

```bash
SUMMARY_TIMEOUT="${WHISP_SUMMARY_TIMEOUT:-900}"
```

Блок саммари заменить на:

```bash
    SUMMARY_TMP="$(mktemp)"
    SUMMARY_FLAG="$(mktemp)"

    if run_with_timeout "$SUMMARY_TIMEOUT" "$SUMMARY_TMP" "$TRANSCRIPT_FILE" "$SUMMARY_FLAG" \
            claude -p --system-prompt "$SUMMARY_SYSTEM_PROMPT" --model sonnet; then
        mv "$SUMMARY_TMP" "$SUMMARY_FILE"
        echo "Summary saved to $SUMMARY_FILE"
    else
        if [ -s "$SUMMARY_FLAG" ]; then
            echo "Warning: summary generation timed out after ${SUMMARY_TIMEOUT}s; the transcript itself is unaffected and is saved at $TRANSCRIPT_FILE." >&2
        else
            echo "Warning: summary generation failed; the transcript itself is unaffected and is saved at $TRANSCRIPT_FILE. The command's own output follows." >&2
        fi
        cat "$SUMMARY_TMP" >&2 || true
        rm -f "$SUMMARY_TMP"
    fi
    rm -f "$SUMMARY_FLAG"
```

- [ ] **Step 6: Проверить, что различение работает на живом скрипте**

Run: `bash -n whisp.sh && echo "syntax ok"`
Expected: `syntax ok`

Run: `WHISP_SUMMARY_TIMEOUT=900 ./whisp.sh promo-main.mp4 -sum 2>&1 | tail -5`

(`promo-main.mp4` — короткая запись на 64 секунды в рабочей папке; если её нет,
подойдёт любой короткий аудио- или видеофайл.)
Expected: строка про саммари содержит `failed`, а не `timed out`, поскольку `claude` падает мгновенно на неавторизованной сессии. Если вы уже выполнили `claude login`, ожидается `Summary saved to ...`.

- [ ] **Step 7: Коммит**

```bash
git add whisp-lib.sh whisp.sh tests/test_whisp_lib.sh
git commit -m "Distinguish summary timeout from summary failure

The watchdog now records whether it was what killed the command, so an
instant auth failure is no longer reported as a 300s timeout. Timeout is
configurable via WHISP_SUMMARY_TIMEOUT and defaults to 900s.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Переключить whisp.sh на драйвер

**Files:**
- Modify: `whisp.sh`

**Interfaces:**
- Consumes: `whisp.__main__.main` через `python -m whisp`
- Produces: неизменный пользовательский контракт `whisp <файл> [-sum] [--no-diarize]`

- [ ] **Step 1: Добавить разбор `--no-diarize`**

В цикле разбора аргументов в `whisp.sh`, рядом с веткой `-sum)`, добавить:

```bash
        --no-diarize)
            DIARIZE=0
            ;;
```

и в блоке инициализации переменных рядом с `SUMMARIZE=0` добавить `DIARIZE=1`.

В строку `Usage:` в обоих местах, где она печатается, подставить:

```
Usage: whisp <filename> [-sum] [--no-diarize]
```

- [ ] **Step 2: Заменить вызов whisperx на драйвер**

Блок

```bash
"$SCRIPT_DIR/.venv/bin/whisperx" "$FILE" \
    --model "$MODEL" \
    --language "$LANG_CODE" \
    --compute_type "$COMPUTE_TYPE" \
    --diarize \
    --hf_token "$HF_TOKEN" \
    --output_dir "$OUTPUT_DIR" \
    --output_format txt
```

заменить на

```bash
DIARIZE_ARGS=()
if [ "$DIARIZE" -eq 0 ]; then
    DIARIZE_ARGS+=(--no-diarize)
fi

# The whisperx CLI takes one --device for every stage, which pins diarization
# to the CPU because CTranslate2 has no Metal backend. This driver picks a
# device per stage and runs diarization alongside ASR instead.
PYTHONPATH="$SCRIPT_DIR" "$SCRIPT_DIR/.venv/bin/python" -m whisp "$FILE" \
    --model "$MODEL" \
    --language "$LANG_CODE" \
    --compute-type "$COMPUTE_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    "${DIARIZE_ARGS[@]}"
```

`HF_TOKEN` уже экспортирован через `set -a` при чтении `.env`, поэтому драйвер получает его из окружения.

- [ ] **Step 3: Проверить синтаксис и разбор аргументов**

Run: `bash -n whisp.sh && echo "syntax ok"`
Expected: `syntax ok`

Run: `./whisp.sh 2>&1 | head -3`
Expected: сообщение об ошибке содержит `Usage: whisp <filename> [-sum] [--no-diarize]`

- [ ] **Step 4: Прогнать на коротком файле**

Run: `time ./whisp.sh promo-main.mp4`
Expected: завершается успешно, создаёт `promo-main.txt`, в stderr видны строки `whisp: decode ...`, `whisp: diarize ...`, `whisp: asr ...`, `whisp: align ...`, `whisp: TOTAL ...`

- [ ] **Step 5: Коммит**

```bash
git add whisp.sh
git commit -m "Call the whisp driver instead of the whisperx CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Документация

**Files:**
- Modify: `README.md`
- Modify: `README.en.md`

- [ ] **Step 1: Обновить таблицу переменных окружения в README.md**

Добавить в таблицу «Настройка» строки:

| `WHISP_DEVICE` | `auto` | Устройство для диаризации и выравнивания: `auto`, `mps`, `cpu` |
| `WHISP_PARALLEL` | `1` | `0` — считать диаризацию последовательно, а не параллельно с распознаванием |
| `WHISP_SUMMARY_TIMEOUT` | `900` | Таймаут генерации саммари, секунды |
| `WHISP_BATCH_SIZE` | `8` | Батч распознавания |
| `WHISP_DIARIZE_BATCH_SIZE` | `64` | Батч диаризации |
| `WHISP_ASR_THREADS` | `0` | Потоки CTranslate2; `0` — как у whisperx по умолчанию |

- [ ] **Step 2: Обновить раздел «Использование» в README.md**

Заменить строку `whisp <файл> [-sum]` на `whisp <файл> [-sum] [--no-diarize]` и добавить абзац:

```markdown
- Флаг `--no-diarize` отключает разметку по спикерам. Имеет смысл только для
  заведомо одноголосых записей — лекций, промо-роликов: экономит около 12%
  времени, но лишает транскрипт единственного источника информации о том, кто
  что сказал.
```

- [ ] **Step 3: Переписать раздел «Ограничения» в README.md**

Заменить пункт про «Только CPU на Apple Silicon» на:

```markdown
- **Распознавание идёт на CPU.** Движок CTranslate2 не поддерживает Metal.
  Диаризация и выравнивание при этом выполняются на GPU через MPS и считаются
  параллельно с распознаванием, поэтому на Apple Silicon узкое место —
  именно распознавание.
```

- [ ] **Step 4: Обновить README.en.md теми же правками**

Добавить в таблицу «Configuration» строки:

| `WHISP_DEVICE` | `auto` | Device for diarization and alignment: `auto`, `mps`, `cpu` |
| `WHISP_PARALLEL` | `1` | `0` runs diarization serially instead of alongside transcription |
| `WHISP_SUMMARY_TIMEOUT` | `900` | Summary generation timeout, seconds |
| `WHISP_BATCH_SIZE` | `8` | Transcription batch size |
| `WHISP_DIARIZE_BATCH_SIZE` | `64` | Diarization batch size |
| `WHISP_ASR_THREADS` | `0` | CTranslate2 threads; `0` keeps the whisperx default |

Заменить `whisp <file> [-sum]` на `whisp <file> [-sum] [--no-diarize]` и добавить:

```markdown
- `--no-diarize` turns off speaker labelling. It only makes sense for
  recordings you know have a single voice — lectures, promo clips: it saves
  around 12% of the runtime, but strips the transcript of its only source of
  who-said-what.
```

Заменить пункт про CPU-only в «Limitations» на:

```markdown
- **Transcription runs on the CPU.** CTranslate2 has no Metal backend.
  Diarization and alignment do run on the GPU through MPS, and they run
  alongside transcription, so on Apple Silicon transcription is the
  bottleneck.
```

- [ ] **Step 5: Проверить, что таблицы не разъехались**

Run: `grep -c "^| \`WHISP" README.md README.en.md`
Expected: по 9 строк в каждом файле (3 существующих плюс 6 новых)

- [ ] **Step 6: Коммит**

```bash
git add README.md README.en.md
git commit -m "Document the new tuning knobs and --no-diarize

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Приёмка на эталонной записи

Этот шаг доказывает и корректность, и достижение цели. Без него план не выполнен.

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: весь пайплайн
- Produces: ничего, только проверки

- [ ] **Step 1: Написать интеграционный тест**

`tests/test_integration.py`:

```python
"""Opt-in end-to-end check.

Needs cached models and a real recording, so it is marked `integration` and
skipped unless WHISP_TEST_AUDIO points at a file. Run it with:

    WHISP_TEST_AUDIO=/path/to/recording.webm ./.venv/bin/pytest -m integration -v
"""

import hashlib
import os
import pathlib

import pytest

pytestmark = pytest.mark.integration

AUDIO = os.environ.get("WHISP_TEST_AUDIO")
EXPECTED_MD5 = os.environ.get("WHISP_TEST_MD5")


@pytest.mark.skipif(not AUDIO, reason="set WHISP_TEST_AUDIO to run")
def test_pipeline_reproduces_the_reference_transcript(tmp_path):
    from whisp import devices, pipeline

    result = pipeline.run(
        audio_path=AUDIO,
        output_dir=str(tmp_path),
        language=os.environ.get("WHISP_LANG", "ru"),
        model_name=os.environ.get("WHISP_MODEL", "turbo"),
        compute_type=os.environ.get("WHISP_COMPUTE_TYPE", "int8"),
        device=devices.select_device("auto"),
        hf_token=os.environ["HF_TOKEN"],
        batch_size=8,
        diarize_batch_size=64,
        asr_thread_count=devices.asr_threads(0),
        diarize_enabled=True,
        parallel=True,
    )

    produced = pathlib.Path(result)
    assert produced.exists()
    digest = hashlib.md5(produced.read_bytes()).hexdigest()
    if EXPECTED_MD5:
        assert digest == EXPECTED_MD5, (
            f"transcript changed: {digest} != {EXPECTED_MD5}. "
            "The optimisation must not alter the output."
        )
```

- [ ] **Step 2: Убедиться, что тест пропускается без переменной**

Run: `./.venv/bin/pytest -m integration -v`
Expected: 1 skipped

- [ ] **Step 3: Прогнать приёмку на эталонной записи**

Запись — тот же файл длительностью 1:20:40, на котором снят базовый замер. Перед запуском закрыть тяжёлые приложения: базовый прогон 20 августа шёл втрое медленнее чистого замера именно из-за конкуренции за ресурсы.

Эталонная запись — единственный файл в рабочей папке длительностью 1:20:40
(4840 с); опознать его можно по `ffprobe`, а проверить, что он тот самый, по
md5 уже лежащего рядом транскрипта:

```bash
md5 -q "<файл>.txt"   # ожидается c1d06fca712248ef5a48d22f7304a807
```

Сам файл в репозиторий не коммитится, поэтому путь здесь не зафиксирован.

Run:
```bash
set -a; source .env; set +a
time WHISP_TEST_AUDIO="<путь, проверенный выше>" \
     WHISP_TEST_MD5=c1d06fca712248ef5a48d22f7304a807 \
     ./.venv/bin/pytest -m integration -v -s
```
Expected: 1 passed. Строки `whisp:` показывают примерно `diarize ~350s RTF ~0.072`, `asr ~1034s RTF ~0.214`, `align ~150s RTF ~0.031`, `TOTAL ~1215s RTF ~0.251`.

- [ ] **Step 4: Сверить с критериями приёмки**

Проверить по выводу предыдущего шага:
- MD5 совпал — тест прошёл, значит вывод не изменился;
- `TOTAL` уложился в 25 минут (1500 с);
- `diarize` меньше `asr`, то есть диаризация действительно ушла с критического пути.

Если `TOTAL` превысил 25 минут, не подгонять числа: зафиксировать фактические строки `whisp:` и разобраться, какой этап выбился, прежде чем продолжать.

- [ ] **Step 5: Прогнать весь набор тестов**

Run: `./.venv/bin/pytest -v && bash tests/test_whisp_lib.sh`
Expected: 24 passed, 1 skipped; шесть строк `ok` от bash-теста

- [ ] **Step 6: Коммит**

```bash
git add tests/test_integration.py
git commit -m "Add opt-in end-to-end acceptance test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## После выполнения

- Прогнать полный сценарий через автоматизацию — `./automation/process-new-file.sh <файл>` — и убедиться, что уведомление приходит, а лог содержит строки `whisp:` с таймингами.
- Вернуть коммит со спекой и планом в `main` и запушить вместе с реализацией, после чего поставить тег `v*`: релиз соберётся сам через `.github/workflows/release.yml`.
- Отложенное, не входит в этот план: показ этапов средствами macOS и перевод ASR на mlx-whisper. Оба описаны в разделе «Отложено» спеки.
