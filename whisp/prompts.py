"""Summary prompt templates.

Lifted out of whisp.sh, where the prompt was a single hardcoded Russian
string. Three kinds per language:

  single -- the whole transcript fits one context; emit the final sections.
  map    -- one fragment of a long transcript; emit dense attributed notes,
            deliberately NOT the final sections.
  reduce -- merge the map notes into the final sections.

The map/reduce split matters: a per-chunk summary written in the final format
invents "decisions" out of partial context and discards the speaker
attribution that the reduce pass needs to assign action-item owners.

The selection rules below are not decoration. Measured on
2026-07-30_ts_documet_agree_show.txt (a product demo, no commitments made):
without them the model filled Decisions with four items and Action items with
three, restating how the demoed software behaves, where the reference summary
correctly said "Нет" for both.
"""

_RU_RULES = """Правила отбора:
- РЕШЕНИЕ — это выбор, который участники сделали В ХОДЕ разговора («договорились», «принято», «делаем так», «отказались от»). Описание того, как работает продукт, регламент или система, решением НЕ является.
- ЗАДАЧА — это конкретное действие, которое человек взял на себя ПОСЛЕ разговора, с намерением его выполнить. Шаг демонстрации, возможность продукта или обязанность абстрактной роли задачей НЕ является.
- Если это демонстрация, презентация или обучение и участники ни о чём не договаривались — в разделах «Решения» и «Задачи» пиши ровно «Нет». Это нормальный и ожидаемый ответ, не пытайся заполнить разделы любой ценой.
- Разделы не пересекаются: один и тот же пункт не должен стоять и в «Решениях», и в «Задачах». Решение — о чём договорились; задача — что кто-то будет делать дальше.
- Не выводи ничего, чего нет в расшифровке."""

_EN_RULES = """Selection rules:
- A DECISION is a choice the participants made DURING the conversation ("we agreed", "we'll go with", "we rejected"). A description of how a product, process or system works is NOT a decision.
- An ACTION ITEM is a concrete action a person committed to doing AFTER the conversation. A step in a demo, a product capability, or the duty of an abstract role is NOT an action item.
- If this is a demo, presentation or training session and the participants agreed on nothing, write exactly "None" in both Decisions and Action items. That is a normal, expected answer - do not pad the sections.
- The sections do not overlap: the same item must not appear under both Decisions and Action items. A decision is what was agreed; an action item is what someone will do next.
- Do not state anything that is not in the transcript."""

_RU_FORMAT = """## Основные темы
(список ключевых обсуждённых тем)

## Решения
(список решений, принятых участниками; если их не было — «Нет»)

## Задачи
(список задач с ответственным, если он определим; если их не было — «Нет»)

Каждый пункт — отдельной строкой, начинающейся с «- ». Не объединяй пункты
в один абзац через запятую. Единственное исключение — слово «Нет», которое
пишется без дефиса."""

_EN_FORMAT = """## Key topics
(bullet list of key topics discussed)

## Decisions
(decisions the participants made; "None" if there were none)

## Action items
(action items with an owner where identifiable; "None" if there were none)

Put every item on its own line starting with "- ". Never merge items into a
single comma-separated paragraph. The one exception is the word "None", which
is written without a dash."""

RU_SINGLE = f"""Ты составляешь резюме расшифровки деловой встречи или звонка на русском языке.
Строки вида [SPEAKER_00]: текст обозначают говорящих (могут отсутствовать).

{_RU_RULES}

Ответь ТОЛЬКО резюме на русском языке, ровно в этом формате:

{_RU_FORMAT}

Без вступлений, дисклеймеров и любого текста вне этой структуры. Простой текст, без markdown-ограждений."""

RU_MAP = """Ты обрабатываешь ФРАГМЕНТ расшифровки деловой встречи на русском языке.
Это не вся встреча — не делай выводов о встрече целиком и не пиши итогового резюме.

Выпиши плотные фактические заметки по фрагменту на русском языке:
- обсуждаемые темы;
- договорённости, к которым участники пришли в этом фрагменте (если их нет — так и напиши);
- поручения, которые кто-то взял на себя, обязательно с указанием КТО;
- значимые имена, названия, числа, даты и сроки.

Не превращай описание работы продукта в «решение» или «поручение». Сохраняй атрибуцию говорящих.
Сжато, списком, без заголовков верхнего уровня, без вступлений."""

RU_REDUCE = f"""Ниже — последовательные заметки по фрагментам одной деловой встречи.
Сведи их в одно итоговое резюме на русском языке.

{_RU_RULES}

Формат — ровно такой:

{_RU_FORMAT}

Объединяй повторы, сохраняй ответственных. Без вступлений и текста вне структуры. Простой текст, без markdown-ограждений."""

EN_SINGLE = f"""You summarize transcripts of business calls and meetings.
Lines like [SPEAKER_00]: text mark speakers (they may be absent).

{_EN_RULES}

Respond ONLY with the summary in English, in exactly this format:

{_EN_FORMAT}

No preamble, disclaimers, or text outside this structure. Plain text, no markdown code fences."""

EN_MAP = """You are processing a FRAGMENT of a business meeting transcript.
This is not the whole meeting - do not draw conclusions about it as a whole and do not write a final summary.

Write dense factual notes about this fragment in English:
- topics discussed;
- agreements the participants reached in this fragment (say so if there are none);
- action items someone committed to, always naming WHO;
- significant names, numbers, dates and deadlines.

Do not turn a description of how a product works into a decision or a commitment. Preserve speaker attribution.
Terse bullets, no top-level headings, no preamble."""

EN_REDUCE = f"""Below are sequential notes on fragments of one business meeting.
Merge them into a single final summary in English.

{_EN_RULES}

Use exactly this format:

{_EN_FORMAT}

Merge duplicates, preserve owners. No preamble or text outside the structure. Plain text, no markdown code fences."""

TEMPLATES = {
    ("ru", "single"): RU_SINGLE,
    ("ru", "map"): RU_MAP,
    ("ru", "reduce"): RU_REDUCE,
    ("en", "single"): EN_SINGLE,
    ("en", "map"): EN_MAP,
    ("en", "reduce"): EN_REDUCE,
}


def select(language: str | None, kind: str) -> str:
    """Template for a language code and phase.

    Russian is the only non-English template, so anything that is not Russian
    falls back to English rather than guessing.
    """
    lang = "ru" if (language or "").lower().startswith("ru") else "en"
    return TEMPLATES[(lang, kind)]
