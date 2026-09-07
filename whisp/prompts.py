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

import pathlib

_RU_RULES = """Правила отбора:
- РЕШЕНИЕ — это выбор, который участники сделали В ХОДЕ разговора («договорились», «принято», «делаем так», «отказались от»). Описание того, как работает продукт, регламент или система, решением НЕ является.
- ЗАДАЧА — это конкретное действие, которое человек взял на себя ПОСЛЕ разговора, с намерением его выполнить. Шаг демонстрации, возможность продукта или обязанность абстрактной роли задачей НЕ является.
- Если это демонстрация, презентация или обучение и участники ни о чём не договаривались — в разделах «Решения» и «Задачи» пиши ровно «Нет». Это нормальный и ожидаемый ответ, не пытайся заполнить разделы любой ценой.
- УЧАСТНИК — человек, который представился или был назван по имени в разговоре. Указывай роль и компанию, если они прозвучали, и метку спикера, если соответствие однозначно следует из текста. Не выдумывай имён и не угадывай их по косвенным признакам; если никого опознать не удалось, пиши «Не определены». Каждый участник — отдельным пунктом. Не выписывай метку спикера как отдельного участника: метка либо привязана к имени, либо не упоминается вовсе. Не превращай названия продуктов и компаний в людей.
- Разделы не пересекаются: один и тот же пункт не должен стоять и в «Решениях», и в «Задачах». Решение — о чём договорились; задача — что кто-то будет делать дальше.
- Не выводи ничего, чего нет в расшифровке."""

_EN_RULES = """Selection rules:
- A DECISION is a choice the participants made DURING the conversation ("we agreed", "we'll go with", "we rejected"). A description of how a product, process or system works is NOT a decision.
- An ACTION ITEM is a concrete action a person committed to doing AFTER the conversation. A step in a demo, a product capability, or the duty of an abstract role is NOT an action item.
- If this is a demo, presentation or training session and the participants agreed on nothing, write exactly "None" in both Decisions and Action items. That is a normal, expected answer - do not pad the sections.
- A PARTICIPANT is a person who introduced themselves or was addressed by name. Give their role and company if stated, and their speaker label where the transcript makes the match unambiguous. Never invent names and never guess them from indirect cues; if nobody can be identified, write "Not identified". One participant per bullet. Never list a speaker label as a participant of its own: a label is either attached to a name or left out. Never turn a product or company name into a person.
- The sections do not overlap: the same item must not appear under both Decisions and Action items. A decision is what was agreed; an action item is what someone will do next.
- Do not state anything that is not in the transcript."""

_RU_FORMAT = """## Участники
(участники разговора: имя, роль или компания, если названы, и метка спикера, если определима; если никого опознать не удалось — «Не определены»)

## Основные темы
(список ключевых обсуждённых тем, не больше 15 пунктов — объединяй близкое)

## Решения
(список решений, принятых участниками; если их не было — «Нет»)

## Задачи
(список задач с ответственным, если он определим; если их не было — «Нет»)

Каждый пункт — отдельной строкой, начинающейся с «- ». Не объединяй пункты
в один абзац через запятую. Единственное исключение — слово «Нет», которое
пишется без дефиса."""

_EN_FORMAT = """## Participants
(who took part: name, role or company if stated, and speaker label where identifiable; write "Not identified" if nobody can be identified)

## Key topics
(key topics discussed, at most 15 bullets - merge closely related ones)

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
- кто участвует: кто представился или был назван по имени, с ролью и компанией, если прозвучали, и с какой меткой спикера это связано;
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

Собери участников из всех фрагментов в один список и используй его, чтобы заменить метки спикеров на имена в остальных разделах. Объединяй повторы, сохраняй ответственных. Без вступлений и текста вне структуры. Простой текст, без markdown-ограждений."""

EN_SINGLE = f"""You summarize transcripts of business calls and meetings.
Lines like [SPEAKER_00]: text mark speakers (they may be absent).

{_EN_RULES}

Respond ONLY with the summary in English, in exactly this format:

{_EN_FORMAT}

No preamble, disclaimers, or text outside this structure. Plain text, no markdown code fences."""

EN_MAP = """You are processing a FRAGMENT of a business meeting transcript.
This is not the whole meeting - do not draw conclusions about it as a whole and do not write a final summary.

Write dense factual notes about this fragment in English:
- who is taking part: anyone who introduced themselves or was addressed by name, with role and company if stated, and which speaker label they map to;
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

Assemble the participants from every fragment into one roster and use it to replace speaker labels with names in the other sections. Merge duplicates, preserve owners. No preamble or text outside the structure. Plain text, no markdown code fences."""

TEMPLATES = {
    ("ru", "single"): RU_SINGLE,
    ("ru", "map"): RU_MAP,
    ("ru", "reduce"): RU_REDUCE,
    ("en", "single"): EN_SINGLE,
    ("en", "map"): EN_MAP,
    ("en", "reduce"): EN_REDUCE,
}


KINDS = ("single", "map", "reduce")

# The headings a finished summary must contain, in order. Used to detect a
# generation that stopped early: hitting max_tokens truncates the last
# section, and a summary silently missing its action items is worse than no
# summary at all.
SECTIONS = {
    "ru": ("## Участники", "## Основные темы", "## Решения", "## Задачи"),
    "en": ("## Participants", "## Key topics", "## Decisions", "## Action items"),
}


def sections(language: str | None) -> tuple[str, ...]:
    return SECTIONS["ru" if (language or "").lower().startswith("ru") else "en"]


def _override(directory, kind: str) -> str | None:
    """Read <directory>/<kind>.txt, or None when the user did not supply it.

    A directory of plain files rather than one file with a mini-format: there
    are three templates per run, partial overrides have to keep our defaults
    for the rest, and this needs no parser to explain or to get wrong.
    """
    directory = pathlib.Path(directory)
    if not directory.exists():
        raise ValueError(f"WHISP_SUMMARY_PROMPT: {directory} does not exist")
    if not directory.is_dir():
        raise ValueError(
            f"WHISP_SUMMARY_PROMPT: {directory} is a file; it must be a directory "
            f"holding any of {', '.join(k + '.txt' for k in KINDS)}. "
            "Kinds you leave out keep the built-in prompt."
        )

    path = directory / f"{kind}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"WHISP_SUMMARY_PROMPT: {path} is empty")
    return text


def select(language: str | None, kind: str, override_dir=None) -> str:
    """Template for a language code and phase.

    Russian is the only non-English template, so anything that is not Russian
    falls back to English rather than guessing.

    An override directory replaces individual kinds; whatever it does not
    supply keeps the built-in prompt. Overrides are language-agnostic -- a
    user writing their own prompt has already chosen the language.
    """
    if kind not in KINDS:
        raise KeyError(kind)

    if override_dir is not None:
        custom = _override(override_dir, kind)
        if custom is not None:
            return custom

    lang = "ru" if (language or "").lower().startswith("ru") else "en"
    return TEMPLATES[(lang, kind)]
