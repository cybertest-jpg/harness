"""
Харнесс — минимальная обвязка вокруг LLM, превращающая её в агента.

Модель сама по себе умеет только одно: получить текст и вернуть текст.
Всё остальное делает харнесс. Он состоит из четырёх частей:

  1. ИНСТРУМЕНТЫ  — функции, которые модель может «попросить» выполнить.
  2. ИСПОЛНИТЕЛЬ  — код, который реально запускает их (и следит за безопасностью).
  3. КОНТЕКСТ     — список сообщений: вся память агента в пределах сессии.
  4. ЦИКЛ АГЕНТА  — модель → инструменты → результаты → модель → ... → ответ.

Запуск:
    python harness.py            # живая модель x.ai (ключ XAI_API_KEY берётся из файла .env)
    python harness.py --demo     # сценарий без модели и без ключа: видно, как крутится цикл
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

MODEL = "grok-4.7"
BASE_URL = "https://api.x.ai/v1"  # API x.ai совместимо с форматом OpenAI
MAX_STEPS = 20  # предохранитель: агент не может крутиться бесконечно
WORKSPACE = Path(__file__).parent / "workspace"  # песочница: агент работает только здесь

SYSTEM_PROMPT = f"""Ты — агент, работающий в папке {WORKSPACE}.
У тебя есть инструменты для чтения и записи файлов и запуска shell-команд.
Все пути указывай относительно этой папки. Действуй сам: изучи, сделай, проверь результат.
Когда задача выполнена, коротко расскажи, что сделано."""


# ─── 1. ИНСТРУМЕНТЫ ──────────────────────────────────────────────────────────
# Модель не видит наш код. Она видит только это описание: имя, назначение
# и JSON-схему аргументов. По описанию она решает, что и когда вызвать.

def tool(name: str, description: str, **params: str) -> dict:
    """Собирает описание инструмента в формате x.ai / OpenAI. Все аргументы — обязательные строки."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string", "description": hint} for key, hint in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = [
    tool("list_dir", "Показать файлы и папки в каталоге рабочей папки.", path="Относительный путь, '.' — корень"),
    tool("read_file", "Прочитать текстовый файл целиком.", path="Относительный путь к файлу"),
    tool("write_file", "Создать или перезаписать текстовый файл.", path="Относительный путь к файлу", content="Новое содержимое файла"),
    tool("run_shell", "Выполнить shell-команду в рабочей папке. Пользователь подтверждает каждый запуск.", command="Команда"),
]


# ─── 2. ИСПОЛНИТЕЛЬ ──────────────────────────────────────────────────────────
# Здесь «намерение» модели превращается в действие. Это же и граница
# безопасности: модель может попросить что угодно, а что разрешить — решаем мы.

def safe_path(relative: str) -> Path:
    """Не выпускаем агента за пределы песочницы (никаких ../../etc/passwd)."""
    path = (WORKSPACE / relative).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"путь вне рабочей папки: {relative}")
    return path


def list_dir(path: str) -> str:
    entries = sorted(safe_path(path).iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(пусто)"


def read_file(path: str) -> str:
    return safe_path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"записано {len(content)} символов в {path}"


def run_shell(command: str) -> str:
    # Shell — самый опасный инструмент, поэтому человек подтверждает каждый запуск.
    if input(f"    разрешить команду `{command}`? [y/N] ").strip().lower() != "y":
        return "Пользователь отклонил запуск команды."
    done = subprocess.run(command, shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=60)
    return f"код выхода: {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"


HANDLERS = {"list_dir": list_dir, "read_file": read_file, "write_file": write_file, "run_shell": run_shell}


def execute_tool(name: str, raw_args: str) -> str:
    """Аргументы приходят от модели JSON-строкой. Ошибка не роняет харнесс:
    она уходит модели обычным текстом, и модель сама решает, как её обойти."""
    try:
        return HANDLERS[name](**json.loads(raw_args))
    except Exception as error:
        return f"ОШИБКА {type(error).__name__}: {error}"


# ─── 3 и 4. КОНТЕКСТ И ЦИКЛ АГЕНТА ───────────────────────────────────────────

def run_turn(client, messages: list) -> None:
    """Один ход пользователя: крутим цикл, пока модель не перестанет звать инструменты.

    `messages` — это и есть контекст. API ничего не помнит между запросами,
    поэтому каждый раз мы отправляем всю историю целиком и сами её дополняем.
    """
    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(model=MODEL, tools=TOOLS, messages=messages)
        choice = response.choices[0]
        reply = choice.message

        # Ответ модели кладём в историю — вместе с вызовами инструментов.
        entry = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            entry["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in reply.tool_calls
            ]
        messages.append(entry)

        if reply.content:
            print(f"\nагент: {reply.content}")

        # Главный сигнал для харнесса: попросила ли модель инструменты.
        if not reply.tool_calls:
            if choice.finish_reason == "length":
                print("\n[ответ обрезан по лимиту токенов]")
            return  # модель считает задачу выполненной

        # Выполняем каждый вызов и возвращаем результат отдельным сообщением
        # с ролью "tool", сопоставляя его с вызовом по tool_call_id.
        for call in reply.tool_calls:
            print(f"  [шаг {step}] {call.function.name}({call.function.arguments[:200]})")
            output = execute_tool(call.function.name, call.function.arguments)
            print(f"    → {output[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    print(f"\n[остановлено: превышен лимит в {MAX_STEPS} шагов]")


# ─── ДЕМО-РЕЖИМ ──────────────────────────────────────────────────────────────
# Подставная «модель» с заранее записанными ответами в том же формате, что у API.
# Нужна, чтобы увидеть работу цикла без ключа и без затрат.

class ScriptedClient:
    def __init__(self):
        def call(i, name, **args):
            function = SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False))
            return SimpleNamespace(id=f"demo_{i}", function=function)

        self._script = [
            ("Сначала посмотрю, что лежит в папке.", [call(1, "list_dir", path=".")]),
            (None, [call(2, "write_file", path="hello.txt", content="Привет от агента!\n")]),
            (None, [call(3, "read_file", path="hello.txt"), call(4, "read_file", path="../secret.txt")]),
            ("Готово: создал hello.txt и проверил его. Выйти за пределы папки мне не дали.", None),
        ]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        content, tool_calls = self._script.pop(0)
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


# ─── ТОЧКА ВХОДА ─────────────────────────────────────────────────────────────

def load_env_file() -> None:
    """Читает пары КЛЮЧ=значение из .env рядом со скриптом, чтобы ключ не жил в коде."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    system = {"role": "system", "content": SYSTEM_PROMPT}

    if "--demo" in sys.argv:
        print("ДЕМО: ответы модели записаны заранее, инструменты выполняются по-настоящему.")
        run_turn(ScriptedClient(), [system, {"role": "user", "content": "Создай файл hello.txt и проверь его."}])
        return

    import openai  # импорт здесь, чтобы --demo работал без установленного SDK

    load_env_file()
    if not os.environ.get("XAI_API_KEY"):
        sys.exit("Не найден ключ: добавьте строку XAI_API_KEY=... в файл .env рядом со скриптом.")
    client = openai.OpenAI(api_key=os.environ["XAI_API_KEY"], base_url=BASE_URL)

    messages = [system]  # контекст живёт, пока запущена программа
    print(f"Харнесс запущен. Модель: {MODEL}. Рабочая папка: {WORKSPACE}. Выход — Ctrl+C или пустая строка.")

    while True:
        try:
            user_input = input("\nвы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(client, messages)
        except openai.AuthenticationError:
            sys.exit("Ключ API не принят. Проверьте XAI_API_KEY в файле .env.")
        except openai.RateLimitError:
            print("\n[превышен лимит запросов — подождите и повторите]")
        except openai.APIStatusError as error:
            print(f"\n[ошибка API {error.status_code}: {error.message}]")
        except openai.APIConnectionError:
            print("\n[нет связи с API — проверьте сеть]")


if __name__ == "__main__":
    main()
