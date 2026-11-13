"""
Учебный MCP-сервер: минимальная программа, которая отдаёт харнессу два инструмента.

MCP-сервер общается с клиентом через стандартный ввод и вывод сообщениями JSON-RPC,
по одному на строку. Здесь реализовано ровно то, что нужно для инструментов:
initialize, tools/list и tools/call. Настоящие серверы пишут на готовых библиотеках,
но протокол у них тот же.

Подключение: скопируйте mcp.example.json в mcp.json рядом с harness.py.
"""

import json
import sys
from datetime import datetime

TOOLS = [
    {
        "name": "word_count",
        "description": "Посчитать слова, строки и символы в тексте.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Текст для подсчёта"}},
            "required": ["text"],
        },
    },
    {
        "name": "current_time",
        "description": "Сообщить текущие дату и время на машине пользователя.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, arguments: dict) -> dict:
    if name == "word_count":
        text = arguments["text"]
        answer = f"слов: {len(text.split())}, строк: {len(text.splitlines())}, символов: {len(text)}"
    elif name == "current_time":
        answer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return {"content": [{"type": "text", "text": f"нет инструмента {name}"}], "isError": True}
    return {"content": [{"type": "text", "text": answer}]}


def handle(message: dict) -> dict | None:
    method, params = message.get("method"), message.get("params", {})
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo", "version": "1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        return call_tool(params["name"], params.get("arguments", {}))
    return None


for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:  # уведомление: ответа не ждут
        continue
    try:
        result = handle(request)
        if result is None:
            reply = {"error": {"code": -32601, "message": f"неизвестный метод {request.get('method')}"}}
        else:
            reply = {"result": result}
    except Exception as error:
        reply = {"error": {"code": -32603, "message": f"{type(error).__name__}: {error}"}}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}, ensure_ascii=False), flush=True)
