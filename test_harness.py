"""
Автотесты харнесса. Модель подменяется сценарием (ScriptedClient), поэтому тесты
не ходят в сеть, не тратят деньги и дают одинаковый результат при каждом запуске.

Запуск:
    python -m unittest -v
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import harness as h
from harness import ScriptedClient, scripted_call

ALL = set(h.REGISTRY)


class HarnessTest(unittest.TestCase):
    """Каждый тест получает пустую рабочую папку и чистое состояние сеанса."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name).resolve()
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        patches = {
            "WORKSPACE": self.workspace,
            "LOGS_DIR": root / "logs",
            "SESSIONS_DIR": root / "sessions",
            "MEMORY_FILE": root / "memory.md",
            "PERMISSIONS_FILE": root / "permissions.json",
            "SESSION_FILE": root / "sessions" / "test.json",
            "JOURNAL": h.Journal(),
            "METER": h.Meter(),
            "PERMISSIONS": dict(h.DEFAULT_PERMISSIONS),
        }
        for name, value in patches.items():
            patcher = mock.patch.object(h, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(h.JOURNAL.close)
        h.TODO.clear()
        h.ALWAYS.clear()
        h.CANCEL.clear()
        h.pending_images().clear()
        self.addCleanup(self.tmp.cleanup)
        # Харнесс много печатает — в тестах этот вывод прячем.
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def run_script(self, script, tools=h.TOOLS, user="задача"):
        client = ScriptedClient(script)
        context = h.Context([{"role": "system", "content": "тест"}, {"role": "user", "content": user}])
        result = h.run_turn(client, context, tools)
        return result, context, client

    def answer(self, text):
        """Подставляет ответ человека на вопрос о разрешении."""
        return mock.patch("builtins.input", return_value=text)


class LoopTest(HarnessTest):
    def test_loop_runs_tools_and_returns_final_text(self):
        result, context, client = self.run_script([
            ("Пишу файл.", [scripted_call(1, "write_file", path="a.txt", content="привет")]),
            ("", [scripted_call(2, "read_file", path="a.txt")]),
            ("Готово.", []),
        ])
        self.assertEqual(result.strip(), "Готово.")
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "привет")
        self.assertEqual(len(client.requests), 3)  # три обращения к модели: два с инструментами и финал
        roles = [message["role"] for message in context.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])

    def test_tool_result_is_linked_to_its_call(self):
        _, context, _ = self.run_script([("", [scripted_call(7, "list_dir", path=".")]), ("ок", [])])
        call = context.messages[2]["tool_calls"][0]
        self.assertEqual(call["id"], "call_7")
        self.assertEqual(context.messages[3]["tool_call_id"], "call_7")

    def test_streamed_arguments_are_glued_back(self):
        # ScriptedClient режет аргументы пополам — харнесс должен собрать исходный JSON.
        args = {"path": "длинное имя.txt", "content": "строка\nвторая"}
        _, context, _ = self.run_script([("", [scripted_call(1, "write_file", **args)]), ("ок", [])])
        self.assertEqual(json.loads(context.messages[2]["tool_calls"][0]["function"]["arguments"]), args)

    def test_two_calls_in_one_reply(self):
        (self.workspace / "a.txt").write_text("A")
        (self.workspace / "b.txt").write_text("B")
        _, context, _ = self.run_script([
            ("", [scripted_call(1, "read_file", path="a.txt"), scripted_call(2, "read_file", path="b.txt")]),
            ("ок", []),
        ])
        self.assertIn("A", context.messages[3]["content"])
        self.assertIn("B", context.messages[4]["content"])

    def test_step_limit_stops_endless_loop(self):
        endless = [("", [scripted_call(i, "list_dir", path=".")]) for i in range(h.MAX_STEPS + 5)]
        result, _, client = self.run_script(endless)
        self.assertIn("превышен лимит", result)
        self.assertEqual(len(client.requests), h.MAX_STEPS)

    def test_budget_stops_agent(self):
        h.METER.budget, h.METER.cost = 1.0, 1.5
        result, _, client = self.run_script([("не должно дойти", [])])
        self.assertIn("бюджет", result)
        self.assertEqual(client.requests, [])


class ExecutorTest(HarnessTest):
    def test_validation_error_names_the_field(self):
        self.assertIn("content: Field required", h.execute_tool("write_file", '{"path": "a.txt"}', ALL))

    def test_extra_argument_is_rejected(self):
        self.assertIn("Extra inputs", h.execute_tool("list_dir", '{"path": ".", "x": 1}', ALL))

    def test_broken_json_does_not_crash(self):
        self.assertIn("Invalid JSON", h.execute_tool("list_dir", "не json", ALL))

    def test_unknown_or_hidden_tool(self):
        self.assertIn("недоступен", h.execute_tool("nope", "{}", ALL))
        self.assertIn("недоступен", h.execute_tool("write_file", "{}", {"read_file"}))

    def test_sandbox_blocks_escape(self):
        (self.workspace.parent / "secret.txt").write_text("тайна")
        for name, args in [("read_file", {"path": "../secret.txt"}), ("write_file", {"path": "../x", "content": ""}),
                           ("list_dir", {"path": ".."}), ("git_diff", {"path": "../secret.txt"})]:
            self.assertIn("вне рабочей папки", h.execute_tool(name, json.dumps(args), ALL), name)

    def test_browser_refuses_local_files(self):
        self.assertIn("только адреса http", h.execute_tool("browser_open", '{"url": "file:///etc/passwd"}', ALL))

    def test_long_output_is_clipped(self):
        (self.workspace / "big.txt").write_text("x" * (h.OUTPUT_LIMIT * 2))
        output = h.execute_tool("read_file", '{"path": "big.txt"}', ALL)
        self.assertIn("результат обрезан", output)
        self.assertLess(len(output), h.OUTPUT_LIMIT + 200)


class CodeToolsTest(HarnessTest):
    def setUp(self):
        super().setUp()
        (self.workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n")

    def tool(self, name, **args):
        return h.execute_tool(name, json.dumps(args), ALL)

    def test_read_shows_line_numbers_and_range(self):
        output = self.tool("read_file", path="calc.py", start_line=5, end_line=6)
        self.assertEqual(output.splitlines()[0], "    5│def sub(a, b):")
        self.assertNotIn("def add", output)

    def test_edit_requires_unique_fragment(self):
        self.assertIn("встречается 2 раз", self.tool("edit_file", path="calc.py", old_text="return a - b", new_text="x"))
        self.assertIn("не найден", self.tool("edit_file", path="calc.py", old_text="нет такого", new_text="x"))
        self.assertIn("return a - b", (self.workspace / "calc.py").read_text())  # файл не тронут

    def test_edit_replaces_only_the_fragment(self):
        self.tool("edit_file", path="calc.py", old_text="def add(a, b):\n    return a - b", new_text="def add(a, b):\n    return a + b")
        text = (self.workspace / "calc.py").read_text()
        self.assertIn("def add(a, b):\n    return a + b", text)
        self.assertIn("def sub(a, b):\n    return a - b", text)

    def test_search_and_find(self):
        self.assertEqual(self.tool("search", pattern=r"def \w+").splitlines(),
                         ["calc.py:1: def add(a, b):", "calc.py:5: def sub(a, b):"])
        self.assertIn("некорректное регулярное выражение", self.tool("search", pattern="("))
        self.assertEqual(self.tool("find_files", pattern="**/*.py"), "calc.py")
        self.assertEqual(self.tool("find_files", pattern="../*"), "(файлов не найдено)")

    def test_search_skips_service_folders(self):
        (self.workspace / "node_modules").mkdir()
        (self.workspace / "node_modules" / "lib.py").write_text("def hidden(): pass")
        self.assertNotIn("hidden", self.tool("search", pattern="def"))

    def test_run_python(self):
        (self.workspace / "hello.py").write_text("import sys; print('привет', sys.argv[1])")
        with self.answer("y"):
            output = self.tool("run_python", path="hello.py", args=["мир"])
        self.assertIn("код выхода: 0", output)
        self.assertIn("привет мир", output)


class PermissionsTest(HarnessTest):
    def shell(self):
        return h.execute_tool("run_shell", '{"command": "echo hi"}', ALL)

    def test_ask_yes_runs(self):
        with self.answer("y"):
            self.assertIn("hi", self.shell())

    def test_ask_no_refuses_without_running(self):
        with self.answer("n"), mock.patch("subprocess.run") as run:
            self.assertIn("отклонил", self.shell())
        run.assert_not_called()

    def test_always_is_remembered_for_session(self):
        with self.answer("a") as prompt:
            self.shell()
            self.shell()
        self.assertEqual(prompt.call_count, 1)  # второй раз уже не спрашивали

    def test_closed_input_counts_as_no(self):
        with mock.patch("builtins.input", side_effect=EOFError), mock.patch("subprocess.run") as run:
            self.assertIn("отклонил", self.shell())
        run.assert_not_called()

    def test_deny_rule_never_asks(self):
        h.PERMISSIONS["run_shell"] = "deny"
        with self.answer("y") as prompt:
            self.assertIn("ОТКАЗ", self.shell())
        prompt.assert_not_called()

    def test_allow_rule_never_asks(self):
        h.PERMISSIONS["run_shell"] = "allow"
        with self.answer("n") as prompt:
            self.assertIn("hi", self.shell())
        prompt.assert_not_called()

    def test_file_overrides_defaults(self):
        h.PERMISSIONS_FILE.write_text('{"browser_click": "allow", "write_file": "ask"}')
        h.load_permissions()
        self.assertEqual(h.PERMISSIONS["browser_click"], "allow")
        self.assertEqual(h.PERMISSIONS["write_file"], "ask")
        self.assertEqual(h.PERMISSIONS["run_shell"], "ask")  # остальное — как по умолчанию

    def test_bad_file_is_reported(self):
        h.PERMISSIONS_FILE.write_text('{"run_shell": "maybe"}')
        with self.assertRaises(SystemExit):
            h.load_permissions()


class PlanModeTest(HarnessTest):
    def test_plan_tools_are_read_only(self):
        names = {tool["function"]["name"] for tool in h.PLAN_TOOLS}
        self.assertTrue(names.isdisjoint({"write_file", "edit_file", "run_shell", "run_python", "git_restore", "delegate"}))
        self.assertIn("read_file", names)

    def test_write_is_refused_even_if_model_tries(self):
        _, context, _ = self.run_script(
            [("", [scripted_call(1, "write_file", path="a.txt", content="x")]), ("план", [])], tools=h.PLAN_TOOLS
        )
        self.assertIn("недоступен", context.messages[3]["content"])
        self.assertFalse((self.workspace / "a.txt").exists())


class TodoTest(HarnessTest):
    ITEMS = [{"text": "раз", "status": "done"}, {"text": "два", "status": "in_progress"}, {"text": "три", "status": "pending"}]

    def test_todo_is_stored_and_counted(self):
        output = h.execute_tool("todo", json.dumps({"items": self.ITEMS}), ALL)
        self.assertEqual(output, "список обновлён: выполнено 1 из 3")
        self.assertEqual(h.render_todo(), "☑ раз\n◐ два\n☐ три")

    def test_bad_status_is_rejected(self):
        self.assertIn("ОШИБКА в аргументах", h.execute_tool("todo", '{"items": [{"text": "x", "status": "готово"}]}', ALL))

    def test_todo_survives_compaction(self):
        h.TODO[:] = self.ITEMS
        context = h.Context([{"role": "system", "content": "s"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"сообщение {i}"} for i in range(12)
        ])
        self.assertTrue(h.compact(ScriptedClient([]), context))
        self.assertIn("◐ два", context.messages[1]["content"])


class ContextTest(HarnessTest):
    def history(self):
        call = {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
        return [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "первое поручение"},
            {"role": "assistant", "content": "ответ 1"},
            {"role": "user", "content": "второе поручение"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "результат"},
            {"role": "assistant", "content": "ответ 2"},
        ]

    def test_compact_never_separates_result_from_call(self):
        # Граница сжатия попадает на результат инструмента и должна отступить к его вызову.
        context = h.Context(self.history())
        with mock.patch.object(h, "KEEP_RECENT", 2):
            self.assertTrue(h.compact(ScriptedClient([]), context))
        roles = [message["role"] for message in context.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn("Краткое изложение", context.messages[1]["content"])
        self.assertEqual(context.messages[2]["tool_calls"][0]["id"], context.messages[3]["tool_call_id"])

    def test_compact_skips_short_history(self):
        context = h.Context(self.history()[:3])
        self.assertFalse(h.compact(ScriptedClient([]), context))
        self.assertEqual(len(context.messages), 3)

    def test_auto_compaction_triggers_by_token_count(self):
        client = ScriptedClient([("ок", [])])
        context = h.Context(self.history() + [{"role": "user", "content": "ещё"}], prompt_tokens=h.COMPACT_AT_TOKENS + 1)
        with mock.patch.object(h, "KEEP_RECENT", 2):
            h.run_turn(client, context)
        self.assertNotIn("stream", client.requests[0])  # первым ушёл запрос на сжатие
        self.assertTrue(client.requests[1]["stream"])

    def test_repair_closes_unanswered_calls(self):
        calls = [{"id": f"c{i}", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}} for i in (1, 2)]
        messages = [
            {"role": "user", "content": "задача"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "content": "успел"},
        ]
        h.repair_history(messages)
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "c2", "content": "Прервано пользователем."})
        h.repair_history(messages)  # повторный вызов ничего не добавляет
        self.assertEqual(len(messages), 4)

    def test_interrupt_keeps_history_valid(self):
        h.CLIENT = ScriptedClient([("", [scripted_call(1, "list_dir", path=".")])])
        context = h.Context([{"role": "system", "content": "s"}])
        with mock.patch.object(h.ListDir, "run", side_effect=KeyboardInterrupt):
            h.handle(context, "задача")  # Ctrl+C посреди инструмента не должен уронить программу
        roles = [message["role"] for message in context.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertIn("прервал", context.messages[-1]["content"])

    def test_cancel_flag_stops_agent_before_next_request(self):
        h.CANCEL.set()
        result, _, client = self.run_script([("не должно дойти", [])])
        self.assertIn("прервана", result)
        self.assertEqual(client.requests, [])


class VisionTest(HarnessTest):
    def test_screenshot_goes_as_separate_image_message(self):
        messages = [{"role": "user", "content": "задача"}]
        h.pending_images().append("AAAA")
        h.attach_images(messages, "агент")
        parts = messages[-1]["content"]
        self.assertEqual(parts[1], {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}})
        self.assertEqual(h.pending_images(), [])

    def test_only_latest_screenshot_is_kept(self):
        messages = []
        for data in ("OLD", "NEW"):
            h.pending_images().append(data)
            h.attach_images(messages, "агент")
        self.assertIsInstance(messages[0]["content"], str)
        self.assertIn("NEW", messages[1]["content"][1]["image_url"]["url"])

    def test_render_handles_image_messages(self):
        text = h.render([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}])
        self.assertEqual(text, "ПОЛЬЗОВАТЕЛЬ: [изображение]")


class SessionTest(HarnessTest):
    def test_save_and_resume_roundtrip(self):
        _, context, _ = self.run_script([("", [scripted_call(1, "list_dir", path=".")]), ("готово", [])])
        h.TODO[:] = [{"text": "шаг", "status": "pending"}]
        h.save_session(context)
        path, state = h.load_last_session()
        self.assertEqual(path, h.SESSION_FILE)
        self.assertEqual(state["messages"], context.messages)
        self.assertEqual(state["todo"], h.TODO)

    def test_latest_session_wins(self):
        h.SESSIONS_DIR.mkdir()
        for name in ("2026-01-01_10-00-00.json", "2026-03-01_10-00-00.json", "2026-02-01_10-00-00.json"):
            (h.SESSIONS_DIR / name).write_text(json.dumps({"messages": [name], "todo": []}))
        self.assertEqual(h.load_last_session()[1]["messages"], ["2026-03-01_10-00-00.json"])

    def test_no_sessions(self):
        self.assertIsNone(h.load_last_session())


class PromptTest(HarnessTest):
    def test_project_instructions_and_memory_are_in_prompt(self):
        self.assertEqual(h.system_prompt(), h.BASE_PROMPT)
        (self.workspace / h.INSTRUCTIONS_NAME).write_text("Тесты запускай командой make test.")
        h.execute_tool("remember", '{"note": "пользователь  любит\\nкраткость"}', ALL)
        prompt = h.system_prompt()
        self.assertIn("Тесты запускай командой make test.", prompt)
        self.assertIn("пользователь любит краткость", prompt)  # заметка сведена в одну строку


class MeterTest(HarnessTest):
    def usage(self, prompt, completion, ticks=None):
        return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, cost_in_usd_ticks=ticks)

    def test_tokens_and_cost_add_up(self):
        meter = h.Meter()
        meter.add(self.usage(1000, 50, 20_000_000))  # 20 млн тиков = $0.002
        before = meter.totals()
        meter.add(self.usage(2000, 100, 30_000_000))
        self.assertEqual(meter.totals()[:3], (2, 3000, 150))
        self.assertAlmostEqual(meter.cost, 0.005)
        self.assertIn("ход: 2 000 вх. + 100 вых., $0.0030", meter.report(before))

    def test_missing_cost_field_is_fine(self):
        meter = h.Meter()
        meter.add(SimpleNamespace(prompt_tokens=10, completion_tokens=1))  # провайдер без поля стоимости
        meter.add(None)
        self.assertEqual(meter.totals(), (1, 10, 1, 0.0))

    def test_budget(self):
        meter = h.Meter()
        self.assertFalse(meter.over_budget())  # без --budget лимита нет
        meter.budget = 0.01
        meter.add(self.usage(1, 1, 200_000_000))
        self.assertTrue(meter.over_budget())


class SubagentTest(HarnessTest):
    def test_parallel_subagents_have_separate_contexts(self):
        seen = []  # что видел каждый субагент в момент запроса

        def create(**request):
            """Отвечает каждому субагенту по его собственной задаче. Вызывается из разных потоков."""
            messages = request["messages"]
            seen.append([message["content"] for message in messages])
            return ScriptedClient([(f"итог: {messages[1]['content']}", [])])._stream()

        h.CLIENT = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with mock.patch.object(h, "browser", return_value=mock.Mock()):
            output = h.execute_tool("delegate", json.dumps({"tasks": ["альфа", "бета"]}), ALL)
        self.assertIn("### Итог подзадачи 1\nитог: альфа", output)
        self.assertIn("### Итог подзадачи 2\nитог: бета", output)
        # В контексте каждого субагента только его промпт и его задача — чужой задачи там нет.
        self.assertEqual(sorted(seen), [[h.SUBAGENT_PROMPT, "альфа"], [h.SUBAGENT_PROMPT, "бета"]])

    def test_subagent_cannot_delegate_or_remember(self):
        names = {tool["function"]["name"] for tool in h.SUBAGENT_TOOLS}
        self.assertTrue(names.isdisjoint({"delegate", "remember", "todo"}))

    def test_too_many_tasks_rejected(self):
        tasks = [f"задача {i}" for i in range(h.MAX_PARALLEL + 1)]
        self.assertIn("ОШИБКА в аргументах", h.execute_tool("delegate", json.dumps({"tasks": tasks}), ALL))


if __name__ == "__main__":
    unittest.main()
