"""Unit tests for Qwen25Detector — no server, no model loading.

Qwen25Detector is the base of the Qwen 2.5 / Qwen 3 tool-call family
(subclassed by TrinityDetector). Format:

    <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>

These tests cover one-shot parsing, multi-call parsing, malformed-block
handling, and — most importantly — the Qwen25-specific streaming
``_normal_text_buffer`` logic that strips partial/complete ``</tool_call>``
end tokens leaking into normal text (``parse_streaming_increment`` override).
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
    ]


def _tool_call(name: str, args: dict) -> str:
    return '<tool_call>\n' + json.dumps({"name": name, "arguments": args}) + '\n</tool_call>'


class TestQwen25Detector(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = Qwen25Detector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        self.assertTrue(
            self.detector.has_tool_call(_tool_call("get_weather", {"city": "Beijing"}))
        )

    def test_has_tool_call_plain_text_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is sunny."))

    def test_has_tool_call_requires_newline_after_tag(self):
        # bot_token is "<tool_call>\n"; a bare "<tool_call>" (no newline) is not a match.
        self.assertFalse(self.detector.has_tool_call("<tool_call>{}"))

    # ==================== detect_and_parse ====================

    def test_single_tool_call(self):
        result = self.detector.detect_and_parse(
            _tool_call("get_weather", {"city": "Beijing"}), self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_leading_text_becomes_normal_text(self):
        text = "Sure, let me check. " + _tool_call("get_weather", {"city": "Tokyo"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        # normal_text is the pre-tool text, stripped.
        self.assertEqual(result.normal_text, "Sure, let me check.")

    def test_multiple_tool_calls(self):
        text = (
            _tool_call("get_weather", {"city": "Beijing"})
            + "\n"
            + _tool_call("search", {"query": "food"})
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "search"])
        self.assertEqual(json.loads(result.calls[1].parameters)["query"], "food")

    def test_no_tool_call_returns_normal_text(self):
        result = self.detector.detect_and_parse("Just a plain answer.", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Just a plain answer.")

    def test_malformed_json_block_is_skipped(self):
        # A well-formed <tool_call> wrapper with invalid JSON inside is skipped,
        # not raised (json.loads failure is caught and logged).
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_nested_json_arguments(self):
        result = self.detector.detect_and_parse(
            _tool_call("get_weather", {"city": "Beijing", "opts": {"detailed": True}}),
            self.tools,
        )
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["opts"]["detailed"], True)

    def test_unknown_tool_in_block_dropped_by_default(self):
        text = _tool_call("nonexistent_tool", {"x": 1})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    # ==================== parse_streaming_increment ====================

    def test_streaming_single_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            '\n</tool_call>',
        ]
        calls = []
        for chunk in chunks:
            calls.extend(detector.parse_streaming_increment(chunk, self.tools).calls)

        named = [c for c in calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")
        params = "".join(c.parameters for c in calls if c.parameters)
        self.assertEqual(json.loads(params)["city"], "Paris")

    def test_streaming_normal_text_passthrough(self):
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Hello there", self.tools)
        self.assertEqual(result.normal_text, "Hello there")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        # Normal text first, then a tool call split across chunks (name, then args).
        detector = Qwen25Detector()
        chunks = [
            "Let me check. ",
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Tokyo"}}\n</tool_call>',
        ]
        calls, normal_text = [], ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            calls.extend(result.calls)
            normal_text += result.normal_text

        self.assertIn("Let me check.", normal_text)
        named = [c for c in calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")
        params = "".join(c.parameters for c in calls if c.parameters)
        self.assertEqual(json.loads(params)["city"], "Tokyo")

    def test_streaming_strips_leaked_end_token_from_normal_text(self):
        # Qwen25-specific: a stray complete "</tool_call>" in the normal-text
        # stream is removed by the _normal_text_buffer cleanup.
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("text</tool_call>more", self.tools)
        self.assertNotIn("</tool_call>", result.normal_text)
        self.assertEqual(result.normal_text, "textmore")

    # ==================== structure_info ====================

    def test_structure_info(self):
        info = self.detector.structure_info()("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertEqual(info.trigger, "<tool_call>")
        self.assertEqual(info.end, "}\n</tool_call>")


if __name__ == "__main__":
    unittest.main()
