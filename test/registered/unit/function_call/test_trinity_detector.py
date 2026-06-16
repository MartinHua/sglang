"""Unit tests for TrinityDetector — no server, no model loading.

TrinityDetector extends Qwen25Detector and strips ``<think>``/``</think>``
tags before delegating to the Qwen 2.5 tool-call parser. These tests focus on
the think-tag handling that is unique to Trinity (one-shot and streaming),
complementing the Qwen 2.5 format coverage that lives elsewhere.

Qwen 2.5 wire format (inherited):
    <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.trinity_detector import TrinityDetector
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
    """Build one Qwen 2.5 <tool_call> block."""
    return (
        '<tool_call>\n'
        + json.dumps({"name": name, "arguments": args})
        + '\n</tool_call>'
    )


class TestTrinityDetector(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = TrinityDetector()

    # ==================== _strip_think_tags ====================

    def test_strip_think_tags_removes_both_tags(self):
        self.assertEqual(
            self.detector._strip_think_tags("<think>reasoning</think>answer"),
            "reasoninganswer",
        )

    def test_strip_think_tags_no_tags_unchanged(self):
        self.assertEqual(
            self.detector._strip_think_tags("plain text, no tags"),
            "plain text, no tags",
        )

    def test_strip_think_tags_multiple_blocks(self):
        self.assertEqual(
            self.detector._strip_think_tags("<think>a</think>X<think>b</think>Y"),
            "aXbY",
        )

    def test_strip_think_tags_unbalanced_closing_tag(self):
        # Implementation is a plain str.replace, so a lone closing tag is also removed.
        self.assertEqual(
            self.detector._strip_think_tags("foo</think>bar"), "foobar"
        )

    # ==================== has_tool_call ====================

    def test_has_tool_call_after_think_block(self):
        text = "<think>let me check</think>" + _tool_call(
            "get_weather", {"city": "Beijing"}
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_inside_think_block(self):
        # The case Trinity exists for: the tool call is emitted *inside* <think>.
        text = "<think>" + _tool_call("get_weather", {"city": "Beijing"}) + "</think>"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_reasoning_only_is_false(self):
        self.assertFalse(
            self.detector.has_tool_call("<think>just thinking</think>done")
        )

    # ==================== detect_and_parse ====================

    def test_parse_tool_call_after_think(self):
        text = "<think>checking weather</think>" + _tool_call(
            "get_weather", {"city": "Tokyo"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Tokyo")
        # Think markup must not leak into the surfaced normal text.
        self.assertNotIn("<think>", result.normal_text)
        self.assertNotIn("</think>", result.normal_text)

    def test_parse_tool_call_inside_think(self):
        text = "<think>" + _tool_call("search", {"query": "ramen"}) + "</think>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(json.loads(result.calls[0].parameters)["query"], "ramen")

    def test_parse_multiple_tool_calls(self):
        text = (
            "<think>plan</think>"
            + _tool_call("get_weather", {"city": "Beijing"})
            + "\n"
            + _tool_call("search", {"query": "food"})
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "search"])

    def test_parse_no_tool_call_strips_think(self):
        text = "<think>internal reasoning</think> The weather is nice."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertNotIn("<think>", result.normal_text)
        self.assertIn("The weather is nice.", result.normal_text)

    # ==================== parse_streaming_increment ====================

    def test_streaming_think_then_tool_call(self):
        detector = TrinityDetector()
        chunks = [
            "<think>let me check</think>",
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}\n</tool_call>',
        ]
        calls, normal_text = [], ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            calls.extend(result.calls)
            normal_text += result.normal_text

        named = [c for c in calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

        params = "".join(c.parameters for c in calls if c.parameters)
        self.assertEqual(json.loads(params)["city"], "Paris")
        self.assertNotIn("<think>", normal_text)

    def test_streaming_reasoning_only_yields_normal_text(self):
        detector = TrinityDetector()
        result = detector.parse_streaming_increment(
            "<think>thinking out loud</think>", self.tools
        )
        self.assertEqual(len(result.calls), 0)
        self.assertNotIn("<think>", result.normal_text)


if __name__ == "__main__":
    unittest.main()
