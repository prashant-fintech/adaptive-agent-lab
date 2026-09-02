"""Tests for adaptive_agent.agent — no Neo4j, no LLM, no embeddings required.

All external I/O (chat, skill_box, graph_search, write_episodes, record_outcome)
is patched so the tests run offline and in isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

import adaptive_agent.agent as agent_mod
from adaptive_agent.agent import FALLBACK_TOPIC, _compose, _episode_from_run, answer
from adaptive_agent.ckg.retrieve import ScoredNode
from adaptive_agent.skills.models import Episode, Skill, Step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _skill(name="test-skill", topic="testing", version=1):
    return Skill(
        name=name,
        topic=topic,
        version=version,
        status="active",
        procedure="1. Do the thing.\n2. Verify.",
    )


def _hint(node_id="src/foo.py", score=0.75):
    return ScoredNode(node_id=node_id, kind="CodeFile", ppr=0.5, cosine=0.6, score=score)


# ---------------------------------------------------------------------------
# _compose — pure function, no mocks needed
# ---------------------------------------------------------------------------

class TestCompose:
    def test_no_skill_no_hints_returns_system_only(self):
        result = _compose(None, [])
        assert result == agent_mod.AGENT_SYSTEM

    def test_with_skill_includes_skill_section(self):
        skill = _skill()
        result = _compose((skill, 0.9), [])
        assert "Retrieved skill: test-skill" in result
        assert "Do the thing" in result
        assert "(v1)" in result

    def test_with_hints_includes_hints_section(self):
        hints = [_hint("src/foo.py", 0.8), _hint("src/bar.py", 0.6)]
        result = _compose(None, hints)
        assert "Relevant files/functions" in result
        assert "src/foo.py" in result
        assert "src/bar.py" in result

    def test_with_both_skill_and_hints(self):
        result = _compose((_skill(), 0.85), [_hint()])
        assert "Retrieved skill" in result
        assert "Relevant files/functions" in result

    def test_sections_joined_with_double_newline(self):
        result = _compose((_skill(), 0.9), [_hint()])
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# _episode_from_run — pure function
# ---------------------------------------------------------------------------

class TestEpisodeFromRun:
    def test_minimal_run_has_two_steps(self):
        ep = _episode_from_run("fix the bug", None, None, [], "Here is the fix.")
        assert len(ep.steps) == 2
        assert ep.steps[0].kind == "message"
        assert ep.steps[-1].kind == "message"

    def test_first_step_contains_task(self):
        ep = _episode_from_run("add logging", None, None, [], "reply")
        assert "add logging" in ep.steps[0].content

    def test_last_step_contains_reply(self):
        ep = _episode_from_run("task", None, None, [], "the-reply")
        assert "the-reply" in ep.steps[-1].content

    def test_skill_hit_adds_tool_call_step(self):
        ep = _episode_from_run("task", None, (_skill(), 0.9), [], "reply")
        kinds = [s.kind for s in ep.steps]
        assert "tool_call" in kinds
        tool_step = next(s for s in ep.steps if s.kind == "tool_call")
        assert "test-skill" in tool_step.content
        assert "0.900" in tool_step.content

    def test_hints_add_tool_call_step(self):
        hints = [_hint("src/foo.py"), _hint("src/bar.py")]
        ep = _episode_from_run("task", None, None, hints, "reply")
        tool_steps = [s for s in ep.steps if s.kind == "tool_call"]
        assert len(tool_steps) == 1
        assert "src/foo.py" in tool_steps[0].content
        assert "src/bar.py" in tool_steps[0].content

    def test_topic_from_explicit_argument(self):
        ep = _episode_from_run("task", "explicit-topic", (_skill(topic="other"), 0.9), [], "r")
        assert ep.topic == "explicit-topic"

    def test_topic_from_skill_when_no_explicit_topic(self):
        ep = _episode_from_run("task", None, (_skill(topic="skill-topic"), 0.9), [], "r")
        assert ep.topic == "skill-topic"

    def test_topic_fallback_when_no_skill_and_no_topic(self):
        ep = _episode_from_run("task", None, None, [], "r")
        assert ep.topic == FALLBACK_TOPIC

    def test_id_starts_with_run_prefix(self):
        ep = _episode_from_run("task", None, None, [], "r")
        assert ep.id.startswith("run-")

    def test_unique_ids_across_calls(self):
        # Patch datetime.now() so rapid calls don't collide on the same microsecond.
        distinct_times = [datetime(2024, 1, 1, 0, 0, 0, i) for i in range(5)]
        with patch("adaptive_agent.agent.datetime") as mock_dt:
            mock_dt.now.side_effect = distinct_times
            ids = {_episode_from_run("task", None, None, [], "r").id for _ in range(5)}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# answer() — the fixed feedback loop
# ---------------------------------------------------------------------------

PATCH_CHAT = "adaptive_agent.agent.chat"
PATCH_RETRIEVE = "adaptive_agent.agent._retrieve"
PATCH_WRITE = "adaptive_agent.agent.write_episodes"
PATCH_OUTCOME = "adaptive_agent.agent.record_outcome"


@pytest.fixture()
def mock_retrieve_empty():
    with patch(PATCH_RETRIEVE, return_value=(None, [])) as m:
        yield m


@pytest.fixture()
def mock_retrieve_with_skill():
    skill = _skill()
    with patch(PATCH_RETRIEVE, return_value=((skill, 0.85), [_hint()])) as m:
        yield m


class TestAnswer:
    def test_returns_chat_reply(self, mock_retrieve_empty):
        with patch(PATCH_CHAT, return_value="fixed it") as mock_chat, \
             patch(PATCH_WRITE), patch(PATCH_OUTCOME):
            result = answer("fix the bug")
        assert result == "fixed it"

    def test_record_true_calls_write_episodes(self, mock_retrieve_empty):
        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE) as mock_write, \
             patch(PATCH_OUTCOME):
            answer("task", record=True)
        mock_write.assert_called_once()
        written_episodes = mock_write.call_args[0][0]
        assert len(written_episodes) == 1
        assert isinstance(written_episodes[0], Episode)

    def test_record_true_calls_record_outcome(self, mock_retrieve_empty):
        """The feedback loop fix: record_outcome must always be called."""
        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE) as mock_write, \
             patch(PATCH_OUTCOME) as mock_outcome:
            answer("task", record=True)
        mock_outcome.assert_called_once()

    def test_record_outcome_receives_episode_id(self, mock_retrieve_empty):
        captured_episode = None

        def capture_write(episodes):
            nonlocal captured_episode
            captured_episode = episodes[0]

        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE, side_effect=capture_write), \
             patch(PATCH_OUTCOME) as mock_outcome:
            answer("task", record=True)

        outcome_call_args = mock_outcome.call_args
        assert outcome_call_args[0][0] == captured_episode.id

    def test_default_outcome_string_is_informative(self, mock_retrieve_empty):
        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE), \
             patch(PATCH_OUTCOME) as mock_outcome:
            answer("task", record=True)
        outcome_text = mock_outcome.call_args[0][1]
        assert len(outcome_text) > 10  # not an empty placeholder
        assert "answer" in outcome_text.lower() or "delivered" in outcome_text.lower()

    def test_custom_outcome_passed_to_record_outcome(self, mock_retrieve_empty):
        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE), \
             patch(PATCH_OUTCOME) as mock_outcome:
            answer("task", record=True, outcome="tests passed")
        assert mock_outcome.call_args[0][1] == "tests passed"

    def test_record_false_skips_write_and_outcome(self, mock_retrieve_empty):
        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE) as mock_write, \
             patch(PATCH_OUTCOME) as mock_outcome:
            answer("task", record=False)
        mock_write.assert_not_called()
        mock_outcome.assert_not_called()

    def test_episode_topic_uses_skill_topic(self, mock_retrieve_with_skill):
        captured = []

        def capture(episodes):
            captured.extend(episodes)

        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE, side_effect=capture), \
             patch(PATCH_OUTCOME):
            answer("task", topic=None, record=True)

        assert captured[0].topic == "testing"  # from _skill() fixture

    def test_episode_topic_uses_explicit_topic(self, mock_retrieve_with_skill):
        captured = []

        def capture(episodes):
            captured.extend(episodes)

        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE, side_effect=capture), \
             patch(PATCH_OUTCOME):
            answer("task", topic="override-topic", record=True)

        assert captured[0].topic == "override-topic"

    def test_write_called_before_outcome(self, mock_retrieve_empty):
        """write_episodes must precede record_outcome so the episode exists in the DB."""
        call_order = []

        with patch(PATCH_CHAT, return_value="ok"), \
             patch(PATCH_WRITE, side_effect=lambda _: call_order.append("write")), \
             patch(PATCH_OUTCOME, side_effect=lambda *_: call_order.append("outcome")):
            answer("task", record=True)

        assert call_order == ["write", "outcome"]
