"""Tests for adaptive_agent.skills.skill_box — no Neo4j or GPU required.

Focuses on the embedding cache: verifies that search() only encodes the
query (not the skills) when cached embeddings are present, and that
save_skill() stores the embedding alongside the skill properties.
"""

from unittest.mock import call, patch

import numpy as np
import pytest

from adaptive_agent.skills import skill_box
from adaptive_agent.skills.models import Skill

PATCH_RUN_READ = "adaptive_agent.skills.skill_box.run_read"
PATCH_RUN_WRITE = "adaptive_agent.skills.skill_box.run_write"
PATCH_EMBED = "adaptive_agent.skills.skill_box.embed_texts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skill(name="debug", topic="debugging", version=1, procedure="1. Check logs."):
    return Skill(
        name=name, topic=topic, version=version,
        status="active", procedure=procedure,
    )


def _unit_vec(dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _skill_props(skill: Skill, embedding=None) -> dict:
    return {
        "name": skill.name, "topic": skill.topic, "version": skill.version,
        "status": skill.status, "procedure": skill.procedure,
        "evidence": "", "review_note": "",
        **({"embedding": embedding.tolist()} if embedding is not None else {}),
    }


# ---------------------------------------------------------------------------
# _skill_text
# ---------------------------------------------------------------------------

class TestSkillText:
    def test_contains_name_topic_and_procedure(self):
        skill = _skill(name="foo", topic="bar", procedure="do it")
        text = skill_box._skill_text(skill)
        assert "foo" in text
        assert "bar" in text
        assert "do it" in text

    def test_format_is_stable(self):
        skill = _skill()
        assert skill_box._skill_text(skill) == "debug. topic: debugging.\n1. Check logs."


# ---------------------------------------------------------------------------
# save_skill — embedding is written alongside skill properties
# ---------------------------------------------------------------------------

class TestSaveSkill:
    def test_run_write_called_twice(self):
        skill = _skill()
        fake_vec = _unit_vec()
        with patch(PATCH_RUN_WRITE) as mock_write, \
             patch(PATCH_EMBED, return_value=np.stack([fake_vec])):
            skill_box.save_skill(skill)
        assert mock_write.call_count == 2

    def test_second_write_stores_embedding(self):
        skill = _skill()
        fake_vec = _unit_vec()
        with patch(PATCH_RUN_WRITE) as mock_write, \
             patch(PATCH_EMBED, return_value=np.stack([fake_vec])):
            skill_box.save_skill(skill)
        embedding_write = mock_write.call_args_list[1]
        query = embedding_write.args[0]
        assert "embedding" in query
        assert embedding_write.kwargs["name"] == skill.name
        assert embedding_write.kwargs["version"] == skill.version
        stored = embedding_write.kwargs["vec"]
        np.testing.assert_allclose(stored, fake_vec.tolist(), rtol=1e-5)

    def test_embed_texts_called_with_skill_text(self):
        skill = _skill()
        with patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, return_value=np.stack([_unit_vec()])) as mock_embed:
            skill_box.save_skill(skill)
        texts_embedded = mock_embed.call_args[0][0]
        assert skill_box._skill_text(skill) in texts_embedded


# ---------------------------------------------------------------------------
# search — cached path: only the query is encoded
# ---------------------------------------------------------------------------

class TestSearchCachedEmbeddings:
    def _rows(self, *skills_with_vecs):
        return [{"s": _skill_props(s, v)} for s, v in skills_with_vecs]

    def test_embed_texts_called_once_for_query_only(self):
        skill = _skill()
        skill_vec = _unit_vec(seed=1)
        query_vec = _unit_vec(seed=2)
        rows = self._rows((skill, skill_vec))

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, return_value=np.stack([query_vec])) as mock_embed:
            skill_box.search("some task")

        assert mock_embed.call_count == 1, (
            "embed_texts should be called once (query only) when all skills have cached embeddings"
        )

    def test_returns_empty_when_no_active_skills(self):
        with patch(PATCH_RUN_READ, return_value=[]):
            result = skill_box.search("task")
        assert result == []

    def test_returns_skill_above_threshold(self):
        skill = _skill()
        # Make skill vec identical to query vec so similarity = 1.0
        vec = _unit_vec(seed=7)
        rows = self._rows((skill, vec))

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, return_value=np.stack([vec])):
            result = skill_box.search("task", min_similarity=0.30)

        assert len(result) == 1
        returned_skill, score = result[0]
        assert returned_skill.name == skill.name
        assert score > 0.99

    def test_drops_skill_below_threshold(self):
        skill = _skill()
        # Orthogonal vectors → similarity = 0
        skill_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        rows = self._rows((skill, skill_vec))

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, return_value=np.stack([query_vec])):
            result = skill_box.search("task", min_similarity=0.30)

        assert result == []

    def test_returns_top_k_skills(self):
        skills = [_skill(name=f"s{i}") for i in range(3)]
        # Give each a distinct vec; query vec identical to skill 0
        vecs = [_unit_vec(seed=i) for i in range(3)]
        query_vec = vecs[0]
        rows = [{"s": _skill_props(s, v)} for s, v in zip(skills, vecs)]

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, return_value=np.stack([query_vec])):
            result = skill_box.search("task", k=2, min_similarity=0.0)

        assert len(result) == 2
        assert result[0][0].name == "s0"  # highest similarity first


# ---------------------------------------------------------------------------
# search — uncached path: embedding computed and persisted for legacy nodes
# ---------------------------------------------------------------------------

class TestSearchUncachedEmbeddings:
    def test_embed_texts_called_for_legacy_skill_and_query(self):
        skill = _skill()
        skill_vec = _unit_vec(seed=3)
        query_vec = _unit_vec(seed=4)
        rows = [{"s": _skill_props(skill)}]  # no "embedding" key

        embed_returns = [np.stack([skill_vec]), np.stack([query_vec])]

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE), \
             patch(PATCH_EMBED, side_effect=embed_returns) as mock_embed:
            skill_box.search("task")

        assert mock_embed.call_count == 2

    def test_legacy_embedding_is_persisted(self):
        skill = _skill()
        skill_vec = _unit_vec(seed=5)
        query_vec = _unit_vec(seed=6)
        rows = [{"s": _skill_props(skill)}]

        embed_returns = [np.stack([skill_vec]), np.stack([query_vec])]

        with patch(PATCH_RUN_READ, return_value=rows), \
             patch(PATCH_RUN_WRITE) as mock_write, \
             patch(PATCH_EMBED, side_effect=embed_returns):
            skill_box.search("task")

        assert mock_write.call_count == 1
        write_query = mock_write.call_args.args[0]
        assert "embedding" in write_query
