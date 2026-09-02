"""Tests for adaptive_agent.ckg.build_graph — no Neo4j required.

Verifies that build() issues three independent count queries (one per entity
type) rather than a single chained query that would produce incorrect counts
via a cartesian product. Also covers load_artifacts error handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest

from adaptive_agent.ckg import build_graph


# ---------------------------------------------------------------------------
# Minimal fixture data
# ---------------------------------------------------------------------------

IMPORTS = [{"src": "a.py", "dst": "b.py", "count": 2}]
FUNCTIONS = [{"id": "a.py::foo", "name": "foo", "file": "a.py", "lineno": 1}]
CALLS = [{"caller": "a.py::foo", "callee": "b.py::bar", "ambiguous": False}]
CO_EDITS = [{"a": "a.py", "b": "b.py", "count": 3}]


def _run_read_side_effect(query, **_kwargs):
    """Return plausible counts for the three independent count queries."""
    q = query.strip()
    if "CodeFile" in q:
        return [{"n": 2}]
    if "CodeFunction" in q:
        return [{"n": 1}]
    if "IMPORTS|CALLS|CO_EDITED|CONTAINS" in q:
        return [{"n": 4}]
    return []


# ---------------------------------------------------------------------------
# load_artifacts
# ---------------------------------------------------------------------------

class TestLoadArtifacts:
    def test_raises_when_artifact_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_graph.config, "ARTIFACTS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="ckg_imports.json"):
            build_graph.load_artifacts()

    def test_raises_with_hint_to_run_parse_stage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_graph.config, "ARTIFACTS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="05_build_ckg"):
            build_graph.load_artifacts()

    def test_loads_all_four_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_graph.config, "ARTIFACTS_DIR", tmp_path)
        (tmp_path / "ckg_imports.json").write_text(json.dumps(IMPORTS))
        (tmp_path / "ckg_functions.json").write_text(json.dumps(FUNCTIONS))
        (tmp_path / "ckg_calls.json").write_text(json.dumps(CALLS))
        (tmp_path / "ckg_co_edits.json").write_text(json.dumps(CO_EDITS))

        imports, functions, calls, co_edits = build_graph.load_artifacts()
        assert imports == IMPORTS
        assert functions == FUNCTIONS
        assert calls == CALLS
        assert co_edits == CO_EDITS


# ---------------------------------------------------------------------------
# build() — count query structure (the fix)
# ---------------------------------------------------------------------------

PATCH_LOAD = "adaptive_agent.ckg.build_graph.load_artifacts"
PATCH_READ = "adaptive_agent.ckg.build_graph.run_read"
PATCH_WRITE = "adaptive_agent.ckg.build_graph.run_write"


class TestBuildCounts:
    def test_returns_dict_with_required_keys(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect):
            counts = build_graph.build()

        assert set(counts.keys()) == {"files", "functions", "edges"}

    def test_issues_exactly_three_read_queries(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        assert mock_read.call_count == 3

    def test_each_count_query_targets_single_entity(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        queries = [c.args[0].strip() for c in mock_read.call_args_list]
        # Every query must be a simple MATCH … RETURN count — no WITH chaining
        for q in queries:
            assert "WITH" not in q, f"count query must not chain WITH: {q!r}"

    def test_no_chained_match_with_pattern(self):
        """The original bug: MATCH … WITH … MATCH … WITH … chained in one query."""
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        for c in mock_read.call_args_list:
            q = c.args[0]
            # The old bug pattern: multiple MATCH clauses in one query
            assert q.upper().count("MATCH") <= 1, (
                f"count query must not chain multiple MATCH clauses: {q!r}"
            )

    def test_file_count_query_targets_codefile(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        queries = [c.args[0] for c in mock_read.call_args_list]
        assert any("CodeFile" in q and "CodeFunction" not in q for q in queries)

    def test_function_count_query_targets_codefunction(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        queries = [c.args[0] for c in mock_read.call_args_list]
        assert any("CodeFunction" in q for q in queries)

    def test_edge_count_query_targets_relationship_types(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect) as mock_read:
            build_graph.build()

        queries = [c.args[0] for c in mock_read.call_args_list]
        assert any("IMPORTS" in q and "CALLS" in q for q in queries)

    def test_count_values_flow_from_read_into_result(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE), \
             patch(PATCH_READ, side_effect=_run_read_side_effect):
            counts = build_graph.build()

        assert counts["files"] == 2
        assert counts["functions"] == 1
        assert counts["edges"] == 4


# ---------------------------------------------------------------------------
# build() — write query structure
# ---------------------------------------------------------------------------

class TestBuildWrites:
    def test_issues_five_write_queries(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE) as mock_write, \
             patch(PATCH_READ, side_effect=_run_read_side_effect):
            build_graph.build()

        # files, functions+CONTAINS, IMPORTS, CALLS, CO_EDITED
        assert mock_write.call_count == 5

    def test_file_paths_include_all_referenced_files(self):
        with patch(PATCH_LOAD, return_value=(IMPORTS, FUNCTIONS, CALLS, CO_EDITS)), \
             patch(PATCH_WRITE) as mock_write, \
             patch(PATCH_READ, side_effect=_run_read_side_effect):
            build_graph.build()

        first_call = mock_write.call_args_list[0]
        paths = first_call.kwargs.get("paths") or first_call.args[1] if len(first_call.args) > 1 else first_call.kwargs["paths"]
        assert "a.py" in paths
        assert "b.py" in paths
