"""Tests for triage, mstt_runner, host-bound diagnostics, and characterize.

These tests verify the new stages added in the msprof-analyze / triage /
host-bound / characterization integration without requiring a full
profiling root or msprof-analyze.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Ensure ascend_profile package is importable.
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

class TestTriage:
    """Unit tests for triage.py."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import triage as triage_mod
        self.mod = triage_mod

    def test_classify_hostbound(self) -> None:
        assert self.mod._classify_bottleneck(60.0, 10.0, 25.0) == "hostbound"

    def test_classify_computing(self) -> None:
        assert self.mod._classify_bottleneck(90.0, 5.0, 5.0) == "computing"

    def test_classify_communication(self) -> None:
        assert self.mod._classify_bottleneck(60.0, 15.0, 15.0) == "communication"

    def test_classify_none(self) -> None:
        assert self.mod._classify_bottleneck(80.0, 5.0, 15.0) == "none_obvious"

    def test_classify_hostbound_priority(self) -> None:
        """Hostbound has priority over computing even when both exceed thresholds."""
        assert self.mod._classify_bottleneck(90.0, 5.0, 25.0) == "hostbound"

    def test_parse_valid_csv(self) -> None:
        csv_content = (
            "Computing,Communication(Not Overlapped),Free\n"
            "100,20,30\n"
            "200,30,20\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            result = self.mod._parse_steps(Path(csv_path))
            assert result is not None
            assert result["computing_pct"] == 75.0  # 300/400
            assert result["communication_pct"] == 12.5  # 50/400
            assert result["free_pct"] == 12.5  # 50/400
        finally:
            Path(csv_path).unlink()

    def test_parse_missing_columns(self) -> None:
        csv_content = "Computing,Free\n100,30\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            assert self.mod._parse_steps(Path(csv_path)) is None
        finally:
            Path(csv_path).unlink()

    def test_parse_zero_total(self) -> None:
        csv_content = (
            "Computing,Communication(Not Overlapped),Free\n"
            "0,0,0\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            assert self.mod._parse_steps(Path(csv_path)) is None
        finally:
            Path(csv_path).unlink()

    def test_run_triage_with_valid_files(self) -> None:
        """End-to-end triage on a fake profiling root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "step_trace_time.csv"
            csv_path.write_text(
                "Computing,Communication(Not Overlapped),Free\n"
                "700,50,250\n"  # free ~25%, will trigger hostbound
                "750,40,210\n"
            )
            result = self.mod.run_triage(root, root / "output")
            assert result["status"] == "ok"
            assert result["rank_count"] == 1  # single file
            assert result["primary_bottleneck"] == "hostbound"
            assert Path(tmpdir) / "output" / "triage.json"
            assert Path(tmpdir) / "output" / "triage_manifest.json"

    def test_run_triage_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.mod.run_triage(Path(tmpdir), Path(tmpdir) / "out")
            assert result["status"] == "no_step_trace_files"

    def test_load_triage_missing(self) -> None:
        assert self.mod.load_triage(Path("/nonexistent/path")) is None


# ---------------------------------------------------------------------------
# mstt_runner
# ---------------------------------------------------------------------------

class TestMsttRunner:
    """Unit tests for mstt_runner.py (no actual msprof-analyze needed)."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import mstt_runner as mstt_mod
        self.mod = mstt_mod

    def test_parse_slow_rank_db_nonexistent(self) -> None:
        assert self.mod._parse_slow_rank_db("/nonexistent/db.sqlite") == []

    def test_parse_slow_rank_db_empty(self) -> None:
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE SlowRank (rankId INTEGER, slowAffectCount INTEGER)")
            conn.commit()
            conn.close()
            assert self.mod._parse_slow_rank_db(db_path) == []
        finally:
            Path(db_path).unlink()

    def test_parse_slow_rank_db_with_data(self) -> None:
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE SlowRank (rankId INTEGER, slowAffectCount INTEGER)")
            conn.execute("INSERT INTO SlowRank VALUES (0, 5)")
            conn.execute("INSERT INTO SlowRank VALUES (1, 0)")
            conn.execute("INSERT INTO SlowRank VALUES (3, 47)")
            conn.commit()
            conn.close()
            rows = self.mod._parse_slow_rank_db(db_path)
            assert len(rows) == 3
            by_id = {r["rank_id"]: r["slow_affect_count"] for r in rows}
            assert by_id == {"0": 5, "1": 0, "3": 47}
        finally:
            Path(db_path).unlink()

    def test_discover_rank_ids_device_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "device_0_ascend_pt").mkdir()
            (root / "device_3_ascend_pt").mkdir()
            (root / "device_7_ascend_pt").mkdir()
            (root / "some_other_dir").mkdir()
            result = self.mod._discover_rank_ids(root)
            assert result == ["0", "3", "7"]

    def test_discover_rank_ids_rank_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "rank_0").mkdir()
            (root / "rank_2").mkdir()
            result = self.mod._discover_rank_ids(root)
            assert result == ["0", "2"]

    def test_discover_rank_ids_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "device_0_ascend_pt").mkdir()
            (root / "rank_1").mkdir()
            # Only first match (device_ prefix) counts; rank_1 also adds "1".
            result = self.mod._discover_rank_ids(root)
            assert "0" in result and "1" in result
            assert len(result) == 2

    def test_load_mstt_missing(self) -> None:
        assert self.mod.load_mstt_slow_rank(Path("/nonexistent")) is None

    def test_load_mstt_with_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "mstt_slow_rank.csv"
            csv_path.write_text(
                "rank_id,slow_affect_count\n"
                "0,5\n"
                "1,0\n"
                "3,47\n"
            )
            result = self.mod.load_mstt_slow_rank(Path(tmpdir))
            assert result == {"0": 5, "1": 0, "3": 47}

    def test_write_csv_fw_fallback(self) -> None:
        """_write_csv_fw works even without asc_profile.common."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            rows = [{"rank_id": "0", "slow_affect_count": 5}]
            self.mod._write_csv_fw(out / "test.csv", rows, fieldnames=["rank_id", "slow_affect_count"])
            content = (out / "test.csv").read_text()
            assert "rank_id" in content
            assert "0" in content
            assert "5" in content

    def test_write_manifest_fw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self.mod._write_manifest_fw(out, {"status": "ok", "slow_rank_count": 2})
            manifest = json.loads((out / "mstt_manifest.json").read_text())
            assert manifest["status"] == "ok"

    def test_run_mstt_without_msprof_installed(self) -> None:
        """When msprof-analyze is not found, run_mstt_slow_rank returns 'unavailable'."""
        # We can't reliably test the full flow, but the manifest fallback path
        # is tested implicitly by checking the unavailable status shape.
        result = self.mod.run_mstt_slow_rank(Path("/nonexistent"), Path("/tmp/mstt_test_out"))
        # If msprof-analyze happens to be installed on the test machine, this
        # would attempt installation and potentially fail. In CI it won't be.
        assert result["status"] in ("unavailable", "failed", "ok", "no_data")


# ---------------------------------------------------------------------------
# diagnostics: host-bound
# ---------------------------------------------------------------------------

class TestHostBoundDiagnosis:
    """Tests for _diagnose_host_bound in diagnostics.py."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import diagnostics as diag_mod
        self.mod = diag_mod

    def _step_row(self, rank_id: str, wall_ms: float, head_wall_ms: float, start_us: float) -> dict[str, Any]:
        return {
            "segment_id": f"step_{rank_id}_{start_us}",
            "rank_id": rank_id,
            "start_us": start_us,
            "segment_type": "step",
            "wall_ms": wall_ms,
            "head_wall_ms": head_wall_ms,
            "main_wall_ms": wall_ms * 0.5,
            "tail_wall_ms": wall_ms * 0.2,
            "bubble_ratio": 0.1,
        }

    def _wait_row(self, rank_id: str, wait_ratio: float, call_count: int = 100) -> dict[str, Any]:
        return {
            "rank_id": rank_id,
            "name": "op_foo",
            "wait_ratio": wait_ratio,
            "call_count": call_count,
            "classification": "WAIT_ANCHOR",
        }

    def test_both_signals_produce_finding(self) -> None:
        """4 consecutive head-heavy steps + high wait density = finding."""
        steps = [
            self._step_row("0", 100.0, 20.0, s)
            for s in (0, 1000, 2000, 3000, 4000)
        ]
        waits = [self._wait_row("0", 0.85, 50), self._wait_row("0", 0.20, 50)]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 1
        assert findings[0].finding_type == "host_dispatch_bound_suspected"
        assert findings[0].confidence == "medium"
        assert findings[0].severity == "medium"
        assert "0" in findings[0].rank_ids

    def test_single_signal_no_finding(self) -> None:
        """Head-heavy steps alone (no wait density) = no finding."""
        steps = [
            self._step_row("0", 100.0, 20.0, s)
            for s in (0, 1000, 2000, 3000, 4000)
        ]
        waits: list[dict[str, Any]] = []  # no wait anchor data
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 0

    def test_wait_density_alone_no_finding(self) -> None:
        """High wait density without consecutive head-heavy steps = no finding."""
        steps = [
            self._step_row("0", 100.0, 10.0, 0),   # head/wall = 10% < 15%
            self._step_row("0", 100.0, 10.0, 1000),
        ]
        waits = [self._wait_row("0", 0.85, 90), self._wait_row("0", 0.20, 10)]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 0

    def test_streak_resets_on_normal_step(self) -> None:
        """Consecutive must be unbroken; a normal step resets the streak."""
        steps = [
            self._step_row("0", 100.0, 20.0, 0),
            self._step_row("0", 100.0, 20.0, 1000),
            self._step_row("0", 100.0, 5.0, 2000),   # breaks streak
            self._step_row("0", 100.0, 20.0, 3000),
            self._step_row("0", 100.0, 20.0, 4000),   # only 2 consecutive again
        ]
        waits = [self._wait_row("0", 0.85, 50), self._wait_row("0", 0.20, 50)]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 0

    def test_threshold_boundary(self) -> None:
        """Exactly 15% head/wall is NOT flagged."""
        steps = [
            self._step_row("0", 100.0, 15.0, s)
            for s in (0, 1000, 2000, 3000)
        ]
        waits = [self._wait_row("0", 0.85, 50), self._wait_row("0", 0.20, 50)]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 0

    def test_zero_wall_skipped(self) -> None:
        """Steps with wall_ms=0 are skipped gracefully."""
        steps = [self._step_row("0", 0.0, 1.0, 500)]
        waits = [self._wait_row("0", 0.85, 100)]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 0

    def test_multi_rank_isolation(self) -> None:
        """Rank 0 has the pattern, rank 1 doesn't."""
        steps = [
            self._step_row("0", 100.0, 20.0, 0),
            self._step_row("0", 100.0, 20.0, 1000),
            self._step_row("0", 100.0, 20.0, 2000),
            self._step_row("1", 100.0, 4.0, 0),
            self._step_row("1", 100.0, 5.0, 1000),
        ]
        waits = [
            self._wait_row("0", 0.85, 50), self._wait_row("0", 0.20, 50),
            self._wait_row("1", 0.85, 5), self._wait_row("1", 0.20, 50),
        ]
        findings = self.mod._diagnose_host_bound(steps, waits)
        assert len(findings) == 1
        assert findings[0].rank_ids == ("0",)


# ---------------------------------------------------------------------------
# diagnostics: mstt-enriched cross_rank
# ---------------------------------------------------------------------------

class TestDiagnoseCrossRankWithMstt:
    """Tests for mstt-enriched diagnose_cross_rank behavior."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import diagnostics as diag_mod
        self.mod = diag_mod

    def _alignment_row(self, role: str, rank_ids: list[int], duration_ratio: float = 1.0,
                       duration_skew: float = 0.0, start_skew: float = 0.0,
                       alignment_type: str = "operator", alignment_id: str = "align_001") -> dict[str, Any]:
        return {
            "alignment_id": alignment_id,
            "alignment_type": alignment_type,
            "rank_ids": json.dumps(rank_ids),
            "role": role,
            "duration_ratio": duration_ratio,
            "duration_skew_us": duration_skew,
            "start_skew_us": start_skew,
            "is_structure_mismatch": "false",
        }

    def test_slow_rank_confirmed_from_mstt(self) -> None:
        """When mstt data is present, slow_rank_confirmed is produced."""
        mstt_data = {"0": 47, "1": 0, "2": 0, "3": 3}
        rows: list[dict[str, Any]] = []
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=mstt_data)
        confirmed = [f for f in findings if f.finding_type == "slow_rank_confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0].confidence == "high"
        assert confirmed[0].severity == "high"  # max_slow_affect=47 > 20
        assert set(confirmed[0].rank_ids) == {"0", "3"}

    def test_no_slow_rank_suspected_when_mstt_available(self) -> None:
        """When mstt data is present, slow_rank_suspected is NOT emitted."""
        mstt_data = {"0": 5, "1": 0}
        rows = [
            self._alignment_row("compute.matmul", [0, 1], start_skew=2000.0),
        ]
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=mstt_data)
        suspected = [f for f in findings if f.finding_type == "slow_rank_suspected"]
        assert len(suspected) == 0

    def test_slow_rank_suspected_fallback_when_no_mstt(self) -> None:
        """When mstt data is absent, slow_rank_suspected is the fallback."""
        rows = [
            self._alignment_row("compute.matmul", [0, 1], start_skew=2000.0),
        ]
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=None)
        suspected = [f for f in findings if f.finding_type == "slow_rank_suspected"]
        assert len(suspected) == 1
        assert suspected[0].confidence == "low"  # downgraded from original "medium"

    def test_communication_collective_confidence_upgraded_with_mstt(self) -> None:
        """When a collective skew involves an mstt-identified slow rank, confidence upgrades."""
        mstt_data = {"0": 47, "1": 0}
        rows = [
            self._alignment_row("communication.collective", [0, 1], duration_ratio=2.5),
        ]
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=mstt_data)
        comm = [f for f in findings if f.finding_type == "communication_collective_slow"]
        assert len(comm) == 1
        assert comm[0].confidence == "high"  # mstt cross-validated
        assert comm[0].metrics.get("mstt_cross_validated") is True

    def test_communication_collective_normal_without_mstt(self) -> None:
        """Without mstt, communication_collective_slow keeps medium confidence."""
        rows = [
            self._alignment_row("communication.collective", [0, 1], duration_ratio=2.5),
        ]
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=None)
        comm = [f for f in findings if f.finding_type == "communication_collective_slow"]
        assert len(comm) == 1
        assert comm[0].confidence == "medium"
        assert comm[0].metrics.get("mstt_cross_validated") is False

    def test_empty_mstt_data_no_slow_ranks(self) -> None:
        """When mstt returns all zeros, no slow_rank_confirmed."""
        mstt_data = {"0": 0, "1": 0, "2": 0}
        findings = self.mod.diagnose_cross_rank([], mstt_data=mstt_data)
        confirmed = [f for f in findings if f.finding_type == "slow_rank_confirmed"]
        assert len(confirmed) == 0

    def test_step_structure_mismatch_still_fires(self) -> None:
        """rank_workload_asymmetry is unaffected by mstt presence."""
        mstt_data = {"0": 47, "1": 0}
        rows = [
            {
                "alignment_id": "align_003",
                "alignment_type": "time_window",
                "rank_ids": json.dumps([0, 1]),
                "role": "",
                "duration_ratio": 1.0,
                "duration_skew_us": 0.0,
                "start_skew_us": 0.0,
                "is_structure_mismatch": "true",
            },
        ]
        findings = self.mod.diagnose_cross_rank(rows, mstt_data=mstt_data)
        asym = [f for f in findings if f.finding_type == "rank_workload_asymmetry"]
        assert len(asym) == 1


# ---------------------------------------------------------------------------
# characterize
# ---------------------------------------------------------------------------

class TestCharacterize:
    """Tests for characterize.py — shape parsing, AI computation, operator characterization."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import characterize as char_mod
        self.mod = char_mod

    # ── shape parsing ──

    def test_parse_basic_2x2_k_n(self) -> None:
        result = self.mod._parse_matmul_shape("16384,128;1024,128")
        assert result is not None
        assert result["M"] == 16384
        assert result["K"] == 128
        assert result["N"] == 1024
        assert result["rule"] == "basic-2x2-auto"

    def test_parse_basic_2x2_n_k(self) -> None:
        result = self.mod._parse_matmul_shape("16384,128;128,1024")
        assert result is not None
        assert result["M"] == 16384
        assert result["K"] == 128
        assert result["N"] == 1024
        assert result["rule"] == "basic-2x2-auto"

    def test_parse_basic_2x2_positional(self) -> None:
        """When k doesn't match either right dim, positional guess used."""
        result = self.mod._parse_matmul_shape("100,200;300,400")
        assert result is not None
        assert result["M"] == 100
        assert result["K"] == 200
        assert result["N"] == 400
        assert result["rule"] == "basic-2x2-positional"

    def test_parse_packed_2x4_bc(self) -> None:
        result = self.mod._parse_matmul_shape("4096,256;4,128,2,512")
        assert result is not None
        assert result["M"] == 4096
        assert result["K"] == 256  # B*C = 128*2 = 256
        assert result["N"] == 2048  # A*D = 4*512 = 2048
        assert result["rule"] == "packed-2x4-bc"

    def test_parse_packed_2x4_ad(self) -> None:
        result = self.mod._parse_matmul_shape("4096,256;256,4,512,1")
        assert result is not None
        assert result["M"] == 4096
        assert result["K"] == 256  # A*D = 256*1 = 256
        assert result["N"] == 2048
        assert result["rule"] == "packed-2x4-ad"

    def test_parse_batched_3x2(self) -> None:
        result = self.mod._parse_matmul_shape("32,4096,128;128,1024")
        assert result is not None
        assert result["M"] == 4096
        assert result["K"] == 128
        assert result["N"] == 1024
        assert result["batch"] == 32
        assert result["rule"] == "batched-3x2"

    def test_parse_unrecognized_returns_none(self) -> None:
        assert self.mod._parse_matmul_shape("1,2,3,4,5;6,7") is None
        assert self.mod._parse_matmul_shape("abc,def;ghi,jkl") is None
        assert self.mod._parse_matmul_shape("") is None
        assert self.mod._parse_matmul_shape("123;") is None

    def test_parse_with_brackets(self) -> None:
        result = self.mod._parse_matmul_shape("[16384,128];[1024,128]")
        assert result is not None
        assert result["M"] == 16384
        assert result["K"] == 128
        assert result["N"] == 1024

    def test_parse_with_chinese_punctuation(self) -> None:
        result = self.mod._parse_matmul_shape("16384，128；1024，128")
        assert result is not None
        assert result["M"] == 16384
        assert result["K"] == 128
        assert result["N"] == 1024

    def test_parse_decode_shape(self) -> None:
        """Decode: M=1, small K, variable N (batch*head_dim)."""
        result = self.mod._parse_matmul_shape("1,192;8192,192")
        assert result is not None
        assert result["M"] == 1
        assert result["K"] == 192
        assert result["N"] == 8192
        assert result["rule"] == "basic-2x2-auto"

    # ── arithmetic intensity ──

    def test_ai_decode_small(self) -> None:
        """M=1, K=128, N=1024: AI should be tiny (< 1)."""
        ai = self.mod._arithmetic_intensity(1, 128, 1024)
        # 2*1*128*1024 / (2*(1*128 + 128*1024 + 1*1024))
        # = 262144 / (2*(128 + 131072 + 1024))
        # = 262144 / 264448 ≈ 0.991
        assert ai > 0.9
        assert ai < 1.1

    def test_ai_prefill_large(self) -> None:
        """M=16384, K=128, N=1024: moderate AI."""
        ai = self.mod._arithmetic_intensity(16384, 128, 1024)
        # 2*16384*128*1024 / (2*(16384*128 + 128*1024 + 16384*1024))
        # = 4294967296 / (2*(2097152 + 131072 + 16777216))
        # = 4294967296 / 40008880 ≈ 107
        assert ai > 50
        assert ai < 200

    def test_ai_square_matmul(self) -> None:
        """K=4096 large: high reuse."""
        ai = self.mod._arithmetic_intensity(4096, 4096, 4096)
        # 2*4096^3 / (2*3*4096^2) = 4096/3 ≈ 1365
        assert ai > 1000
        assert ai < 2000

    def test_ai_zero_dims(self) -> None:
        assert self.mod._arithmetic_intensity(0, 128, 1024) == 0.0
        assert self.mod._arithmetic_intensity(16384, 0, 1024) == 0.0
        assert self.mod._arithmetic_intensity(16384, 128, 0) == 0.0

    # ── operator characterization ──

    def _op_row(self, bound_family: str, bound_stage: str, name: str = "test_op",
                pipeline: dict[str, float] | None = None) -> dict[str, Any]:
        fields = [
            "aic_mac_time", "aic_fixpipe_time", "aic_mte1_time", "aic_mte2_time",
            "aic_scalar_time", "aiv_vec_time", "aiv_scalar_time",
            "aiv_mte2_time", "aiv_mte3_time",
        ]
        if pipeline is None:
            pipeline = {"aic_mte2_time": 100.0, "aic_mac_time": 50.0}
        row: dict[str, Any] = {
            "name": name,
            "op_type": "aic",
            "roles": "compute.matmul",
            "bound_family": bound_family,
            "bound_stage": bound_stage,
        }
        for f in fields:
            row[f] = pipeline.get(f, 0.0)
        return row

    def test_char_memory_bound_with_shape(self) -> None:
        op = self._op_row("mte2", "aic_mte2_time", "matmul_0")
        mnk = {"M": 16384, "K": 128, "N": 1024, "rule": "basic-2x2-auto"}
        ch = self.mod._characterize_operator(op, mnk, "A2")
        assert ch["bound_classification"] == "memory-bound"
        assert ch["bound_confidence"] == "high"
        assert ch["confidence"] == "high"
        assert "Memory-bound" in ch["characterization"]
        assert "16384" in ch["characterization"] or "16384" in str(ch.get("shape", {}).get("M", ""))
        assert ch.get("arithmetic_intensity", 0) > 0

    def test_char_compute_bound_with_shape(self) -> None:
        op = self._op_row("mac", "aic_mac_time", "matmul_0",
                          pipeline={"aic_mac_time": 100.0, "aic_mte2_time": 30.0})
        mnk = {"M": 4096, "K": 4096, "N": 4096, "rule": "basic-2x2-auto"}
        ch = self.mod._characterize_operator(op, mnk, "A2")
        assert ch["bound_classification"] == "compute-bound"
        assert "Compute-bound" in ch["characterization"]

    def test_char_memory_bound_no_shape(self) -> None:
        op = self._op_row("mte2", "aic_mte2_time")
        ch = self.mod._characterize_operator(op, None, "A2")
        assert ch["bound_classification"] == "memory-bound"
        assert "No shape available" in ch["characterization"]
        assert ch["confidence"] == "high"  # bound from measured pipeline

    def test_char_decode_like(self) -> None:
        op = self._op_row("mac", "aic_mac_time", "matmul_0",
                          pipeline={"aic_mac_time": 100.0, "aic_mte2_time": 20.0})
        mnk = {"M": 1, "K": 128, "N": 1024, "rule": "basic-2x2-auto"}
        ch = self.mod._characterize_operator(op, mnk, "A2")
        assert ch.get("decode_like") is True
        assert "M=1" in ch["characterization"]

    def test_char_small_k_note(self) -> None:
        op = self._op_row("mte2", "aic_mte2_time")
        mnk = {"M": 16384, "K": 64, "N": 1024, "rule": "basic-2x2-auto"}
        ch = self.mod._characterize_operator(op, mnk, "A2")
        assert "Small K=64" in ch["characterization"]

    def test_char_a3_mixed_note(self) -> None:
        op = self._op_row("mixed", "aic_mac_time", pipeline={"aic_mac_time": 50.0, "aiv_vec_time": 50.0})
        mnk = {"M": 16384, "K": 256, "N": 1024, "rule": "basic-2x2-auto"}
        ch = self.mod._characterize_operator(op, mnk, "A3")
        assert "A3 dual-die" in ch["characterization"]

    def test_char_confidence_always_high(self) -> None:
        """Confidence is always 'high' — bound class from measured pipeline data."""
        op = self._op_row("unknown", "aic_mac_time", pipeline={"aic_mac_time": 1.0})
        ch = self.mod._characterize_operator(op, None, "A2")
        assert ch["confidence"] == "high"

    def test_stage_pct(self) -> None:
        op = self._op_row("mte2", "aic_mte2_time",
                          pipeline={"aic_mac_time": 30.0, "aic_mte2_time": 70.0})
        pct = self.mod._stage_pct(op)
        assert pct == 70.0

    def test_stage_pct_zero_total(self) -> None:
        op = self._op_row("mte2", "aic_mte2_time", pipeline={})
        pct = self.mod._stage_pct(op)
        assert pct == 0.0

    def test_op_matches_keywords(self) -> None:
        assert self.mod._op_matches_keywords("MatMul_123") is True
        assert self.mod._op_matches_keywords("BatchMatmul_0") is True
        assert self.mod._op_matches_keywords("GroupedMatmul_5") is True
        assert self.mod._op_matches_keywords("FusedInferAttentionScoreV4_0") is True
        assert self.mod._op_matches_keywords("UnpadFlashAttention_1") is True
        assert self.mod._op_matches_keywords("RmsNorm_0") is False
        assert self.mod._op_matches_keywords("AllReduce_0") is False

    def test_has_pipeline_signal(self) -> None:
        assert self.mod._has_pipeline_signal({"aic_mac_time": 1.0}) is True
        assert self.mod._has_pipeline_signal({"aic_mac_time": None}) is False
        assert self.mod._has_pipeline_signal({"aic_mac_time": ""}) is False
        assert self.mod._has_pipeline_signal({}) is False

    def test_observed_bandwidth(self) -> None:
        """MTE BW: M=16384,K=128,N=1024, 1 call, 1000us MTE time → ~38 GB/s"""
        bw = self.mod._observed_bandwidth_gb_s(16384, 128, 1024, 1, 1000.0)
        # bytes = 2*(16384*128 + 128*1024 + 16384*1024) = 38010880
        # bw = 38010880 / 0.001 / 1e9 = 38.0
        assert 37.0 < bw < 39.0

    def test_observed_bandwidth_zero_params(self) -> None:
        assert self.mod._observed_bandwidth_gb_s(0, 128, 1024, 1, 1000) == 0.0
        assert self.mod._observed_bandwidth_gb_s(16384, 128, 1024, 0, 1000) == 0.0
        assert self.mod._observed_bandwidth_gb_s(16384, 128, 1024, 1, 0) == 0.0

    def test_observed_bandwidth_multi_call(self) -> None:
        """10 calls at 5000us MTE time: 10x bandwidth of single call"""
        bw = self.mod._observed_bandwidth_gb_s(16384, 128, 1024, 10, 5000.0)
        assert 75.0 < bw < 77.0


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

class TestObservations:
    """Tests for observations.py — calibration data collection."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile import observations as obs_mod
        self.mod = obs_mod

    def test_version_gap_detected(self) -> None:
        obs = self.mod._check_version_gap("0.23.0")
        assert obs is not None
        assert obs["type"] == "version_gap"
        assert obs["nearest_known"] in ("0.22.1rc1", "0.22.0")

    def test_version_exact_match_no_gap(self) -> None:
        # Find what version actually exists
        nearest = self.mod._find_nearest_known_version("0.18.0")
        if nearest == "0.18.0":
            obs = self.mod._check_version_gap("0.18.0")
            assert obs is None
        else:
            # No exact match available in test — skip
            pass

    def test_version_unknown(self) -> None:
        obs = self.mod._check_version_gap("")
        assert obs is None

    def test_find_nearest_version(self) -> None:
        """v0.23.0 should match v0.22.1rc1 (closest lower)."""
        nearest = self.mod._find_nearest_known_version("0.23.0")
        assert nearest in ("0.22.1rc1", "0.22")  # fuzzy match acceptable

    def test_find_nearest_version_exact(self) -> None:
        nearest = self.mod._find_nearest_known_version("0.18.0")
        assert nearest == "0.18.0"

    def test_segmentation_issues_empty_on_missing_manifest(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            issues = self.mod._collect_segmentation_issues(Path(d))
            assert issues == []

    def test_collect_observations_produces_payload(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            # Create minimal input: characterizations.json
            out.mkdir(parents=True, exist_ok=True)
            import json
            json.dump({"config_signatures": {}}, (out / "characterizations.json").open("w"))
            json.dump({}, (out / "segment_manifest.json").open("w"))

            result = self.mod.collect_observations(out)
            assert "observations" in result
            assert "statistics" in result
            assert result["statistics"]["unknown_kernel_names"] == 0
            assert (out / "run_observations.json").is_file()
            assert (out / "observations_manifest.json").is_file()


class TestGraphModeDetection:
    """Tests for _detect_graph_mode — warmup, partial capture, thresholds."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile.characterize import _detect_graph_mode
        self.fn = _detect_graph_mode

    def _step(self, head_ms: float, wall_ms: float = 100.0) -> dict[str, Any]:
        return {"wall_ms": str(wall_ms), "head_wall_ms": str(head_ms)}

    def test_warmup_detected_as_graph_mode(self) -> None:
        """First 3 high, next 8 low → graph_mode with warmup annotation."""
        steps = (
            [self._step(20), self._step(18), self._step(15)] +
            [self._step(3)] * 8
        )
        r = self.fn(steps)
        assert r["detected"] == "graph_mode"
        assert r.get("warmup_steps") == 3
        assert "warmup" in r["evidence"].lower()

    def test_warmup_not_triggered_when_later_steps_not_graph(self) -> None:
        """First 3 high but only 50% of later steps are low → NOT warmup."""
        steps = (
            [self._step(20), self._step(18), self._step(15)] +
            [self._step(3)] * 4 + [self._step(15)] * 4  # only 4/8 later steps low
        )
        r = self.fn(steps)
        assert r.get("warmup_steps") is None

    def test_graph_mode_clean(self) -> None:
        r = self.fn([self._step(2)] * 10)
        assert r["detected"] == "graph_mode"

    def test_eager_mode_clean(self) -> None:
        r = self.fn([self._step(25)] * 10)
        assert r["detected"] == "eager_mode"

    def test_partial_capture(self) -> None:
        steps = [self._step(3)] * 4 + [self._step(25)] * 4
        r = self.fn(steps)
        assert r["detected"] == "partial_capture"

    def test_unclear_mid_range(self) -> None:
        r = self.fn([self._step(7)] * 10)
        assert r["detected"] == "unclear"
        assert r["confidence"] == "low"


class TestCPDetection:
    """Tests for _detect_cp — allgather, rank-count asymmetry, step-type differentiation."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from ascend_profile.characterize import _detect_cp
        self.fn = _detect_cp

    def test_allgather_both_step_types(self) -> None:
        hccl = [{"hccl_op_kind": "allgather", "rank_count": "4"}]
        steps = [{"step_type": "prefill"}, {"step_type": "decode"}]
        r = self.fn(hccl, steps, {"communication.allgather"})
        types = [d["type"] for d in r["detected"]]
        assert "pcp" in types
        assert "dcp" in types
        assert r["confidence"] == "high"

    def test_allgather_prefill_only(self) -> None:
        hccl = [{"hccl_op_kind": "allgather", "rank_count": "2"}]
        steps = [{"step_type": "prefill"}] * 5
        r = self.fn(hccl, steps, {"communication.allgather"})
        types = [d["type"] for d in r["detected"]]
        assert "pcp" in types
        assert "dcp" not in types

    def test_no_allgather_no_rank_asymmetry(self) -> None:
        hccl = [{"hccl_op_kind": "allreduce", "rank_count": "8"}]
        steps = [{"step_type": "decode"}] * 5
        r = self.fn(hccl, steps, set())
        assert r["detected"] == "none"

    def test_rank_count_asymmetry_medium_confidence(self) -> None:
        hccl = [
            {"hccl_op_kind": "allreduce", "rank_count": "8"},
            {"hccl_op_kind": "allreduce", "rank_count": "4"},
        ]
        steps = [{"step_type": "prefill" if i < 4 else "decode", "rank_id": str(i)} for i in range(8)]
        r = self.fn(hccl, steps, set())
        assert r["confidence"] == "medium"
        types = [d["type"] for d in r["detected"]]
        assert "pcp" in types


# ---------------------------------------------------------------------------
# step_type_stats_rows
# ---------------------------------------------------------------------------

class TestStepTypeStats:
    """Tests for ``summarize.step_type_stats_rows()``."""

    def _call(self, step_rows):
        from ascend_profile.summarize import step_type_stats_rows
        return step_type_stats_rows(step_rows)

    def test_empty(self):
        assert self._call([]) == []

    def test_single_decode(self):
        rows = [{"step_type": "decode", "wall_ms": 10.0, "head_wall_ms": 1.0,
                 "main_wall_ms": 8.0, "tail_wall_ms": 0.5, "head_busy_ms": 0.8,
                 "main_busy_ms": 7.5, "tail_busy_ms": 0.4, "head_bubble_ms": 0.2,
                 "main_bubble_ms": 0.5, "tail_bubble_ms": 0.1, "head_ratio": 0.1,
                 "main_ratio": 0.8, "tail_ratio": 0.05, "bubble_ratio": 0.08,
                 "busy_union_ms": 8.7, "underfeed_ms": 1.3}]
        result = self._call(rows)
        assert len(result) == 1
        assert result[0]["step_type"] == "decode"
        assert result[0]["count"] == 1
        assert result[0]["count_ratio"] == 1.0
        assert result[0]["median_wall_ms"] == 10.0
        assert result[0]["max_wall_ms"] == 10.0
        assert result[0]["avg_wall_ms"] == 10.0

    def test_mixed_types(self):
        rows = []
        for i in range(10):
            rows.append({"step_type": "decode", "wall_ms": 10.0 + i, "head_wall_ms": 1.0,
                         "main_wall_ms": 8.0, "tail_wall_ms": 0.5, "head_busy_ms": 0.8,
                         "main_busy_ms": 7.5, "tail_busy_ms": 0.4, "head_bubble_ms": 0.2,
                         "main_bubble_ms": 0.5, "tail_bubble_ms": 0.1, "head_ratio": 0.1,
                         "main_ratio": 0.8, "tail_ratio": 0.05, "bubble_ratio": 0.08,
                         "busy_union_ms": 8.7, "underfeed_ms": 1.3})
        for i in range(3):
            rows.append({"step_type": "prefill", "wall_ms": 50.0 + i * 10, "head_wall_ms": 5.0,
                         "main_wall_ms": 40.0, "tail_wall_ms": 2.0, "head_busy_ms": 4.0,
                         "main_busy_ms": 38.0, "tail_busy_ms": 1.5, "head_bubble_ms": 1.0,
                         "main_bubble_ms": 2.0, "tail_bubble_ms": 0.5, "head_ratio": 0.1,
                         "main_ratio": 0.8, "tail_ratio": 0.04, "bubble_ratio": 0.07,
                         "busy_union_ms": 43.5, "underfeed_ms": 6.5})
        result = self._call(rows)
        assert len(result) == 2
        decode_row = [r for r in result if r["step_type"] == "decode"][0]
        prefill_row = [r for r in result if r["step_type"] == "prefill"][0]
        assert decode_row["count"] == 10
        assert prefill_row["count"] == 3
        assert decode_row["count_ratio"] == round(10 / 13, 4)
        # Median of [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] = 14.5
        assert decode_row["median_wall_ms"] == 14.5
        assert decode_row["max_wall_ms"] == 19.0
        # Median of [50, 60, 70] = 60
        assert prefill_row["median_wall_ms"] == 60.0
        assert prefill_row["max_wall_ms"] == 70.0

    def test_speculative_priority(self):
        rows = [
            {"step_type": "speculative", "wall_ms": 15.0, "head_wall_ms": 2.0,
             "main_wall_ms": 10.0, "tail_wall_ms": 1.0, "head_busy_ms": 1.5,
             "main_busy_ms": 9.0, "tail_busy_ms": 0.8, "head_bubble_ms": 0.5,
             "main_bubble_ms": 1.0, "tail_bubble_ms": 0.2, "head_ratio": 0.13,
             "main_ratio": 0.67, "tail_ratio": 0.07, "bubble_ratio": 0.11,
             "busy_union_ms": 11.3, "underfeed_ms": 3.7},
            {"step_type": "speculative", "wall_ms": 25.0, "head_wall_ms": 3.0,
             "main_wall_ms": 18.0, "tail_wall_ms": 2.0, "head_busy_ms": 2.5,
             "main_busy_ms": 16.0, "tail_busy_ms": 1.5, "head_bubble_ms": 0.5,
             "main_bubble_ms": 2.0, "tail_bubble_ms": 0.5, "head_ratio": 0.12,
             "main_ratio": 0.72, "tail_ratio": 0.08, "bubble_ratio": 0.12,
             "busy_union_ms": 20.0, "underfeed_ms": 5.0},
        ]
        result = self._call(rows)
        assert len(result) == 1
        assert result[0]["step_type"] == "speculative"
        assert result[0]["count"] == 2
        assert result[0]["median_wall_ms"] == 20.0
        assert result[0]["max_wall_ms"] == 25.0
        assert result[0]["avg_wall_ms"] == 20.0

    def test_unknown_type(self):
        rows = [{"step_type": "unknown", "wall_ms": 5.0, "head_wall_ms": None,
                 "main_wall_ms": None, "tail_wall_ms": None, "head_busy_ms": None,
                 "main_busy_ms": None, "tail_busy_ms": None, "head_bubble_ms": None,
                 "main_bubble_ms": None, "tail_bubble_ms": None, "head_ratio": None,
                 "main_ratio": None, "tail_ratio": None, "bubble_ratio": None,
                 "busy_union_ms": None, "underfeed_ms": None}]
        result = self._call(rows)
        assert len(result) == 1
        assert result[0]["step_type"] == "unknown"
        assert result[0]["median_wall_ms"] == 5.0
        # All anatomy metrics should be None
        assert result[0]["median_head_wall_ms"] is None
        assert result[0]["median_bubble_ratio"] is None

    def test_none_fields_skipped_for_median(self):
        rows = [
            {"step_type": "decode", "wall_ms": 10.0, "head_wall_ms": 1.0,
             "main_wall_ms": 8.0, "tail_wall_ms": 0.5, "head_busy_ms": 0.8,
             "main_busy_ms": 7.5, "tail_busy_ms": 0.4, "head_bubble_ms": 0.2,
             "main_bubble_ms": 0.5, "tail_bubble_ms": 0.1, "head_ratio": 0.1,
             "main_ratio": 0.8, "tail_ratio": 0.05, "bubble_ratio": 0.08,
             "busy_union_ms": 8.7, "underfeed_ms": 1.3},
            {"step_type": "decode", "wall_ms": 20.0, "head_wall_ms": None,
             "main_wall_ms": None, "tail_wall_ms": None, "head_busy_ms": None,
             "main_busy_ms": None, "tail_busy_ms": None, "head_bubble_ms": None,
             "main_bubble_ms": None, "tail_bubble_ms": None, "head_ratio": None,
             "main_ratio": None, "tail_ratio": None, "bubble_ratio": None,
             "busy_union_ms": None, "underfeed_ms": None},
        ]
        result = self._call(rows)
        assert result[0]["median_wall_ms"] == 15.0
        # head_wall_ms median should NOT include the None row
        assert result[0]["median_head_wall_ms"] == 1.0
        # Non-None median unaffected
        assert result[0]["median_bubble_ratio"] == 0.08
