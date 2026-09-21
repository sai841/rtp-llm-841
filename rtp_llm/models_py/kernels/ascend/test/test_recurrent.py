"""Unit tests for the device-tensor Ascend recurrent GDN wrapper.

These tests target the graph-capture-safe implementation: page resolution and
state seeding are plain torch ops (device-agnostic, so they run on CPU in CI)
and the FLA-NPU operator is mocked.  ``recurrent.py`` imports
``linear_attention`` at module level, so the loader registers that module
under its full dotted name without triggering the ``rtp_llm`` package
``__init__`` (which needs compiled extensions).
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "recurrent.py"
LINEAR_ATTENTION_PATH = Path(__file__).resolve().parents[1] / "linear_attention.py"


def _tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))


class TestRecurrentWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_tree(), str(MODULE_PATH), "exec")

    def test_accelerator_dependencies_are_lazy(self):
        imports = [
            node
            for node in _tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for node in imports:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            else:
                names.append(node.module or "")
        self.assertFalse(any("triton" in name or "fla_npu" in name for name in names))

    def test_host_side_helpers_are_gone(self):
        # Regression guard: the host-side (.cpu()/.tolist()) page-resolution
        # helpers must not come back — they trigger aclrtSynchronizeStream
        # inside ACL graph capture.
        names = {
            node.name
            for node in _tree().body
            if isinstance(node, ast.FunctionDef)
        }
        for legacy in ("_to_int_list", "_resolve_state_pages", "_seed_first_write_pages"):
            self.assertNotIn(legacy, names, f"host-side helper {legacy} reintroduced")
        for required in (
            "_resolve_state_pages_device",
            "_seed_first_write_pages_device",
            "fused_recurrent_gated_delta_rule",
        ):
            self.assertIn(required, names)


try:
    import torch
except ImportError:
    torch = None


def _register_kernel_module(name: str) -> None:
    """按完整点路径注册 kernels/ascend 下的模块（父包空壳占位）。"""
    dotted = f"rtp_llm.models_py.kernels.ascend.{name}"
    if dotted in sys.modules:
        return
    parent = ""
    for part in dotted.split(".")[:-1]:
        parent = f"{parent}.{part}" if parent else part
        if parent not in sys.modules:
            stub = types.ModuleType(parent)
            stub.__path__ = []
            sys.modules[parent] = stub
    spec = importlib.util.spec_from_file_location(dotted, MODULE_PATH.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)


def _load_recurrent_module():
    """Load recurrent.py standalone with kernel deps pre-registered."""

    for name in ("linear_attention", "state_migration"):
        _register_kernel_module(name)

    module_name = "ascendc_recurrent_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestResolveStatePagesDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_recurrent_module()

    def test_tracks_cross_block_tokens(self):
        # Mirror of the original host-logic test: length 3, block size 2,
        # three speculative tokens read page 10 and write pages 11/12/13.
        block_map = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        read_pages, write_pages = self.module._resolve_state_pages_device(
            block_map,
            torch.tensor([3], dtype=torch.int32),
            batch=1,
            token_count=3,
            seq_size_per_block=2,
        )
        self.assertEqual(read_pages.tolist(), [10])
        self.assertEqual(write_pages.tolist(), [11, 12, 13])
        self.assertEqual(read_pages.dtype, torch.int32)
        self.assertEqual(write_pages.dtype, torch.int32)

    def test_first_token_has_no_read_page(self):
        # first_length == 1 -> no committed state yet -> read masked to the
        # sentinel (-1), write still resolves normally.
        block_map = torch.tensor([[4, 5]], dtype=torch.int32)
        read_pages, write_pages = self.module._resolve_state_pages_device(
            block_map,
            torch.tensor([1], dtype=torch.int32),
            batch=1,
            token_count=1,
            seq_size_per_block=4,
        )
        self.assertEqual(read_pages.tolist(), [-1])
        self.assertEqual(write_pages.tolist(), [4])

    def test_out_of_range_positions_are_masked(self):
        block_map = torch.tensor([[7, 8]], dtype=torch.int32)
        # length 12, block size 2 -> read position 5 and write positions 5, 6
        # all exceed the 2-column block map.
        read_pages, write_pages = self.module._resolve_state_pages_device(
            block_map,
            torch.tensor([12], dtype=torch.int32),
            batch=1,
            token_count=2,
            seq_size_per_block=2,
        )
        self.assertEqual(read_pages.tolist(), [-1])
        self.assertEqual(write_pages.tolist(), [-1, -1])

    def test_three_dim_block_map_uses_group_zero(self):
        block_map = torch.tensor(
            [[[5, 6], [7, 8]], [[50, 60], [70, 80]]], dtype=torch.int32
        )
        read_pages, write_pages = self.module._resolve_state_pages_device(
            block_map,
            torch.tensor([3, 5], dtype=torch.int32),
            batch=2,
            token_count=1,
            seq_size_per_block=4,
        )
        # b0: len 3 -> read block 0 (page 5), write block 0 (page 5)
        # b1: len 5 -> read block 0 (page 7), write block 1 (page 8)
        self.assertEqual(read_pages.tolist(), [5, 7])
        self.assertEqual(write_pages.tolist(), [5, 8])

    def test_degenerate_block_map_falls_back_to_synthetic_pages(self):
        read_pages, write_pages = self.module._resolve_state_pages_device(
            None,
            None,
            batch=2,
            token_count=2,
            seq_size_per_block=1,
        )
        self.assertEqual(read_pages.tolist(), [0, 1])
        self.assertEqual(write_pages.tolist(), [0, 1, 2, 3])


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestSeedFirstWritePagesDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_recurrent_module()

    def test_migrates_only_crossing_rows(self):
        state = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
        before = state.clone()
        read_pages = torch.tensor([1, 2], dtype=torch.int32)
        # T=2: flat write pages [b0t0, b0t1, b1t0, b1t1]; first writes are
        # page 3 (differs from read page 1 -> migrate) and page 2 (equal to
        # read page 2 -> self copy).
        write_pages = torch.tensor([3, 4, 2, 2], dtype=torch.int32)

        self.module._seed_first_write_pages_device(state, read_pages, write_pages)

        torch.testing.assert_close(state[3], before[1])  # crossing row migrated
        torch.testing.assert_close(state[1], before[1])  # source untouched
        torch.testing.assert_close(state[2], before[2])  # read == write
        torch.testing.assert_close(state[4], before[4])  # non-first page untouched
        torch.testing.assert_close(state[0], before[0])

    def test_sentinel_rows_self_copy_onto_reserved_page(self):
        state = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
        before = state.clone()
        # read == -1 (sentinel, e.g. first token with no state yet): no
        # migration anywhere, the write page self-copies.
        read_pages = torch.tensor([-1], dtype=torch.int32)
        write_pages = torch.tensor([2, 2], dtype=torch.int32)

        self.module._seed_first_write_pages_device(state, read_pages, write_pages)

        torch.testing.assert_close(state, before)


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestFusedRecurrentGatedDeltaRulePlumbing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_recurrent_module()

    def test_passes_device_metadata_to_operator(self):
        torch.manual_seed(0)
        B, T, HK, HV, DK, DV = 2, 2, 2, 2, 2, 2
        PAGES = 8
        q = torch.randn(B, T, HK, DK, dtype=torch.bfloat16)
        k = torch.randn(B, T, HK, DK, dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, DV, dtype=torch.bfloat16)
        g = -torch.rand(B, T, HV, dtype=torch.float32)
        beta = torch.rand(B, T, HV, dtype=torch.bfloat16)
        state = torch.randn(PAGES, HV, DV, DK, dtype=torch.float32)
        block_map = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
        sequence_lengths = torch.tensor([5, 3], dtype=torch.int32)
        state_before = state.clone()
        captured = {}

        def _fake_op(q_, k_, v_, state_, beta=None, scale=None,
                     actual_seq_lengths=None, ssm_state_indices=None, g=None):
            captured.update(
                q=q_, k=k_, v=v_, state=state_, beta=beta, scale=scale,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=ssm_state_indices, g=g,
            )
            return torch.ones(B * T, HV, DV, dtype=torch.bfloat16)

        fake_ascendc = types.SimpleNamespace(
            npu_recurrent_gated_delta_rule=_fake_op
        )
        with mock.patch.object(
            self.module, "_get_ascendc_ops", return_value=fake_ascendc
        ):
            out, returned_state = self.module.fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, scale=None,
                initial_state=state, inplace_final_state=True,
                block_map=block_map, seq_size_per_block=2,
                sequence_lengths=sequence_lengths,
                use_qk_l2norm_in_kernel=False,
            )

        # ssm_state_indices: flat [b0t0, b0t1, b1t0, b1t1] layout as device
        # int32 tensors — never host lists.
        # b0: len 5, sspb 2 -> write blocks 2,3 -> pages 3,4
        # b1: len 3, sspb 2 -> write blocks 1,2 -> pages 6,7
        self.assertIsInstance(captured["ssm_state_indices"], torch.Tensor)
        self.assertEqual(captured["ssm_state_indices"].dtype, torch.int32)
        self.assertEqual(captured["ssm_state_indices"].tolist(), [3, 4, 6, 7])
        # actual_seq_lengths: [0] placeholder + token count per batch.
        self.assertIsInstance(captured["actual_seq_lengths"], torch.Tensor)
        self.assertEqual(captured["actual_seq_lengths"].dtype, torch.int32)
        self.assertEqual(captured["actual_seq_lengths"].tolist(), [0, 2, 2])
        # q/k/v/beta are flattened to (B*T, ...) in bfloat16; g in float32.
        self.assertEqual(tuple(captured["q"].shape), (B * T, HK, DK))
        self.assertEqual(captured["q"].dtype, torch.bfloat16)
        self.assertEqual(captured["k"].dtype, torch.bfloat16)
        self.assertEqual(captured["v"].dtype, torch.bfloat16)
        self.assertEqual(captured["beta"].dtype, torch.bfloat16)
        self.assertEqual(captured["g"].dtype, torch.float32)
        # Default scale is k_dim ** -0.5.
        self.assertAlmostEqual(captured["scale"], DK ** -0.5)
        # State is passed in place and returned unchanged by the fake op.
        self.assertIs(returned_state, state)
        # Output reshaped to (B, T, HV, DV) in the query dtype.
        self.assertEqual(tuple(out.shape), (B, T, HV, DV))
        self.assertEqual(out.dtype, q.dtype)
        torch.testing.assert_close(
            out, torch.ones(B, T, HV, DV, dtype=torch.bfloat16)
        )
        # Cross-page seeding happened before the launch: the first write page
        # of each sequence holds the state of its read page.
        # b0: read page 2, first write page 3 -> migrated.
        torch.testing.assert_close(state[3], state_before[2])
        # b1: read page 5, first write page 6 -> migrated.
        torch.testing.assert_close(state[6], state_before[5])
        # The read pages themselves stay untouched.
        torch.testing.assert_close(state[2], state_before[2])
        torch.testing.assert_close(state[5], state_before[5])


if __name__ == "__main__":
    unittest.main()
