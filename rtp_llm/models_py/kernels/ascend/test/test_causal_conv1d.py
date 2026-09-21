"""Unit tests for the device-tensor Ascend causal-conv1d wrappers.

These tests target the graph-capture-safe implementation: index math is
plain torch ops (device-agnostic, so they run on CPU in CI) and the FLA-NPU
operators are mocked.  Contract assertions cover the device-tensor call
convention of ``npu_causal_conv1d_fn`` / ``npu_causal_conv1d_update``
(device int32 metadata, ``out=`` static buffer, ``null_block_id`` skip
sentinel), the ``PAD_SLOT_ID = 0`` reserved-page convention, and the
paged-cache semantics (cross-block state migration, multi-block prefill
snapshots).
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

CAUSAL_CONV1D_PATH = Path(__file__).resolve().parents[1] / "causal_conv1d.py"


def _source_tree():
    return ast.parse(
        CAUSAL_CONV1D_PATH.read_text(encoding="utf-8"),
        filename=str(CAUSAL_CONV1D_PATH),
    )


def _register_state_migration() -> None:
    """causal_conv1d 依赖 state_migration（点路径导入），独立加载时需先注册，
    避免触发 rtp_llm 包 __init__（需要编译产物）。"""
    dotted = "rtp_llm.models_py.kernels.ascend.state_migration"
    if dotted in sys.modules:
        return
    parent = ""
    for part in dotted.split(".")[:-1]:
        parent = f"{parent}.{part}" if parent else part
        if parent not in sys.modules:
            stub = types.ModuleType(parent)
            stub.__path__ = []
            sys.modules[parent] = stub
    path = CAUSAL_CONV1D_PATH.parent / "state_migration.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)


class TestCausalConv1dWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_source_tree(), str(CAUSAL_CONV1D_PATH), "exec")

    def test_accelerator_implementations_are_lazy_imports(self):
        eager_imports = [
            node
            for node in _source_tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported_modules = []
        for node in eager_imports:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            else:
                imported_modules.append(node.module or "")

        self.assertFalse(
            any(
                "triton" in module or "fla_npu" in module for module in imported_modules
            ),
            imported_modules,
        )

    def test_public_api_is_present(self):
        public_names = {
            node.name
            for node in _source_tree().body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertTrue(
            {
                "CausalConv1dMetadata",
                "prepare_causal_conv1d_metadata",
                "causal_conv1d_fn",
                "causal_conv1d_update",
            }.issubset(public_names)
        )

    def test_host_side_helpers_are_gone(self):
        # Regression guard: the host-side (.cpu()/.tolist()) helpers must not
        # come back — they trigger aclrtSynchronizeStream inside ACL graph
        # capture.  (The prefill-only _scatter_prefill_states_host is the
        # documented exception and stays.)
        names = {
            node.name
            for node in _source_tree().body
            if isinstance(node, ast.FunctionDef)
        }
        for legacy in (
            "_as_int_list",
            "_as_int_rows",
            "_mapped_page",
            "_load_npu_causal_conv1d",
        ):
            self.assertNotIn(legacy, names, f"host-side helper {legacy} reintroduced")
        for required in (
            "_gather_pages_from_block_map",
            "_cross_page_copy_device",
            "_load_fla_npu_fn",
            "_load_fla_npu_update",
        ):
            self.assertIn(required, names)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestCausalConv1dDeviceHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_state_migration()
        module_name = "ascendc_causal_conv1d_test"
        spec = importlib.util.spec_from_file_location(module_name, CAUSAL_CONV1D_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = cls.module
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("ascendc_causal_conv1d_test", None)

    def test_pad_slot_id_is_reserved_page_zero(self):
        # fla_npu's null_block_id must be non-negative; page 0 is the
        # engine-reserved block (real block IDs start at 1), so 0 is the only
        # sentinel that can never collide with a live page.
        self.assertEqual(self.module.PAD_SLOT_ID, 0)

    def test_normalize_activation_for_fla(self):
        normalize = self.module._normalize_activation_for_fla
        self.assertIsNone(normalize(None))
        self.assertIsNone(normalize(False))
        self.assertEqual(normalize(True), "silu")
        self.assertEqual(normalize("silu"), "silu")
        self.assertEqual(normalize("swish"), "swish")
        with self.assertRaises(ValueError):
            normalize("gelu")

    def test_gather_pages_from_block_map_masks_out_of_range(self):
        block_map = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)

        # Out-of-range block positions fall back to the pad sentinel.
        pages = self.module._gather_pages_from_block_map(
            block_map, torch.tensor([1, 5], dtype=torch.int64), 0
        )
        self.assertEqual(pages.tolist(), [11, 0])
        self.assertEqual(pages.dtype, torch.int32)

        # Negative positions (no block allocated yet) mask to the pad too;
        # the valid row (batch 1, column 0) still resolves normally.
        neg = self.module._gather_pages_from_block_map(
            block_map, torch.tensor([-3, 0], dtype=torch.int64), 0
        )
        self.assertEqual(neg.tolist(), [0, 12])

    def test_gather_pages_from_block_map_strips_group_axis(self):
        block_map = torch.tensor(
            [[[5, 6], [7, 8]], [[50, 60], [70, 80]]], dtype=torch.int32
        )
        pages = self.module._gather_pages_from_block_map(
            block_map, torch.tensor([0, 1], dtype=torch.int64), 0
        )
        # Group 0 (the LINEAR cache group) is selected.
        self.assertEqual(pages.tolist(), [5, 8])

    def test_cross_page_copy_device_migrates_only_crossing_rows(self):
        conv_state = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        before = conv_state.clone()
        # Row 0: read page 0 (pad) -> no migration, write page 1 self-copies.
        # Row 1: read == write (2) -> self copy.
        # Row 2: read page 1, write page 3 -> migration.
        write_page = torch.tensor([1, 2, 3], dtype=torch.int32)
        read_page = torch.tensor([0, 2, 1], dtype=torch.int32)

        self.module._cross_page_copy_device(conv_state, write_page, read_page, 0)

        torch.testing.assert_close(conv_state[3], before[1])  # migrated row
        torch.testing.assert_close(conv_state[1], before[1])  # pad-read self copy
        torch.testing.assert_close(conv_state[2], before[2])  # read == write
        torch.testing.assert_close(conv_state[0], before[0])  # untouched


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestCausalConv1dNpuAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_state_migration()
        module_name = "ascendc_causal_conv1d_test"
        spec = importlib.util.spec_from_file_location(module_name, CAUSAL_CONV1D_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = cls.module
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("ascendc_causal_conv1d_test", None)

    def test_prefill_passes_device_metadata_and_scatters_snapshots(self):
        # GPU-facing layout is (D, T); the FLA-NPU operator receives (T, D).
        x = torch.tensor(
            [[10, 20, 30, 40, 50], [11, 21, 31, 41, 51]],
            dtype=torch.float16,
        )
        weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        conv_states = torch.zeros(6, 2, 3, dtype=torch.float32)
        conv_states[3] = torch.tensor(
            [[100, 101, 102], [200, 201, 202]], dtype=torch.float32
        )
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
        prefix_lengths = torch.tensor([2], dtype=torch.int32)
        block_map = torch.tensor([[3, 4]], dtype=torch.int32)
        calls = []

        def fake_npu_causal_conv1d_fn(**kwargs):
            calls.append(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items()
                }
            )
            return kwargs["x"]

        with mock.patch.object(
            self.module,
            "_load_fla_npu_fn",
            return_value=fake_npu_causal_conv1d_fn,
        ):
            output = self.module.causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=None,
                conv_states=conv_states,
                query_start_loc=query_start_loc,
                block_map=block_map,
                prefix_lengths=prefix_lengths,
                seq_size_per_block=4,
                activation="swish",
            )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        # FLA-NPU layouts: x (tokens, dim); weight (width, dim); initial
        # states (batch, state_len, dim) gathered from the prefix's page.
        self.assertEqual(tuple(call["x"].shape), (5, 2))
        self.assertEqual(tuple(call["weight"].shape), (4, 2))
        self.assertEqual(tuple(call["conv_states"].shape), (1, 3, 2))
        torch.testing.assert_close(
            call["conv_states"][0],
            torch.tensor([[100, 200], [101, 201], [102, 202]], dtype=torch.float32),
        )
        # Metadata must be device int32 tensors — never host lists.
        self.assertIsInstance(call["query_start_loc"], torch.Tensor)
        self.assertEqual(call["query_start_loc"].dtype, torch.int32)
        self.assertEqual(call["query_start_loc"].tolist(), [0, 5])
        self.assertIsInstance(call["has_initial_state"], torch.Tensor)
        self.assertEqual(call["has_initial_state"].dtype, torch.int32)
        self.assertEqual(call["has_initial_state"].tolist(), [1])
        self.assertEqual(call["activation"], "swish")
        self.assertEqual(call["pad_slot_id"], 0)
        self.assertFalse(call["validate_data"])
        # Output contract: (dim, tokens) in the original dtype.
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        torch.testing.assert_close(output, x)

        # prefix=2, block-size=4: token 2 closes page 3; token 5 ends page 4.
        torch.testing.assert_close(
            conv_states[3],
            torch.tensor([[102, 10, 20], [202, 11, 21]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            conv_states[4],
            torch.tensor([[30, 40, 50], [31, 41, 51]], dtype=torch.float32),
        )

    def test_decode_is_tokenwise_with_device_indices_and_out_buffer(self):
        x = torch.tensor([[[20, 21], [30, 31]]], dtype=torch.float16)
        weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        conv_state = torch.zeros(6, 2, 3, dtype=torch.float32)
        conv_state[2] = torch.tensor([[1, 2, 3], [11, 12, 13]], dtype=torch.float32)
        block_map = torch.tensor([[2, 5]], dtype=torch.int32)
        sequence_lengths = torch.tensor([2], dtype=torch.int32)
        calls = []

        def fake_npu_causal_conv1d_update(**kwargs):
            calls.append(
                {
                    "x_shape": tuple(kwargs["x"].shape),
                    "weight_shape": tuple(kwargs["weight"].shape),
                    "state_shape": tuple(kwargs["conv_state"].shape),
                    "indices_raw": kwargs["conv_state_indices"],
                    "conv_state_indices": kwargs["conv_state_indices"].tolist(),
                    "null_block_id": kwargs["null_block_id"],
                    "activation": kwargs["activation"],
                    "validate_data": kwargs["validate_data"],
                    "out_shape": tuple(kwargs["out"].shape),
                    "out_is_tensor": isinstance(kwargs["out"], torch.Tensor),
                }
            )
            states = kwargs["conv_state"]
            for batch_index, page_index in enumerate(
                kwargs["conv_state_indices"].tolist()
            ):
                if page_index == kwargs["null_block_id"]:
                    continue
                previous = states[page_index].clone()
                states[page_index, :-1].copy_(previous[1:])
                states[page_index, -1].copy_(kwargs["x"][batch_index, 0])
            kwargs["out"].copy_(kwargs["x"] + 100)

        with mock.patch.object(
            self.module,
            "_load_fla_npu_update",
            return_value=fake_npu_causal_conv1d_update,
        ):
            output = self.module.causal_conv1d_update(
                x=x,
                conv_state=conv_state,
                weight=weight,
                activation=True,
                block_map=block_map,
                seq_size_per_block=4,
                sequence_lengths=sequence_lengths,
            )

        # One op call per speculative token: RTP snapshots every speculative
        # token in its own consecutive block-map entry, so token 2 uses page 2
        # and token 3 uses page 5.
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call["conv_state_indices"] for call in calls], [[2], [5]]
        )
        for call in calls:
            # Indices are device int32 tensors — never host lists.
            self.assertIsInstance(call["indices_raw"], torch.Tensor)
            self.assertEqual(call["indices_raw"].dtype, torch.int32)
            # FLA layouts: x (batch, 1, dim); weight (width, dim); conv_state
            # is the transposed (pages, state, dim) view of the RTP cache.
            self.assertEqual(call["x_shape"], (1, 1, 2))
            self.assertEqual(call["weight_shape"], (4, 2))
            self.assertEqual(call["state_shape"], (6, 3, 2))
            # out= static buffer slice, required for graph replay safety.
            self.assertTrue(call["out_is_tensor"])
            self.assertEqual(call["out_shape"], (1, 1, 2))
            # Skip sentinel must be the reserved page 0 (non-negative).
            self.assertEqual(call["null_block_id"], 0)
            # RTP boolean activation alias normalized to the FLA string form.
            self.assertEqual(call["activation"], "silu")
            # Never request the ctypes data-validation path (it D2H-syncs).
            self.assertFalse(call["validate_data"])

        # Output assembled from the out buffer: (batch, dim, tokens), original
        # dtype, values written by the fake op via out.copy_.
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertEqual(output.dtype, x.dtype)
        torch.testing.assert_close(output, x + 100)

        # Cross-page migration: before the second op call, page 5 was seeded
        # from the *already updated* page 2 (RTP (dim, state) layout).
        torch.testing.assert_close(
            conv_state[2],
            torch.tensor([[2, 3, 20], [12, 13, 30]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            conv_state[5],
            torch.tensor([[3, 20, 21], [13, 30, 31]], dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
