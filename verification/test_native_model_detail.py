"""Full-grid learned model arithmetic stays global and decoder seams stay absent."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ipde import learned_depth as learned
from ipde.display_training import _DisplayPool
from ipde.training import TrainingOptions, _SamplePool


class NativeModelDetailTests(unittest.TestCase):
    def attention(self):
        torch.manual_seed(11)
        return SimpleNamespace(num_heads=3, qkv=torch.nn.Linear(12, 36),
            q_norm=torch.nn.LayerNorm(4), k_norm=torch.nn.LayerNorm(4), rope=None,
            proj=torch.nn.Linear(12, 12), proj_drop=torch.nn.Identity())

    def test_query_slices_retain_every_key_and_match_full_masked_attention(self):
        attention = self.attention()
        x = torch.randn(2, 37, 12)
        mask = torch.rand(2, 37, 37) > 0.4
        q, k, v = attention.qkv(x).reshape(2, 37, 3, 3, 4).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attention.q_norm(q), attention.k_norm(k)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        expected = attention.proj(expected.transpose(1, 2).reshape(2, 37, 12))
        calls = []
        original = torch.nn.functional.scaled_dot_product_attention
        def record(q, k, v, **options):
            calls.append((q.shape[-2], k.shape[-2], v.shape[-2]))
            return original(q, k, v, **options)
        with torch.inference_mode(), patch.object(learned, "DA3_ATTENTION_SCORE_BYTES", 2 * 3 * 37 * 4 * 5), \
                patch("torch.nn.functional.scaled_dot_product_attention", side_effect=record):
            actual = learned._da3_attention_forward(attention, x, attn_mask=mask)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(rows <= 5 and keys == 37 and values == 37 for rows, keys, values in calls))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_bilinear_stripes_keep_full_grid_sampling_coordinates(self):
        torch.manual_seed(8)
        source = torch.rand(1, 8, 9, 13)
        for shape in ((33, 41), (6, 7), (1, 1)):
            expected = torch.nn.functional.interpolate(source, shape, mode="bilinear", align_corners=True)
            actual = torch.cat([learned._bilinear_rows(source, start, min(start + 5, shape[0]), shape)
                                for start in range(0, shape[0], 5)], dim=2)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_decoder_halos_match_full_convolution_at_every_stripe_boundary(self):
        torch.manual_seed(12)
        source = torch.rand(1, 8, 9, 13)
        decoder = torch.nn.Sequential(torch.nn.Conv2d(8, 5, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(5, 2, 1))
        head = SimpleNamespace(pos_embed=False, scratch=SimpleNamespace(output_conv2=decoder))
        helper = SimpleNamespace(position_grid_to_embed=None)
        expected = decoder(torch.nn.functional.interpolate(source, (33, 41), mode="bilinear", align_corners=True))
        with torch.inference_mode(), patch.dict(sys.modules, {"depth_anything_3.model.utils.head_utils": helper}), \
                patch.object(learned, "DA3_DECODER_ROWS", 5):
            actual = learned._da3_primary_logits_rows(head, source, (33, 41), 41 / 33)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_attention_adaptation_is_local_to_one_model_instance(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                fixture = NativeModelDetailTests().attention()
                for name, value in vars(fixture).items():
                    setattr(self, name, value)
            def forward(self, x, pos=None, attn_mask=None):
                return learned._da3_attention_forward(self, x, pos, attn_mask)
        Attention.__module__ = "depth_anything_3.model.dinov2.layers.attention"
        original = Attention.forward
        first, second = Attention(), Attention()
        self.assertEqual(learned._install_bounded_attention(first, da3=True), 1)
        self.assertIs(Attention.forward, original)
        self.assertIs(second.forward.__func__, original)
        self.assertIsNot(first.forward.__func__, original)

    def test_positional_embeddings_use_global_coordinates_across_stripes(self):
        source = torch.rand(1, 8, 9, 13)
        decoder = torch.nn.Sequential(torch.nn.Conv2d(8, 5, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(5, 2, 1))
        head = SimpleNamespace(pos_embed=True, scratch=SimpleNamespace(output_conv2=decoder))
        def embedding(uv, channels):
            return uv.repeat(1, 1, channels // 2).sin()
        helper = SimpleNamespace(position_grid_to_embed=embedding)
        ratio = 41 / 33
        diag = (ratio**2 + 1)**0.5
        yy, xx = torch.meshgrid(torch.linspace(-1 / diag * 32 / 33, 1 / diag * 32 / 33, 33),
            torch.linspace(-ratio / diag * 40 / 41, ratio / diag * 40 / 41, 41), indexing="ij")
        positional = embedding(torch.stack((xx, yy), -1), 8).permute(2, 0, 1)[None] * 0.1
        expected = decoder(torch.nn.functional.interpolate(source, (33, 41), mode="bilinear", align_corners=True) + positional)
        with torch.inference_mode(), patch.dict(sys.modules, {"depth_anything_3.model.utils.head_utils": helper}), \
                patch.object(learned, "DA3_DECODER_ROWS", 5):
            actual = learned._da3_primary_logits_rows(head, source, (33, 41), ratio)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


class PrefetchReuseTests(unittest.TestCase):
    def test_actual_training_and_validation_consume_ready_futures_once(self):
        from test_training_lifecycle import TrainingLifecycleTests, _mark_native
        from test_training_progress import _dataset
        import json
        import ipde.training as training
        fixture = TrainingLifecycleTests()
        options = TrainingOptions(patch_size=64, iterations=1, device="cpu", learning_rate=1e-3,
            epochs=1, checkpoint_schedule="end", prefetch_samples=2, workers=2, cache_samples=64)
        original_read, original_prefetch = training._load_sample, training._SamplePool.prefetch
        reads = []
        def read(root, record, settings):
            reads.append(record["id"])
            return original_read(root, record, settings)
        def ready(pool, indices):
            original_prefetch(pool, indices)
            for future in pool.pending.values():
                future.result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            _mark_native(manifest, dataset)
            manifest_path.write_text(json.dumps(manifest))
            with patch("ipde.training._load_sample", side_effect=read), \
                    patch.object(training._SamplePool, "prefetch", ready):
                fixture._run(root, dataset, root / "model.pth", options)
            self.assertEqual(sorted(reads), [f"sample-{index}" for index in range(6)])

    def test_ready_training_future_is_consumed_once_before_next_prefetch(self):
        records = [{"id": str(index), "rgb": {"shape": [2, 2, 3]}} for index in range(4)]
        pool = _SamplePool(Path("unused"), records, TrainingOptions(prefetch_samples=2, cache_samples=2, workers=2))
        reads = []
        def read(root, record, options):
            reads.append(record["id"])
            return {"details": {"sample_id": record["id"]}, "identity": record["id"]}
        try:
            with patch("ipde.training._load_sample", side_effect=read):
                pool.prefetch(range(4))
                for future in pool.pending.values():
                    future.result()
                for index in range(4):
                    self.assertEqual(pool[index]["identity"], str(index))
                    pool.prefetch(range(index + 1, 4))
                    for future in pool.pending.values():
                        future.result()
            self.assertEqual(reads, ["0", "1", "2", "3"])
        finally:
            pool.close()

    def test_prefetch_skips_cached_leading_images_without_reducing_queue(self):
        records = [{"id": str(index), "rgb": {"shape": [2, 2, 3]}} for index in range(5)]
        pool = _SamplePool(Path("unused"), records, TrainingOptions(prefetch_samples=2, workers=2))
        pool.cache[0], pool.cache[1] = {}, {}
        try:
            with patch("ipde.training._load_sample", return_value={"details": {}}):
                pool.prefetch(range(5))
                self.assertEqual(set(pool.pending), {2, 3})
        finally:
            pool.close()

    def test_display_prefetch_replaces_stale_epoch_order_and_respects_bound(self):
        records = [{"rgb": {"shape": [2, 2, 3], "dtype": "|u1"},
            "right_rgb": {"shape": [2, 2, 3], "dtype": "|u1"},
            "teacher": {"target": {"shape": [4, 4]}}} for _ in range(5)]
        pool = _DisplayPool(Path("unused"), records, TrainingOptions(prefetch_samples=2, workers=2))
        try:
            with patch.object(pool, "_read", side_effect=lambda index: {"index": index}):
                pool.prefetch([0, 1])
                for future in pool.pending.values():
                    future.result()
                pool.prefetch([3, 4])
                self.assertEqual(set(pool.pending), {3, 4})
                self.assertEqual(pool.get(3), {"index": 3})
                pool.prefetch([3, 4, 2])
                self.assertEqual(set(pool.pending), {2, 4})
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
