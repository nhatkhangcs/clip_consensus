import unittest

import torch

from caption_geometry.scoring.bootstrap import bootstrap_mean_ci
from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scripts.common import corrupt_support_captions
from caption_geometry.scripts.protocols import evaluate_protocol, protocol_splits


class ProtocolSplitsTest(unittest.TestCase):
    def test_fixed_folds_hold_out_one_query_disjoint_from_supports(self):
        captions = torch.arange(3 * 5 * 2).float().reshape(3, 5, 2)

        splits = list(protocol_splits(captions, "fixed"))

        self.assertEqual(len(splits), 5)

        for fold, split in enumerate(splits):
            self.assertEqual(tuple(split.query_captions.shape), (3, 1, 2))
            self.assertEqual(tuple(split.support_captions.shape), (3, 4, 2))
            self.assertFalse(split.leave_one_out)
            self.assertTrue(torch.equal(split.query_captions[:, 0], captions[:, fold]))
            query_rows = split.query_captions.reshape(-1, 2)
            support_rows = split.support_captions.reshape(-1, 2)
            self.assertFalse((query_rows[:, None] == support_rows[None]).all(-1).any())

    def test_loo_is_a_single_split(self):
        captions = torch.randn(3, 5, 2)

        splits = list(protocol_splits(captions, "loo"))

        self.assertEqual(len(splits), 1)
        self.assertTrue(splits[0].leave_one_out)

    def test_fixed_protocol_outcomes_cover_every_fold_with_image_clusters(self):
        images = normalize(torch.randn(4, 6))
        captions = normalize(torch.randn(4, 5, 6))

        result = evaluate_protocol("full", images, captions, "fixed", alpha=1.0, beta=0.4, gamma=1.8)

        self.assertEqual(result.outcomes["i2t"].numel(), 4 * 5)
        self.assertEqual(result.outcomes["t2i"].numel(), 4 * 5)
        self.assertTrue(torch.equal(result.clusters["t2i"], torch.arange(4).repeat(5)))

    def test_fixed_base_ignores_supports(self):
        images = normalize(torch.randn(4, 6))
        captions = normalize(torch.randn(4, 5, 6))

        result = evaluate_protocol("base", images, captions, "fixed", alpha=1.0, beta=0.0, gamma=0.0)
        hits = (captions @ images.T).argmax(dim=-1) == torch.arange(4)[:, None]

        self.assertAlmostEqual(result.metrics["t2i"]["R@1"], hits.float().mean().item() * 100.0, places=4)


class ClusteredBootstrapTest(unittest.TestCase):
    def test_constant_within_cluster_widens_interval(self):
        # 50 clusters of 10 identical outcomes: effective sample size is 50, not 500.
        per_cluster = (torch.rand(50, generator=torch.Generator().manual_seed(0)) > 0.5).float()
        values = per_cluster.repeat_interleave(10)
        clusters = torch.arange(50).repeat_interleave(10)

        naive = bootstrap_mean_ci(values, samples=2000, seed=0)
        clustered = bootstrap_mean_ci(values, samples=2000, seed=0, clusters=clusters)

        self.assertAlmostEqual(naive["estimate"], clustered["estimate"])
        self.assertGreater(clustered["high"] - clustered["low"], 2.0 * (naive["high"] - naive["low"]))


class ExternalPoolTest(unittest.TestCase):
    def test_external_pool_draws_from_pool_only(self):
        support = torch.zeros(4, 3, 2)
        pool = torch.ones(2, 5, 2)

        corrupted, _, replaced = corrupt_support_captions(
            support_captions=support,
            support_source_indices=torch.zeros(4, 3, dtype=torch.long),
            caption_pool=pool,
            mismatch_rate=0.5,
            seed=0,
            exclude_same_image=False,
        )

        self.assertEqual(replaced, 6)
        self.assertEqual(int((corrupted == 1).all(-1).sum().item()), 6)


if __name__ == "__main__":
    unittest.main()
