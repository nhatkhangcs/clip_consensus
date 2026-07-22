import unittest

import torch

from caption_geometry.scripts.common import sample_caption_subset
from caption_geometry.scripts.common import corrupt_support_captions, sample_support_captions


class ScriptCommonTest(unittest.TestCase):
    def test_sample_caption_subset_is_deterministic(self):
        captions = torch.arange(2 * 5 * 3).reshape(2, 5, 3)

        left = sample_caption_subset(captions, support_count=3, seed=7)
        right = sample_caption_subset(captions, support_count=3, seed=7)

        self.assertEqual(tuple(left.shape), (2, 3, 3))
        self.assertTrue(torch.equal(left, right))

    def test_sample_support_captions_returns_source_indices(self):
        captions = torch.arange(2 * 5 * 3).reshape(2, 5, 3)

        support, source_indices = sample_support_captions(captions, support_count=3, seed=7)

        self.assertEqual(tuple(support.shape), (2, 3, 3))
        self.assertEqual(tuple(source_indices.shape), (2, 3))

    def test_corrupt_support_captions_marks_replacements(self):
        captions = torch.arange(4 * 5 * 3).float().reshape(4, 5, 3)
        support, source_indices = sample_support_captions(captions, support_count=None, seed=0)

        corrupted, corrupted_sources, replaced = corrupt_support_captions(
            support_captions=support,
            support_source_indices=source_indices,
            caption_pool=captions,
            mismatch_rate=0.5,
            seed=11,
        )

        self.assertEqual(replaced, 10)
        self.assertEqual(int((corrupted_sources == -1).sum().item()), 10)
        self.assertFalse(torch.equal(corrupted, support))


if __name__ == "__main__":
    unittest.main()
