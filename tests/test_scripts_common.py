import unittest

import torch

from caption_geometry.scripts.common import sample_caption_subset


class ScriptCommonTest(unittest.TestCase):
    def test_sample_caption_subset_is_deterministic(self):
        captions = torch.arange(2 * 5 * 3).reshape(2, 5, 3)

        left = sample_caption_subset(captions, support_count=3, seed=7)
        right = sample_caption_subset(captions, support_count=3, seed=7)

        self.assertEqual(tuple(left.shape), (2, 3, 3))
        self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()

