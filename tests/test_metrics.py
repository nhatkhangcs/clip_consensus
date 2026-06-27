import unittest

import torch

from caption_geometry.scoring.metrics import recall_at_k_i2t, recall_at_k_t2i


class RetrievalMetricsTest(unittest.TestCase):
    def test_perfect_t2i_retrieval(self):
        scores = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 9.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 9.0],
            ]
        )

        metrics = recall_at_k_t2i(scores, k_captions=2)

        self.assertEqual(metrics["R@1"], 100.0)
        self.assertEqual(metrics["MR"], 1.0)

    def test_perfect_i2t_retrieval(self):
        scores = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 9.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 9.0],
            ]
        )

        metrics = recall_at_k_i2t(scores, k_captions=2)

        self.assertEqual(metrics["R@1"], 100.0)
        self.assertEqual(metrics["MR"], 1.0)


if __name__ == "__main__":
    unittest.main()

