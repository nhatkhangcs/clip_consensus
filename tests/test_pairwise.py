import unittest

import torch

from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.pairwise import (
    candidate_features,
    score_image_text_pairs,
    score_text_against_images,
)


class PairwiseScoringTest(unittest.TestCase):
    def test_pair_scores_match_score_matrix_diagonal(self):
        images = normalize(torch.randn(2, 4))
        captions = normalize(torch.randn(2, 3, 4))
        image_features, consensus, residual = candidate_features(images, captions)
        text = captions[:, 0]
        indices = torch.tensor([0, 1])

        pair_scores = score_image_text_pairs(text, indices, image_features, consensus, residual)
        matrix_scores = score_text_against_images(text, image_features, consensus, residual)

        self.assertTrue(torch.allclose(pair_scores, matrix_scores.diag(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

