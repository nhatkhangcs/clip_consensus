import unittest

import torch

from caption_geometry.scoring.consensus_residual import normalize
from caption_geometry.scoring.retrieval_scores import (
    score_base_avg_support,
    score_base_max_support,
    score_consensus_residual,
    score_retrieval_variant,
)


class RetrievalScoresTest(unittest.TestCase):
    def test_avg_support_loo_matches_reference(self):
        images = normalize(torch.randn(3, 5))
        captions = normalize(torch.randn(3, 4, 5))
        actual = score_base_avg_support(
            images=images,
            query_captions=captions,
            support_captions=captions,
            alpha=1.0,
            support_weight=0.7,
            leave_one_out=True,
        )
        expected = reference_avg_support(images, captions, alpha=1.0, weight=0.7)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_max_support_loo_matches_reference(self):
        images = normalize(torch.randn(3, 5))
        captions = normalize(torch.randn(3, 4, 5))
        actual = score_base_max_support(
            images=images,
            query_captions=captions,
            support_captions=captions,
            alpha=1.0,
            support_weight=0.7,
            leave_one_out=True,
        )
        expected = reference_max_support(images, captions, alpha=1.0, weight=0.7)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_full_variant_accepts_separate_support(self):
        images = normalize(torch.randn(3, 5))
        queries = normalize(torch.randn(3, 4, 5))
        support = normalize(torch.randn(3, 2, 5))
        source_indices = torch.tensor([[0, 2], [1, -1], [3, 0]])

        scores = score_retrieval_variant(
            "full",
            images=images,
            query_captions=queries,
            support_captions=support,
            support_source_indices=source_indices,
            leave_one_out=True,
        )

        self.assertEqual(tuple(scores.shape), (12, 3))

    def test_base_consensus_variant_matches_direct_scorer(self):
        images = normalize(torch.randn(2, 4))
        captions = normalize(torch.randn(2, 3, 4))
        via_variant = score_retrieval_variant(
            "base_consensus",
            images=images,
            query_captions=captions,
            support_captions=captions,
            beta=0.5,
            leave_one_out=True,
        )
        direct = score_consensus_residual(
            images=images,
            query_captions=captions,
            support_captions=captions,
            beta=0.5,
            gamma=0.0,
            leave_one_out=True,
        )

        self.assertTrue(torch.allclose(via_variant, direct, atol=1e-6))


def reference_avg_support(images, captions, alpha, weight):
    n_images, k_captions, _ = captions.shape
    scores = torch.empty(n_images * k_captions, n_images)

    for i in range(n_images):
        for m in range(k_captions):
            query = captions[i, m]

            for j in range(n_images):
                support = captions[j, [idx for idx in range(k_captions) if i != j or idx != m]]
                component = support @ query
                scores[i * k_captions + m, j] = alpha * torch.dot(images[j], query) + weight * component.mean()

    return scores


def reference_max_support(images, captions, alpha, weight):
    n_images, k_captions, _ = captions.shape
    scores = torch.empty(n_images * k_captions, n_images)

    for i in range(n_images):
        for m in range(k_captions):
            query = captions[i, m]

            for j in range(n_images):
                support = captions[j, [idx for idx in range(k_captions) if i != j or idx != m]]
                component = support @ query
                scores[i * k_captions + m, j] = alpha * torch.dot(images[j], query) + weight * component.max()

    return scores


if __name__ == "__main__":
    unittest.main()

