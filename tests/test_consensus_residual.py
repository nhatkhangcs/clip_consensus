import unittest

import torch

from caption_geometry.scoring.consensus_residual import (
    compute_full_consensus,
    compute_residual,
    leave_one_out_consensus,
    normalize,
    score_matrix_no_loo,
    score_matrix_with_loo,
)


class ConsensusResidualTest(unittest.TestCase):
    def test_consensus_and_residual_are_normalized(self):
        captions = normalize(torch.randn(4, 5, 8))
        images = normalize(torch.randn(4, 8))

        consensus = compute_full_consensus(captions)
        residual = compute_residual(images, consensus)

        self.assertTrue(torch.allclose(consensus.norm(dim=-1), torch.ones(4), atol=1e-6))
        self.assertTrue(torch.allclose(residual.norm(dim=-1), torch.ones(4), atol=1e-6))

    def test_leave_one_out_matches_manual_average(self):
        captions = normalize(torch.randn(2, 3, 5))
        loo = leave_one_out_consensus(captions)
        manual = torch.stack(
            [
                torch.stack(
                    [
                        normalize(captions[i, [j for j in range(3) if j != m]].mean(dim=0))
                        for m in range(3)
                    ]
                )
                for i in range(2)
            ]
        )

        self.assertTrue(torch.allclose(loo, manual, atol=1e-6))

    def test_no_loo_score_matches_direct_formula(self):
        images = normalize(torch.randn(3, 6))
        captions = normalize(torch.randn(3, 4, 6))
        consensus = compute_full_consensus(captions)
        residual = compute_residual(images, consensus)
        queries = captions.reshape(12, 6)

        expected = (
            1.0 * (queries @ images.T)
            + 0.4 * (queries @ consensus.T)
            + 1.8 * (queries @ residual.T)
        )
        actual = score_matrix_no_loo(images, captions, alpha=1.0, beta=0.4, gamma=1.8)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_loo_score_matches_reference_loop(self):
        images = normalize(torch.randn(3, 6))
        captions = normalize(torch.randn(3, 4, 6))
        expected = reference_loo_scores(images, captions, alpha=1.0, beta=0.4, gamma=1.8)
        actual = score_matrix_with_loo(images, captions, alpha=1.0, beta=0.4, gamma=1.8)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


def reference_loo_scores(images, captions, alpha, beta, gamma):
    n_images, k_captions, _ = captions.shape
    full_consensus = compute_full_consensus(captions)
    full_residual = compute_residual(images, full_consensus)
    loo_consensus = leave_one_out_consensus(captions)
    scores = torch.empty(n_images * k_captions, n_images)

    for i in range(n_images):
        for m in range(k_captions):
            query = captions[i, m]

            for j in range(n_images):
                consensus = loo_consensus[i, m] if i == j else full_consensus[j]
                image = images[j]
                residual = compute_residual(image[None], consensus[None])[0] if i == j else full_residual[j]
                scores[i * k_captions + m, j] = (
                    alpha * torch.dot(image, query)
                    + beta * torch.dot(consensus, query)
                    + gamma * torch.dot(residual, query)
                )

    return scores


if __name__ == "__main__":
    unittest.main()

