"""Tests for Perception AI terrain classification module."""
import unittest
import numpy as np
from drishti.perception import (
    PerceptionModel, extract_patch_features,
    CLASS_TRAVERSABLE, CLASS_OBSTACLE, CLASS_WATER_MUD, CLASS_VEGETATION
)


class PerceptionAITests(unittest.TestCase):
    def setUp(self):
        self.model = PerceptionModel()

    def test_feature_vector_dimensionality(self):
        rgb_patch = np.zeros((20, 20, 3), dtype=np.uint8)
        depth_patch = np.full((20, 20), 2.5, dtype=np.float32)
        feat = extract_patch_features(rgb_patch, depth_patch)
        self.assertEqual(feat.shape, (32,))
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_model_forward_probabilities(self):
        x = np.random.randn(5, 32).astype(np.float32)
        probs = self.model.forward(x)
        self.assertEqual(probs.shape, (5, 4))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_vegetation_classification(self):
        # Green patch with vegetation color ratio
        patch = np.zeros((24, 24, 3), dtype=np.uint8)
        patch[:, :, 1] = 190  # Strong green
        patch[:, :, 0] = 50   # Low red
        patch[:, :, 2] = 40   # Low blue
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        feat = extract_patch_features(patch, depth)
        pred = self.model.predict(feat)[0]
        self.assertEqual(pred, CLASS_VEGETATION)

    def test_water_mud_classification(self):
        # Dark murky brown-blue water/mud patch with smooth texture
        patch = np.zeros((24, 24, 3), dtype=np.uint8)
        patch[:, :, 0] = 45   # Dark mud
        patch[:, :, 1] = 55
        patch[:, :, 2] = 60
        # Partial disparity dropout characteristic of water/sludge
        depth = np.full((24, 24), np.nan, dtype=np.float32)
        depth[10:14, 10:14] = 3.0
        feat = extract_patch_features(patch, depth)
        pred = self.model.predict(feat)[0]
        self.assertEqual(pred, CLASS_WATER_MUD)

    def test_segment_frame_shapes(self):
        rgb = np.zeros((400, 640, 3), dtype=np.uint8)
        rgb[:200, :, 1] = 180  # Upper half vegetation
        rgb[200:, :, :] = 120  # Lower half ground
        depth = np.full((400, 640), 2.5, dtype=np.float32)

        sem_grid, conf_grid, colored_mask = self.model.segment_frame(rgb, depth, patch_size=20)
        self.assertEqual(sem_grid.shape, (20, 32))
        self.assertEqual(conf_grid.shape, (20, 32, 4))
        self.assertEqual(colored_mask.shape, (400, 640, 3))

    def test_traversability_cost_mapping(self):
        conf_grid = np.zeros((10, 10, 4), dtype=np.float32)
        # Class 0: TRAVERSABLE -> cost should be low
        conf_grid[:, :, 0] = 1.0
        cost_safe = self.model.compute_traversability_cost(conf_grid)
        self.assertAlmostEqual(float(np.mean(cost_safe)), 0.0, places=3)

        # Class 1: OBSTACLE -> cost should be 1.0
        conf_grid[:, :, :] = 0.0
        conf_grid[:, :, 1] = 1.0
        cost_obs = self.model.compute_traversability_cost(conf_grid)
        self.assertAlmostEqual(float(np.mean(cost_obs)), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
