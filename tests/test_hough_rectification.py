import numpy as np
import skimage

from tests import IMATestCase
from .test_morphological_reconstruction import TestImageResult


class TestCanny(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/hough_rectification-expected_edges.png')


class TestHoughLines(IMATestCase):
    def test_hough_lines(self):
        expected_lines = np.array(
            [
                [108.0, 1.361],
                [367.0, -0.506],
                [499.0, 1.292],
                [14.0, -0.297],
            ]
        )
        lines = self.params['lines']
        for exp_dist, exp_angle in expected_lines:
            self.assertLess(np.abs(lines[:, 0] - exp_dist).min(), 10.0, msg='Detected lines do not contain outer borders.')
            self.assertLess(np.abs(lines[:, 1] - exp_angle).min(), 0.15, msg='Detected lines do not contain outer borders.')


class TestOuterHoughLines(IMATestCase):
    def test_hough_lines(self):
        expected_lines = np.array(
            [
                [108.0, 1.361],
                [367.0, -0.506],
                [499.0, 1.292],
                [14.0, -0.297],
            ]
        )
        lines = self.params['outer_lines']
        self.assertArraysClose(np.sort(lines[:, 0]), np.sort(expected_lines[:, 0]), rtol=0.0, atol=10.0)
        self.assertArraysClose(np.sort(lines[:, 1]), np.sort(expected_lines[:, 1]), rtol=0.0, atol=0.15)


class TestLineIntersection(IMATestCase):
    def test_line_intersection(self):
        line_intersect_hnf = self.params['line_intersect_hnf_fn']
        line1 = np.array([123.456, 1.234])
        line2 = np.array([654.321, -0.789])
        x, y = line_intersect_hnf(line1, line2)
        self.assertAlmostEqual(x, 783.970, places=3, msg='Incorrect x coordinate of line intersection.')
        self.assertAlmostEqual(y, -143.692, places=3, msg='Incorrect y coordinate of line intersection.')


class TestROI(TestImageResult):
    def test_image(self):
        expected_gray = skimage.util.img_as_float(skimage.io.imread(self.rpath('tests/data/sudoku-alt3-roi.png')))
        gray = skimage.util.img_as_float(self.params['img'])
        if gray.ndim == 3:
            gray = skimage.color.rgb2gray(gray)
        gray = skimage.transform.resize(gray, expected_gray.shape, anti_aliasing=True)
        shift, _, _ = skimage.registration.phase_cross_correlation(expected_gray, gray, upsample_factor=10)
        self.assertLess(np.linalg.norm(shift), 10.0, msg='Extracted ROI is misaligned with the expected one.')
