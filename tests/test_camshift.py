import numpy as np
import scipy

from tests import IMATestCase


class TestCamShiftInit(IMATestCase):
    def test_implementation(self):
        self.assertNotCalling(
            self.params['init_camshift_fn'],
            ['calcHist', 'cvtColor'],
            msg='OpenCV not allowed',
        )

    def test_init_camshift(self):
        self.seed_everything()

        img, roi = toy_square((50, 50))
        hist, bins = self.params['init_camshift_fn'](img, roi)
        self.assertAlmostEqual(np.sum(hist), 1.0, places=5, msg='Histogram must be normalized to sum to 1.0.')

        max_prob = np.max(hist)
        self.assertAlmostEqual(
            max_prob,
            1.0,
            places=5,
            msg='The input ROI contains only one color. Therefore, a correct histogram of that ROI should have all its '
            'probability mass (1.0) in a single bin.',
        )


class TestBackprojectHistogram(IMATestCase):
    def test_implementation(self):
        self.assertNotCalling(
            self.params['backproject_histogram_fn'],
            ['calcBackProject', 'cvtColor'],
            msg='OpenCV not allowed',
        )

    def test_backproject_histogram(self):
        self.seed_everything()

        hist, bins = toy_histogram()
        data = np.array([0.0, 0.7, 0.65, 0.66, 0.3, 1.9, 1.31, 1.39, 1.22, 1.0])
        bp = self.params['backproject_histogram_fn'](data, hist, bins)
        expected_bp = np.array([0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0])
        self.assertArraysClose(bp, expected_bp, msg='Backprojected values are incorrect.')


class TestCenterOfGravity(IMATestCase):
    def test_center_of_gravity(self):
        self.seed_everything()
        blob = toy_blob((58, 33))
        xc, yc = self.params['center_of_gravity_fn'](blob)
        self.assertAlmostEqual(xc, 58.0, places=1, msg='Incorrect x coordinate of center of gravity.')
        self.assertAlmostEqual(yc, 33.0, places=1, msg='Incorrect y coordinate of center of gravity.')


class TestCamShiftStep(IMATestCase):
    def test_implementation(self):
        self.assertNotCalling(
            self.params['camshift_step_fn'],
            ['CamShift', 'cvtColor', 'meanShift', 'calcBackProject'],
            msg='OpenCV not allowed',
        )
        self.assertCalling(self.params['camshift_step_fn'], ['backproject_histogram', 'center_of_gravity'])

    def test_camshift_step(self):
        self.seed_everything()

        init_frame, init_box = toy_square(center=(50, 50))
        hist, bins = self.params['init_camshift_fn'](init_frame, init_box)
        frame, expected_box = toy_square(center=(57, 44))
        box = self.params['camshift_step_fn'](frame, hist, bins, init_box, steps=10)

        self.assertIsInstance(box, tuple, msg='Box must be a tuple.')
        self.assertEqual(len(box), 4, msg='Box must be a tuple of length 4.')
        self.assertArraysClose(
            np.array(box), np.array(expected_box), rtol=0.0, atol=1.0, msg='Box coordinates are incorrect.'
        )


def toy_square(center: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]
    img[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10] = [128, 0, 0]
    img[y1 : y2 + 1, x1 : x2 + 1] = [0, 0, 255]
    img[y1 + 10 : y2 - 10, x1 + 10 : x2 - 10] = [0, 0, 128]
    return img, (x1, y1, x2, y2)


def toy_histogram() -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 2.0, 11)
    hist = np.zeros(10)
    hist[3] = 0.2  # 0.6..0.8
    hist[6] = 0.8  # 1.2..1.4
    return hist, bins


def toy_blob(center: tuple[int, int]) -> np.ndarray:
    blob = np.zeros((80, 80))
    blob[center[1], center[0]] = 1.0
    blob = scipy.ndimage.gaussian_filter(blob, sigma=5.0)
    return blob
