import numpy as np
import skimage

from tests import IMATestCase


class TestImageResult(IMATestCase):
    def setUp(self):
        self.im_path = ''

    def test_image(self):
        expected_gray = skimage.util.img_as_float(skimage.io.imread(self.im_path, as_gray=True))
        gray = skimage.util.img_as_float(self.params['img'])
        self.assertEqual(gray.shape, expected_gray.shape, msg='Morphological output must not change image size')
        psnr = skimage.metrics.peak_signal_noise_ratio(expected_gray, gray)
        print(f'PSNR for {self.im_path}: {psnr:.2f} dB')
        self.assertGreaterEqual(
            psnr, 40, msg=f'Morphologically processed image too different from the reference (PSNR={psnr:.2f}dB)'
        )


class TestOpenReconstruction(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/morphological_reconstruction-open.png')


class TestTophatReconstruction(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/morphological_reconstruction-tophat.png')


class TestOpenReconstruction2(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/morphological_reconstruction-open2.png')


class TestDilationReconstruction(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/morphological_reconstruction-dilation.png')


class TestBackgroundRemoval(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/morphological_reconstruction-final.png')


class TestDilateReconstructFunction(IMATestCase):
    def setUp(self):
        self.x = np.array(
            [
                [23, 47, 32, 41, 10, 25, 18, 36],
                [14, 9, 15, 19, 20, 44, 23, 15],
                [46, 5, 10, 34, 35, 31, 28, 48],
                [31, 26, 28, 16, 27, 36, 13, 18],
                [28, 30, 14, 10, 8, 43, 37, 17],
                [28, 39, 25, 17, 6, 1, 2, 1],
            ],
            dtype=np.uint8,
        )
        self.mask = np.array(
            [
                [32, 31, 23, 46, 30, 34, 45, 42],
                [25, 20, 39, 42, 25, 40, 27, 46],
                [22, 46, 31, 34, 26, 46, 30, 34],
                [49, 26, 33, 26, 34, 35, 49, 43],
                [23, 21, 28, 20, 49, 35, 47, 35],
                [34, 40, 49, 23, 41, 20, 37, 47],
            ],
            dtype=np.uint8,
        )

    def test_reconstruction(self):
        dilate_reconstruct = self.params['dilate_reconstruct_fn']
        x_rec = dilate_reconstruct(self.x, self.mask)
        expected_x_rec = np.array(
            [
                [32, 31, 23, 41, 30, 34, 42, 42],
                [25, 20, 39, 41, 25, 40, 27, 46],
                [22, 46, 31, 34, 26, 44, 30, 34],
                [46, 26, 31, 26, 34, 35, 43, 43],
                [23, 21, 28, 20, 43, 35, 43, 35],
                [34, 39, 39, 23, 41, 20, 37, 37],
            ],
            dtype=np.uint8,
        )
        self.assertArraysEqual(x_rec, expected_x_rec, msg='dilate_reconstruct produced incorrect reconstruction')

    def test_max_iters(self):
        dilate_reconstruct = self.params['dilate_reconstruct_fn']
        x_rec = dilate_reconstruct(self.x, self.mask, max_iters=2)
        expected_x_rec = np.array(
            [
                [32, 31, 23, 41, 30, 34, 36, 42],
                [25, 20, 39, 41, 25, 40, 27, 46],
                [22, 46, 31, 34, 26, 44, 30, 34],
                [46, 26, 31, 26, 34, 35, 43, 43],
                [23, 21, 28, 20, 43, 35, 43, 35],
                [34, 39, 39, 23, 41, 20, 37, 37],
            ],
            dtype=np.uint8,
        )
        self.assertArraysEqual(x_rec, expected_x_rec, msg='dilate_reconstruct ignores max_iters parameter')
    
    def test_strel(self):
        dilate_reconstruct = self.params['dilate_reconstruct_fn']
        se = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        x_rec = dilate_reconstruct(self.x, self.mask, strel=se)
        expected_x_rec = np.array(
            [
                [32, 31, 23, 41, 30, 34, 36, 36],
                [25, 20, 39, 41, 25, 40, 27, 36],
                [22, 46, 31, 34, 26, 44, 30, 34],
                [46, 26, 31, 26, 34, 35, 43, 43],
                [23, 21, 28, 20, 43, 35, 43, 35],
                [34, 39, 39, 23, 41, 20, 37, 37],
            ],
            dtype=np.uint8,
        )
        self.assertArraysEqual(x_rec, expected_x_rec, msg='dilate_reconstruct ignores strel parameter')
