import skimage

from tests import IMATestCase


class TestImageResult(IMATestCase):
    min_psnr = 30.0  # dB

    def setUp(self):
        self.im_path = ''

    def test_image(self):
        expected_gray = skimage.util.img_as_float(skimage.io.imread(self.im_path))
        gray = skimage.util.img_as_float(self.params['img'])
        self.assertEqual(gray.shape, expected_gray.shape, msg='Morphological output must not change image size')
        psnr = skimage.metrics.peak_signal_noise_ratio(expected_gray, gray)
        print(f'PSNR for {self.im_path}: {psnr:.2f} dB')
        self.assertGreaterEqual(
            psnr, self.min_psnr, msg=f'Morphologically processed image too different from the reference (PSNR={psnr:.2f}dB)'
        )


class TestMask(TestImageResult):
    def setUp(self):
        self.im_path = self.rpath('tests/data/color_popping-expected_mask.png')


class TestColorPop(TestImageResult):
    min_psnr = 40.0  # dB

    def setUp(self):
        self.im_path = self.rpath('tests/data/color_popping-expected_color_pop.png')
