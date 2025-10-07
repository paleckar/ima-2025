import numpy as np
import skimage

from tests import IMATestCase


class TestDepthOfField(IMATestCase):
    def test_dof(self):
        self.seed_everything()

        img = np.random.rand(100, 100)
        mask = np.vstack([np.ones((50, 100)), np.zeros((50, 100))])
        prev_bottom_blur = 0.0
        for sigma in [3.0, 6.0]:
            dof = self.params['depth_of_field_fn'](img, mask, sigma=sigma)
            self.assertTupleEqual(img.shape, dof.shape, msg='depth_of_field function must preserve the image size')
            self.assertEqual(img.dtype, dof.dtype, msg='depth_of_field function must preserve the image dtype')

            img_blur = skimage.measure.blur_effect(img)
            dof_top_blur = skimage.measure.blur_effect(dof[:50, :])
            dof_bottom_blur = skimage.measure.blur_effect(dof[50:, :])
            self.assertGreater(
                dof_top_blur,
                img_blur,
                msg='Depth of field effect should increase blur strength even in focused areas a little',
            )
            self.assertGreater(
                dof_bottom_blur,
                dof_top_blur,
                msg='Blur strength should be higher in defocused areas than in focused areas',
            )
            self.assertGreater(
                dof_bottom_blur,
                prev_bottom_blur,
                msg='Blur strength should be stronger for larger sigma values',
            )
            prev_bottom_blur = dof_bottom_blur


class TestJitterEffect(IMATestCase):
    def test_watermark(self):
        self.seed_everything()

        rgb = skimage.io.imread(self.rpath('data/fruits.jpg'))
        for j in [1, 5, 15]:
            jit1 = self.params['jitter_filter_fn'](rgb, jitter=j)
            self.assertTupleEqual(rgb.shape, jit1.shape, msg='jitter_filter function must preserve the image size')
            self.assertEqual(rgb.dtype, jit1.dtype, msg='jitter_filter function must preserve the image dtype')
            self.assertTrue(
                set(jit1.ravel()) <= set(rgb.ravel()),  # check subset
                msg='jitter_filter function must not introduce new colors',
            )
            same_ratio_1 = np.mean(rgb.ravel() == jit1.ravel()).item()  # ratio of unchanged pixels
            expected_same_ratio = np.mean(
                [
                    np.mean(rgb.ravel() == np.roll(rgb, shift=(dy, dx), axis=(0, 1)).ravel())
                    for dy in range(-j, j + 1)
                    for dx in range(-j, j + 1)
                ]
            ).item()
            self.assertLess(
                abs(same_ratio_1 - expected_same_ratio) / expected_same_ratio,  # relative error
                0.1,  # 10% tolerance
                msg='jitter_filter function must uniformly randomly reorder pixels',
            )

            # Do another jitter to check randomness
            jit2 = self.params['jitter_filter_fn'](rgb, jitter=j)
            same_ratio_2 = np.mean(jit1.ravel() == jit2.ravel()).item()  # ratio of unchanged pixels between two jitters
            self.assertLess(
                abs(same_ratio_2 - expected_same_ratio) / expected_same_ratio,  # relative error
                0.1,  # 10% tolerance
                msg='jitter_filter function must uniformly randomly reorder pixels',
            )
