import numpy as np
import skimage

from tests import IMATestCase


class TestWarpInv(IMATestCase):
    def test_warp_inv(self):
        self.seed_everything()
        gray = np.array([[215, 202, 36], [211, 79, 202]], dtype=np.float64)
        A_hom = np.array([[2.46, -0.3, 1.0], [0.4, 1.87, 2.0], [0.01, -0.02, 1.0]])
        gray_nn = self.params['warp_inv_fn'](gray, A_hom, *compute_tform_span(gray, A_hom), interp='nearest')
        gray_lin = self.params['warp_inv_fn'](gray, A_hom, *compute_tform_span(gray, A_hom), interp='linear')
        expected_gray_nn = np.array(
            [
                [215.0, 215.0, 202.0, 202.0, 202.0, 36.0, 36.0],
                [211.0, 215.0, 202.0, 202.0, 202.0, 36.0, 36.0],
                [211.0, 211.0, 79.0, 79.0, 79.0, 202.0, 202.0],
                [0.0, 211.0, 79.0, 79.0, 202.0, 202.0, 202.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected_gray_lin = np.array(
            [
                [219.107, 213.624, 215.734, 225.697, 140.093, 28.362, -29.0],
                [226.318, 197.574, 175.757, 160.516, 130.395, 81.112, 36.562],
                [215.43, 175.187, 130.278, 105.756, 134.158, 145.081, 94.429],
                [102.023, 89.592, 69.831, 63.655, 107.289, 158.566, 119.01],
                [0.0, 8.952, 13.302, 18.932, 38.324, 64.692, 49.913],
            ],
            dtype=np.float64,
        )
        self.assertArraysClose(
            gray_nn, expected_gray_nn, rtol=0.01, atol=0.01, msg='warp_inv nearest neighbor interpolation failed'
        )
        self.assertArraysClose(
            gray_lin, expected_gray_lin, rtol=0.01, atol=0.01, msg='warp_inv linear interpolation failed'
        )


class TestComputeRotationMatrix(IMATestCase):
    def test_warp_inv(self):
        self.seed_everything()
        mat = self.params['compute_rotation_matrix_fn'](19.732, 951, 388)
        expected_mat = np.array([[0.941, 0.338, 0.0], [-0.338, 0.941, 54.019], [0.0, 0.0, 1.0]])
        self.assertArraysClose(
            mat, expected_mat, rtol=0.01, atol=0.01, msg='compute_rotation_matrix output is incorrect'
        )


class TestRectifyCv(IMATestCase):
    def test_rectify_cv(self):
        rgb = skimage.io.imread(self.rpath('data/sudoku-alt3.jpg'))[..., ::-1]
        src_pts = self.params['src_pts']
        rectify_cv = self.params['rectify_cv_fn']
        self.assertCallingAnyOf(rectify_cv, ['findHomography', 'getPerspectiveTransform'])
        self.assertCalling(rectify_cv, ['warpPerspective'])
        roi = rectify_cv(rgb, src_pts, (270, 270))
        roi = skimage.util.img_as_ubyte(roi)
        expected_roi = skimage.io.imread(self.rpath('tests/data/sudoku-alt3-roi.png'))[..., ::-1]
        mse: float = skimage.metrics.mean_squared_error(roi, expected_roi)
        self.assertLess(
            mse, 2000.0, msg=f'Rectified output is incorrect, mean squared must be lower than 2000, got {mse}'
        )


def compute_tform_span(img: np.ndarray, A: np.ndarray) -> tuple[tuple[int, int], tuple[float, float]]:
    h, w = img.shape[:2]

    # Get image corners and transform them
    X = np.array([0.0, w, w, 0.0])
    Y = np.array([0.0, 0.0, h, h])
    XY = np.vstack((X, Y, np.ones(4)))
    XY_ = np.dot(A, XY)
    XY_ = XY_[:2] / XY_[2]  # normalize homogeneous coordinates - this is extra compared to affine case
    dx_, dy_ = XY_[0, :].min(), XY_[1, :].min()

    # Calculate the new size
    w_ = int(XY_[0, :].max() - XY_[0, :].min())
    h_ = int(XY_[1, :].max() - XY_[1, :].min())

    return (h_, w_), (dx_, dy_)
