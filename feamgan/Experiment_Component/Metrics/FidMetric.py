import numpy as np
from scipy import linalg

from feamgan.Experiment_Component.Metrics.BaseIDMetric import BaseIDMetric

class FidMetric(BaseIDMetric):
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name="inception_v3"):
        super(FidMetric, self).__init__(is_video, model_dir, dataset_name, dis_model_name)

    def _calculateDistance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        return self._calculateFrechetDistance(mu1, sigma1, mu2, sigma2, eps=eps)

    def _calculateFrechetDistance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Args:
        :param mu1: Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        :param mu2: The sample mean over activations, pre-calculated on an
                representative data set.
        :param sigma1: The covariance matrix over activations for generated samples.
        :param sigma2: The covariance matrix over activations, pre-calculated on an
                representative data set.
        :param eps: a value added to the diagonal of cov for numerical stability.
        :return: The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'
        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print('Imaginary component {}'.format(m))
                # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

           