import torch
import os
import numpy as np

from feamgan.utils.distUtils import distAllGatherTensor, isMaster
from feamgan.Experiment_Component.Metrics.BaseIDMetric import BaseIDMetric

class KidMetric(BaseIDMetric):
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name="inception_v3"):
        super(KidMetric, self).__init__(is_video, model_dir, dataset_name, dis_model_name)
        self.save_path = f"{model_dir}/kid_real_act"

    @torch.no_grad()
    def reduceBatches(self, mode, save_real_prefix=None): 
        meter_key = f"{save_real_prefix}_{mode}"
        path = f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}"
        if meter_key in self.meters:
            if save_real_prefix and os.path.exists(path) and (mode != "train"):
                print('Load KID activations from {}'.format(path))
                npz_file = np.load(path)
                real_activations = npz_file['real_activations']
            else:
                if self.meters[meter_key]["real"]:
                    real_activations = self.meters[meter_key]["real"]
                    real_activations = torch.cat(real_activations)
                    real_activations = distAllGatherTensor(real_activations)
                    if isMaster():
                        real_activations = torch.cat(real_activations).cpu().data.numpy()
                    if save_real_prefix:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if isMaster():
                            np.savez(path, real_activations=real_activations)
                else:
                    return None

            if self.meters[meter_key]["fake"]:
                fake_activations = self.meters[meter_key]["fake"]
                fake_activations = torch.cat(fake_activations)
                fake_activations = distAllGatherTensor(fake_activations)
                kid = None
                if isMaster():
                    fake_activations = torch.cat(fake_activations).cpu().data.numpy()
                    mmd, mmd_vars = self._polynomial_mmd_averages(fake_activations, real_activations)
                    kid = mmd.mean()
            else:
                return None
                
            self.meters[meter_key]["real"] = []
            self.meters[meter_key]["fake"] = []
            return kid
        else: 
            return None

    def _polynomial_mmd_averages(self, codes_g, codes_r, n_subsets=1, subset_size=None,
                            ret_var=True, device='cpu', **kernel_args):
        """
        Computes MMD between two sets of features using polynomial kernels. It
        performs a number of repetitions of subset sampling without replacement.
 
        :param codes_g (Tensor): Feature activations of generated images.
        :param codes_r (Tensor): Feature activations of real images.
        :param n_subsets (int): The number of subsets.
        :param subset_size (int): The number of samples in each subset.
        :param ret_var (bool): If ``True``, returns both mean and variance of MMDs,
                otherwise only returns the mean.d
        :return:
            (tuple):
            - mmds (Tensor): Mean of MMDs.
            - mmd_vars (Tensor): Variance of MMDs.
        """
        codes_g = torch.tensor(codes_g, device=torch.device(device))
        codes_r = torch.tensor(codes_r, device=torch.device(device))
        mmds = np.zeros(n_subsets)
        if ret_var:
            mmd_vars = np.zeros(n_subsets)
        choice = np.random.choice

        if subset_size is None:
            subset_size = min(len(codes_r), len(codes_r))
            print("Subset size not provided, "
                "setting it to the data size ({}).".format(subset_size))
        if subset_size > len(codes_g) or subset_size > len(codes_r):
            subset_size = min(len(codes_r), len(codes_r))
            print("Subset size is large than the actual data size, "
                "setting it to the data size ({}).".format(subset_size))

        for i in range(n_subsets):
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = self._polynomial_mmd(g, r, **kernel_args, ret_var=ret_var)
            if ret_var:
                mmds[i], mmd_vars[i] = o
            else:
                mmds[i] = o
        return (mmds, mmd_vars) if ret_var else mmds


    def _polynomial_kernel(self, X, Y=None, degree=3, gamma=None, coef0=1.):
        r"""Compute the polynomial kernel between X and Y"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        if Y is None:
            Y = X

        # K = safe_sparse_dot(X, Y.T, dense_output=True)
        K = torch.matmul(X, Y.t())
        K *= gamma
        K += coef0
        K = K**degree
        return K


    def _polynomial_mmd(self, codes_g, codes_r, degree=3, gamma=None, coef0=1,
                    ret_var=True):
        """
        Computes MMD between two sets of features using polynomial kernels. It
        performs a number of repetitions of subset sampling without replacement.
    
        :param codes_g (Tensor): Feature activations of generated images.
        :param codes_r (Tensor): Feature activations of real images.
        :param degree (int): The degree of the polynomial kernel.
        :param gamma (float or None): Scale of the polynomial kernel.
        :param coef0 (float or None): Bias of the polynomial kernel.
        :param ret_var (bool): If ``True``, returns both mean and variance of MMDs,
                otherwise only returns the mean.
        :return:
            (tuple):
            - mmds (Tensor): Mean of MMDs.
            - mmd_vars (Tensor): Variance of MMDs.
        """
        # use  k(x, y) = (gamma <x, y> + coef0)^degree
        # default gamma is 1 / dim
        X = codes_g
        Y = codes_r

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        K_XX = self._polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = self._polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = self._polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

        return self._mmd2_and_variance(K_XX, K_XY, K_YY, ret_var=ret_var)


    def _mmd2_and_variance(self, K_XX, K_XY, K_YY, unit_diagonal=False,
                        mmd_est='unbiased', ret_var=True):
        """
        Based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
        but changed to not compute the full kernel matrix at once
        """

        m = K_XX.shape[0]
        assert K_XX.shape == (m, m)
        assert K_XY.shape == (m, m)
        assert K_YY.shape == (m, m)
        var_at_m = m

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if unit_diagonal:
            diag_X = diag_Y = 1
            sum_diag_X = sum_diag_Y = m
            sum_diag2_X = sum_diag2_Y = m
        else:
            diag_X = torch.diagonal(K_XX)
            diag_Y = torch.diagonal(K_YY)

            sum_diag_X = diag_X.sum()
            sum_diag_Y = diag_Y.sum()

            sum_diag2_X = self._sqn(diag_X)
            sum_diag2_Y = self._sqn(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)
        K_XY_sums_1 = K_XY.sum(dim=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        if mmd_est == 'biased':
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2 * K_XY_sum / (m * m))
        else:
            assert mmd_est in {'unbiased', 'u-statistic'}
            mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
            if mmd_est == 'unbiased':
                mmd2 -= 2 * K_XY_sum / (m * m)
            else:
                mmd2 -= 2 * (K_XY_sum - torch.trace(K_XY)) / (m * (m - 1))

        if not ret_var:
            return mmd2

        Kt_XX_2_sum = self._sqn(K_XX) - sum_diag2_X
        Kt_YY_2_sum = self._sqn(K_YY) - sum_diag2_Y
        K_XY_2_sum = self._sqn(K_XY)

        dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
        dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

        m1 = m - 1
        m2 = m - 2

        zeta1_est = (
            1 / (m * m1 * m2) * (
                self._sqn(Kt_XX_sums) - Kt_XX_2_sum + self._sqn(Kt_YY_sums) - Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 1 / (m * m * m1) * (
                self._sqn(K_XY_sums_1) + self._sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
            - 2 / m ** 4 * K_XY_sum ** 2
            - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        zeta2_est = (
            1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 2 / (m * m) * K_XY_2_sum
            - 2 / m ** 4 * K_XY_sum ** 2
            - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
                + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

        return mmd2.cpu().numpy(), var_est.cpu().numpy()


    def _sqn(self, arr):
        """
        Squared norm.
        """
        flat = arr.view(-1)
        return flat.dot(flat)
            