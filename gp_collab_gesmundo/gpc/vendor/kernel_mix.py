"""Vendored from GP_collab `code_python/training/GPytorch_kernel_mix.py`.

Trimmed to the MAP path used here: the pyro/HMC regressor and the SSK kernel
were dropped so the cluster environment needs neither pyro nor the SSK module.
Behavioural edits are marked `# DOPE:` and are all opt-in; leaving the new
keywords at their defaults reproduces GP_collab exactly.
"""
import os
import sys

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from gpytorch.kernels import Kernel, ProductKernel, AdditiveKernel
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from tqdm import tqdm

def weighted_tanimoto(x1, x2, eps=1e-6, dist=True, batch_size=128):
    """
    Compute the pairwise weighted Tanimoto similarity or distance.

    ``batch_size`` limits the number of rows from ``x1`` used in each
    broadcasted pairwise calculation. Set it to ``None`` to use the original
    full calculation.
    """
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer or None.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

    def _calculate(x1_block):
        x1e = x1_block.unsqueeze(-2)
        x2e = x2.unsqueeze(-3)

        numerator = torch.min(x1e, x2e).sum(dim=-1)
        denominator = torch.max(x1e, x2e).sum(dim=-1)

        w_t = (numerator + eps) / (denominator + eps)
        if dist:
            w_t = 1.0 - w_t
        return torch.clamp(w_t, min=0.0)

    if batch_size is None or batch_size >= x1.size(-2):
        return _calculate(x1)

    return torch.cat(
        [_calculate(x1_block) for x1_block in x1.split(batch_size, dim=-2)],
        dim=-2,
    )


class Tanimoto(Kernel):
    has_lengthscale = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, x1, x2, diag=False, **params):            
        if diag:    
            if torch.equal(x1, x2):
                return torch.ones(
                    *x1.shape[:-1],
                    device=x1.device,
                    dtype=x1.dtype
                )
            num = torch.min(x1, x2).sum(dim=-1)
            den = torch.max(x1, x2).sum(dim=-1)
            sim = (num + 1e-6) / (den + 1e-6)

            return torch.clamp(sim, min=0.)
        
        else:
            # if self.last_dim_is_batch:
                # x1 = x1.transpose(-1, -2).unsqueeze(-1)
                # x2 = x2.transpose(-1, -2).unsqueeze(-1)
            return weighted_tanimoto(x1, x2, dist=False)

class TanimotoRBF(Kernel):
    is_stationary = False
    has_lengthscale=True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, x1, x2, diag=False, **params):
        if diag:
            if torch.equal(x1, x2):
                return torch.ones(
                    *x1.shape[:-1],
                    device=x1.device,
                    dtype=x1.dtype
                )

            num = torch.min(x1, x2).sum(dim=-1)
            den = torch.max(x1, x2).sum(dim=-1)
            dist = 1.0 - (num + 1e-6) / (den + 1e-6)

            return torch.exp(
                -0.5 * (dist.clamp(min=0.) / self.lengthscale).pow(2)
            )

        dist = weighted_tanimoto(x1, x2)
        return torch.exp(
            -0.5 * (dist / self.lengthscale).pow(2)
        )


class TanimotoMatern(Kernel):
    is_stationary = False
    has_lengthscale=True
    def __init__(self, nu,**kwargs):
        super().__init__(has_lengthscale=True, **kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            if torch.equal(x1, x2):
                return torch.ones(
                    *x1.shape[:-1],
                    device=x1.device,
                    dtype=x1.dtype
                )

            num = torch.min(x1, x2).sum(dim=-1)
            den = torch.max(x1, x2).sum(dim=-1)
            dist = 1.0 - (num + 1e-6) / (den + 1e-6)
            r = dist / self.lengthscale
            if self.nu == 1.5:
                sqrt3_r = 3**0.5 * r
                K = (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
            elif self.nu == 2.5:
                sqrt5_r = 5**0.5 * r
                K = (1.0 + sqrt5_r + 5.0/3.0 * r**2) * torch.exp(-sqrt5_r)
            else:
                raise RuntimeError("nu expected to be 1.5 or 2.5")
            return K

        dist = weighted_tanimoto(x1, x2)
        r = dist / self.lengthscale
        if self.nu == 1.5:
            sqrt3_r = 3**0.5 * r
            K = (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
        elif self.nu == 2.5:
            sqrt5_r = 5**0.5 * r
            K = (1.0 + sqrt5_r + 5.0/3.0 * r**2) * torch.exp(-sqrt5_r)
        else:
            raise RuntimeError("nu expected to be 1.5 or 2.5")
        return K
    
    
kernel_factory = {
    "TanimotoRBF": TanimotoRBF,
    "TanimotoMatern32": lambda **kw: TanimotoMatern(nu=1.5, **kw),
    "TanimotoMatern52": lambda **kw: TanimotoMatern(nu=2.5, **kw),
    "Tanimoto": Tanimoto,
    "RBF": RBFKernel,
    "Matern32": lambda **kw: MaternKernel(nu=1.5, **kw),
    "Matern52": lambda **kw: MaternKernel(nu=2.5, **kw),
}


class MixingKernel:
    def __init__(
        self,
        feat_idx: dict,
        mixing_method: str,
        kernel_method: dict,
        variance: float | None = None,
        ssk_parameters: dict | None = None,
        cuda_avail: dict = None,
        ard: bool = True,  # DOPE: False gives one shared lengthscale per fp block
    ):
        self.feat_idx = feat_idx
        self.mixing_method = mixing_method
        self.kernel_method = kernel_method
        self.variance = variance if variance else 1.0
        self.ssk_parameters = ssk_parameters or {}
        self.cuda_avail = cuda_avail or {}
        self.ard = ard
    def _make_fp_kernels(self):
        fp_kernels = []
        fp_keys = sorted(k for k in self.feat_idx if k.startswith("fp_"))

        for key in fp_keys:
            idx = self.feat_idx[key]
            if idx:
                # print("ard dim for ", key, ":", len(idx))
                
                if "tanimoto" in self.kernel_method["fp"].lower():
                    k = kernel_factory[self.kernel_method["fp"]](
                        active_dims=idx,
                        # ard_num_dims=len(idx)
                    )

                else:

                    if self.kernel_method["fp"].lower() == "ssk":
                        ssk_params = self.ssk_parameters.get(key, {})
                        k = kernel_factory[self.kernel_method["fp"]](
                            active_dims=idx,
                            ard_num_dims=len(idx),
                            **ssk_params,
                            **self.cuda_avail
                        )
                    else:
                        # DOPE: ard=False -> ard_num_dims=None -> isotropic kernel
                        # over the whole block, which is the geometry hazel_gp ran.
                        k = kernel_factory[self.kernel_method["fp"]](
                            active_dims=idx,
                            ard_num_dims=len(idx) if self.ard else None,
                        )
                fp_kernels.append(k)
        # print(len(fp_kernels), "fp kernels created")
        return fp_kernels

    def _make_count_kernels(self):
        count_kernels = []
        count_idx = sorted(self.feat_idx.get("count", []))

        for dim in count_idx:
            k = kernel_factory[self.kernel_method["count"]](
                active_dims=[dim],
                **self.cuda_avail
                # ard_num_dims=1,
            )
            count_kernels.append(k)
        return count_kernels

    @staticmethod
    def _combine(kernels, kernel_cls):
        if len(kernels) == 1:
            return kernels[0]
        return kernel_cls(*kernels)


    def build(self):
        fp_kernels = self._make_fp_kernels()
        count_kernels = self._make_count_kernels()

        if not fp_kernels and not count_kernels:
            raise ValueError(
                "No kernels could be created: provide at least one fingerprint "
                "or continuous ('count') feature."
            )

        if self.mixing_method in {"sum", "product"}:
            kernels = fp_kernels + count_kernels
            if self.mixing_method == "sum":
                return self._combine(kernels, AdditiveKernel)

            return self._combine(kernels, ProductKernel)

        if self.mixing_method == "averageProduct":
            if fp_kernels and count_kernels:
                fp_product_kernel = self._combine(fp_kernels, ProductKernel)
                count_sum_kernel = self._combine(count_kernels, AdditiveKernel)

                averaged_count_kernel = ScaleKernel(count_sum_kernel)
                averaged_count_kernel.outputscale = 1.0 / len(count_kernels)

                return ProductKernel(fp_product_kernel, averaged_count_kernel)

            # With one feature family, the cross-family product reduces to the
            # kernel for the family that is present.  The outer ScaleKernel in
            # GPMix supplies the model variance, so a constant averaging factor
            # for count-only data would be redundant.
            if fp_kernels:
                return self._combine(fp_kernels, ProductKernel)
            return self._combine(count_kernels, AdditiveKernel)

        if self.mixing_method =="(count:+)x(fp:x)":
            if not fp_kernels:
                return self._combine(count_kernels, AdditiveKernel)
            if not count_kernels:
                return self._combine(fp_kernels, ProductKernel)

            fp_product_kernel = self._combine(fp_kernels, ProductKernel)
            count_sum_kernel = self._combine(count_kernels, AdditiveKernel)
            return ProductKernel(fp_product_kernel, count_sum_kernel)


        if self.mixing_method == "(count:+)x(fp:+)":
            if not fp_kernels:
                return self._combine(count_kernels, AdditiveKernel)
            if not count_kernels:
                return self._combine(fp_kernels, AdditiveKernel)

            fp_sum_kernel = self._combine(fp_kernels, AdditiveKernel)
            count_sum_kernel = self._combine(count_kernels, AdditiveKernel)
            return ProductKernel(fp_sum_kernel, count_sum_kernel)

        if self.mixing_method == "(count:x)+(fp:x)":
            if not fp_kernels:
                return self._combine(count_kernels, ProductKernel)
            if not count_kernels:
                return self._combine(fp_kernels, ProductKernel)

            fp_product_kernel = self._combine(fp_kernels, ProductKernel)
            count_product_kernel = self._combine(count_kernels, ProductKernel)
            return AdditiveKernel(fp_product_kernel, count_product_kernel)

        raise ValueError(f"Unknown mixing_method: {self.mixing_method}")



class GPMix(gpytorch.models.ExactGP):
    def __init__(self, X, y, feat_idx,
                  mixing_method:str, kernel_method:dict, likelihood, prior, ssk_parameters, cuda_avail,
                  ard: bool = True, outputscale: float | None = None):
        super().__init__(X, y, likelihood)
        # self.feat_idx = feat_idx
        # self.kernel_method = kernel_method
        # self.ssk_parameters = ssk_parameters
        self.mean_module = gpytorch.means.ZeroMean()

        self.kernel_builder = MixingKernel(
            feat_idx=feat_idx,
            mixing_method=mixing_method,
            kernel_method=kernel_method,
            ssk_parameters=ssk_parameters,
            cuda_avail=cuda_avail,
            ard=ard,
        )
        base_kernel  = self.kernel_builder.build()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        if outputscale is None:
            self.covar_module.register_prior(
                "variance_prior",
                gpytorch.priors.LogNormalPrior(0.0, 1.0),
                "outputscale",
            )
        else:
            # DOPE: hazel_gp held the signal variance at 1.0 instead of learning it.
            self.covar_module.outputscale = float(outputscale)
            self.covar_module.raw_outputscale.requires_grad_(False)
        if self.likelihood.noise_covar.raw_noise.requires_grad:
            self.likelihood.register_prior(
                "noise_prior",
                gpytorch.priors.LogNormalPrior(0.0, 1.0),
                "noise",
            )
        fp_keys = sorted(
            k for k in feat_idx if k.startswith("fp_") and feat_idx[k]
        )

        root_kernel = self.covar_module.base_kernel
        if prior:
            fp_has_lengthscale = bool(fp_keys) and kernel_method["fp"].lower() not in {
                "tanimoto",
                "ssk",
            }

            def _leaf_kernels(kernel):
                if isinstance(kernel, ScaleKernel):
                    return _leaf_kernels(kernel.base_kernel)
                if isinstance(kernel, (AdditiveKernel, ProductKernel)):
                    leaves = []
                    for sub_kernel in kernel.kernels:
                        leaves.extend(_leaf_kernels(sub_kernel))
                    return leaves
                return [kernel]

            def _register_fp_priors(kernels):
                if not fp_has_lengthscale:
                    return

                for i, sk in enumerate(kernels):
                    sk.register_prior(
                        f"fp_lengthscale_prior_{i}",
                        gpytorch.priors.GammaPrior(5.0, 5.0),
                        "lengthscale",
                    )

            def _register_count_priors(kernels):
                for i, sk in enumerate(kernels):
                    sk.register_prior(
                        f"count_lengthscale_prior_{i}",
                        gpytorch.priors.GammaPrior(5.0, 5.0),
                        "lengthscale",
                    )

            leaf_kernels = _leaf_kernels(root_kernel)
            _register_fp_priors(leaf_kernels[:len(fp_keys)])
            _register_count_priors(leaf_kernels[len(fp_keys):])
            

    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    

class CVProgressBar:
    """CV- and multi-bar-safe progress bar for GP MAP training."""

    def __init__(self, total_steps, disable=False, position=0, desc="MAP training"):
        # Auto-disable in CI or pytest parallel runs
        disable = disable or "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ

        self.pbar = tqdm(
            total=total_steps,
            desc=desc,
            leave=False,
            position=position,
            file=sys.stderr,
            disable=disable,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
        )

    def update(self, n=1, postfix=None):
        if postfix is not None:
            self.pbar.set_postfix(postfix)
        self.pbar.update(n)

    def close(self):
        self.pbar.close()



# DOPE: hazel_gp fit in float64; GP_collab fits in float32. Both are selectable.
TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}

# DOPE: a feature group may be an explicit column list, the string "__all__",
# or a callable taking the fold's column list. The one-hot width changes between
# folds (a held-out ligand drops its column), so groups that span the encoded
# matrix cannot be enumerated before the preprocessor is fitted.
def _resolve_group(cols, columns):
    if callable(cols):
        return list(cols(list(columns)))
    if cols == "__all__":
        return list(columns)
    return list(cols)


class GPytorchMAPRegressor:
    def __init__(
        self,
        feat_group: dict,
        lr=1e-2,
        n_epoch=400,
        random_state=42,
        kernel_mixing_method: str = "product",
        kernel_type: dict | None = None,
        ssk_parameters: dict | None = None,
        progbar: bool = True,
        prior=False,
        normalize_y: bool = True,
        use_cuda: bool = True,
        # DOPE: everything below is new and defaults to GP_collab's behaviour.
        ard: bool = True,
        dtype: str = "float32",
        noise: float | None = None,
        noise_floor: float = 1e-2,
        outputscale: float | None = None,
        train_jitter: float = 1e-1,
        predict_jitter: float = 1e-4,
        restarts: int = 1,
    ):
        self.feat_group = feat_group
        self.lr = lr
        self.n_epoch = n_epoch
        self.random_state = random_state
        self.kernel_mixing_method = kernel_mixing_method
        self.kernel_type = (
            kernel_type
            if kernel_type is not None
            else {"fp": "TanimotoRBF", "count": "Matern32"}
        )
        self.ssk_parameters = ssk_parameters
        self.progbar = progbar
        self.prior = prior
        self.normalize_y = normalize_y
        self.use_cuda= use_cuda
        self.ard = ard
        self.dtype = dtype
        self.noise = noise
        self.noise_floor = noise_floor
        self.outputscale = outputscale
        self.train_jitter = train_jitter
        self.predict_jitter = predict_jitter
        self.restarts = restarts
        device = torch.device(
            "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.cuda_avail = {"dtype": TORCH_DTYPES[dtype], "device": device}

    @property
    def y(self):
        """
        Return training targets on the original, unnormalized scale.
        """
        if not hasattr(self, "_y_train"):
            raise AttributeError("Training targets do not exist. Call fit() first.")

        y_original = self._y_train.detach().cpu().numpy() * self.y_std_ + self.y_mean_
        return np.asarray(y_original).ravel()

    @y.setter
    def y(self, y):
        """
        Store normalized y internally if normalize_y=True.
        """
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        y = np.asarray(y, dtype=float).reshape(-1)

        if self.normalize_y:
            self.y_mean_ = float(np.mean(y))
            self.y_std_ = float(np.std(y))

            if self.y_std_ == 0:
                self.y_std_ = 1.0

            y_scaled = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0.0
            self.y_std_ = 1.0
            y_scaled = y

        self._y_train = torch.as_tensor(
            y_scaled,
            **self.cuda_avail,
        ).view(-1)

    def _prepare_X(self, X: pd.DataFrame, fit: bool = False):
        """
        Convert a DataFrame X to a torch tensor.

        This model assumes X is always a pandas DataFrame.
        During fit, feature group names are mapped to column indices.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GPytorchMAPModel expects X to be a pandas DataFrame. "
                f"Got {type(X).__name__} instead."
            )

        if fit:
            self.feat_idx_ = {}

            # DOPE: "__all__" means every encoded column of this fold. The one-hot
            # width changes between folds (a held-out ligand drops its column), so
            # the group cannot be enumerated before the preprocessor is fitted.
            feat_group = {
                key: _resolve_group(cols, X.columns)
                for key, cols in self.feat_group.items()
            }
            self.feat_group = feat_group

            for key, cols in feat_group.items():
                if cols:
                    missing_cols = [c for c in cols if c not in X.columns]
                    if missing_cols:
                        raise ValueError(
                            f"Columns for feature group '{key}' are missing "
                            f"from X: {missing_cols}"
                        )

                    self.feat_idx_[key] = [
                        X.columns.get_loc(c) for c in cols
                    ]

            count_cols = self.feat_group.get("count", [])

            missing_count_cols = [
                c for c in count_cols if c not in X.columns
            ]
            if missing_count_cols:
                raise ValueError(
                    f"Count columns are missing from X: {missing_count_cols}"
                )

            self.count_feat_name_idx_ = {
                c: X.columns.get_loc(c)
                for c in count_cols
            }

            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        else:
            if not hasattr(self, "feature_names_in_"):
                raise RuntimeError("Model is not fitted. Call fit() first.")

            missing_cols = [
                c for c in self.feature_names_in_
                if c not in X.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Prediction X is missing columns seen during fit: {missing_cols}"
                )

            # Keep prediction column order identical to training.
            X = X.loc[:, self.feature_names_in_]

        return torch.as_tensor(
            X.to_numpy(),
            **self.cuda_avail,
        )

    def _build(self, X_train, y_train):
        """One seeded initialisation of the likelihood and GP."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self.noise_floor)
        ).to(**self.cuda_avail)
        if self.noise is not None:
            # DOPE: hazel_gp held the observation noise at 1e-6 in standardised
            # target units instead of learning it against a 1e-2 floor.
            likelihood.noise = float(self.noise)
            likelihood.noise_covar.raw_noise.requires_grad_(False)

        model = GPMix(
            X_train,
            y_train,
            self.feat_idx_,
            self.kernel_mixing_method,
            self.kernel_type,
            likelihood,
            prior=self.prior,
            ssk_parameters=self.ssk_parameters,
            cuda_avail=self.cuda_avail,
            ard=self.ard,
            outputscale=self.outputscale,
        ).to(**self.cuda_avail)
        return model, likelihood

    def _train_once(self, X_train, y_train, seed):
        """GP_collab's original training loop, run from one seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, likelihood = self._build(X_train, y_train)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        pbar = CVProgressBar(total_steps=self.n_epoch, position=0) if self.progbar else None
        loss = None
        for _ in range(self.n_epoch):
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(
                float_value=self.train_jitter,
                double_value=self.train_jitter,
            ), gpytorch.settings.cholesky_max_tries(5):
                y_pred = model(X_train)
                loss = -mll(y_pred, y_train)

            loss.backward()
            optimizer.step()

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return model, likelihood, float(loss.detach().cpu())

    def fit(self, X_train: pd.DataFrame, y_train):
        X_train = self._prepare_X(X_train, fit=True)

        # Calls the y setter.
        self.y = y_train
        y_train = self._y_train

        # DOPE: restarts>1 keeps the initialisation with the best *training*
        # objective, the way hazel_gp did. restarts=1 is GP_collab's single fit.
        best = None
        self.restart_losses_ = []
        for restart in range(max(1, int(self.restarts))):
            model, likelihood, loss = self._train_once(
                X_train, y_train, self.random_state + restart
            )
            self.restart_losses_.append(loss)
            if best is None or loss < best[2]:
                best = (model, likelihood, loss)

        self.gp_model_, self.likelihood_, self.train_loss_ = best
        self.is_fitted_ = True
        return self

    def predict(self, X_test: pd.DataFrame, return_std=False):
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_test = self._prepare_X(X_test, fit=False)

        self.gp_model_.eval()
        self.likelihood_.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            float_value=self.predict_jitter,
            double_value=self.predict_jitter,
        ), gpytorch.settings.cholesky_max_tries(5):
            posterior = self.likelihood_(self.gp_model_(X_test))

            y_pred_scaled = posterior.mean.detach().cpu().numpy()
            y_std_scaled = (
                posterior.stddev.detach().cpu().numpy()
                if return_std
                else None
            )

        y_pred = y_pred_scaled * self.y_std_ + self.y_mean_

        if return_std:
            y_std = y_std_scaled * self.y_std_
        else:
            y_std = None

        return {
            "y_pred": np.asarray(y_pred).ravel(),
            "y_std": np.asarray(y_std).ravel() if y_std is not None else None,
        }

    def _get_lengthscale(self):
        if not hasattr(self, "gp_model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")

        summary = {}

        root_kernel = self.gp_model_.covar_module.base_kernel

        fp_keys = sorted(
            k for k in self.feat_idx_.keys()
            if k.startswith("fp_") and self.feat_idx_[k]
        )

        count_names = [
            name
            for name, _ in sorted(
                self.count_feat_name_idx_.items(),
                key=lambda item: item[1],
            )
        ]

        def _extract_ls(kernel, key_prefix):
            if not hasattr(kernel, "lengthscale") or kernel.lengthscale is None:
                raise RuntimeError(
                    f"No lengthscale found for kernel '{key_prefix}'"
                )

            ls = kernel.lengthscale.detach().cpu().numpy()

            if ls.shape[-1] > 1:
                return {
                    f"{key_prefix}[{i}]": float(ls_i)
                    for i, ls_i in enumerate(ls.squeeze(0))
                }

            return {
                key_prefix: float(np.asarray(ls).squeeze())
            }

        def _leaf_kernels(kernel):
            if isinstance(kernel, ScaleKernel):
                return _leaf_kernels(kernel.base_kernel)
            if isinstance(kernel, (AdditiveKernel, ProductKernel)):
                leaves = []
                for sub_kernel in kernel.kernels:
                    leaves.extend(_leaf_kernels(sub_kernel))
                return leaves
            return [kernel]

        def _add_fp_lengthscales(kernels):
            if len(kernels) > len(fp_keys):
                raise IndexError(
                    "More fingerprint sub-kernels were found than expected "
                    "from `feat_idx_`."
                )

            for i, sk in enumerate(kernels):
                fp_key = fp_keys[i]

                if self.kernel_type["fp"].lower() in {"tanimoto", "ssk"}:
                    summary[fp_key] = None
                else:
                    summary.update(_extract_ls(sk, fp_key))

        def _add_count_lengthscales(kernels):
            if len(kernels) > len(count_names):
                raise IndexError(
                    "More count sub-kernels were found than expected from "
                    "`count_feat_name_idx_`."
                )

            for i, sk in enumerate(kernels):
                count_key = count_names[i]
                summary.update(_extract_ls(sk, count_key))

        leaf_kernels = _leaf_kernels(root_kernel)
        _add_fp_lengthscales(leaf_kernels[:len(fp_keys)])
        _add_count_lengthscales(leaf_kernels[len(fp_keys):])

        return summary
    

class GPytorchMAPsklearnRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        feat_group: dict,
        lr=1e-2,
        n_epoch=400,
        random_state=42,
        kernel_mixing_method: str = "product",
        kernel_type: dict | None = None,
        ssk_parameters: dict | None = None,
        progbar: bool = True,
        prior=False,
        normalize_y: bool = True,
        use_cuda: bool = True,
        ard: bool = True,
        dtype: str = "float32",
        noise: float | None = None,
        noise_floor: float = 1e-2,
        outputscale: float | None = None,
        train_jitter: float = 1e-1,
        predict_jitter: float = 1e-4,
        restarts: int = 1,
    ):
        self.feat_group = feat_group
        self.lr = lr
        self.n_epoch = n_epoch
        self.random_state = random_state
        self.kernel_mixing_method = kernel_mixing_method
        self.kernel_type = kernel_type
        self.ssk_parameters = ssk_parameters
        self.progbar = progbar
        self.prior = prior
        self.normalize_y = normalize_y
        self.use_cuda = use_cuda
        self.ard = ard
        self.dtype = dtype
        self.noise = noise
        self.noise_floor = noise_floor
        self.outputscale = outputscale
        self.train_jitter = train_jitter
        self.predict_jitter = predict_jitter
        self.restarts = restarts

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GPytorchMAPRegressor expects X to be a pandas DataFrame. "
                f"Got {type(X).__name__} instead."
            )

        self.regressor_ = GPytorchMAPRegressor(
            feat_group=self.feat_group,
            lr=self.lr,
            n_epoch=self.n_epoch,
            random_state=self.random_state,
            kernel_mixing_method=self.kernel_mixing_method,
            kernel_type=self.kernel_type,
            ssk_parameters=self.ssk_parameters,
            progbar=self.progbar,
            prior=self.prior,
            normalize_y=self.normalize_y,
            use_cuda=self.use_cuda,
            ard=self.ard,
            dtype=self.dtype,
            noise=self.noise,
            noise_floor=self.noise_floor,
            outputscale=self.outputscale,
            train_jitter=self.train_jitter,
            predict_jitter=self.predict_jitter,
            restarts=self.restarts,
        )

        self.regressor_.fit(X, y)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def predict(self, X, return_std=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GPytorchMAPRegressor expects X to be a pandas DataFrame. "
                f"Got {type(X).__name__} instead."
            )

        return self.regressor_.predict(X, return_std=return_std)

    def _get_lengthscale(self):
        return self.regressor_._get_lengthscale()





