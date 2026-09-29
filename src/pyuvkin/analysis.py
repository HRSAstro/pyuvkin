"""The PyAutoFit `Analysis`: parameters -> cube -> visibilities -> likelihood.

    log L = -1/2 [ sum_k ( (Re r_k / sigma_re,k)^2 + (Im r_k / sigma_im,k)^2 )
                   + sum_k log(2 pi sigma_re,k^2) + log(2 pi sigma_im,k^2) ]

over every unflagged visibility of every channel, with ``r = data - model``.
The noise normalisation is a constant, so the likelihood differs from
``-chi^2 / 2`` by a constant; both are reported.
"""

from __future__ import annotations

import logging

import numpy as np

import autofit as af

from .models.base import Renderer
from .models.parameters import DiscParameters
from .transform import CubeTransformer

logger = logging.getLogger("pyuvkin")


class KinematicAnalysis(af.Analysis):
    def __init__(self, transformer: CubeTransformer, renderer: Renderer):
        super().__init__(use_jax=False)
        self.transformer = transformer
        self.renderer = renderer
        self.n_evaluations = 0

    # ------------------------------------------------------------- forward
    def cube_from(self, instance: DiscParameters) -> np.ndarray:
        return self.renderer.cube(instance)

    def model_visibilities_from(self, instance: DiscParameters) -> np.ndarray:
        return self.transformer.model_visibilities(self.cube_from(instance))

    # ---------------------------------------------------------- likelihood
    def log_likelihood_function(self, instance) -> float:
        self.n_evaluations += 1
        try:
            model_vis = self.model_visibilities_from(instance)
        except FloatingPointError as e:
            logger.debug("non-finite model: %s", e)
            raise af.exc.FitException from e
        ll = self.transformer.log_likelihood(model_vis)
        if not np.isfinite(ll):
            raise af.exc.FitException("non-finite log likelihood")
        return float(ll)

    def fit_summary(self, instance: DiscParameters) -> dict:
        """chi^2 and friends for one parameter set."""
        model_vis = self.model_visibilities_from(instance)
        chi2 = self.transformer.chi_squared(model_vis)
        n = self.transformer.n_data
        return {
            "log_likelihood": self.transformer.log_likelihood(model_vis),
            "chi_squared": chi2,
            "n_data": int(n),
            "chi_squared_reduced": chi2 / n,
            "chi_squared_per_channel": self.transformer.chi_squared_per_channel(model_vis).tolist(),
        }

    # autofit calls these during a search; there is nothing to draw per step
    def visualize(self, paths, instance, during_analysis):
        return None

    def visualize_before_fit(self, paths, model):
        return None
