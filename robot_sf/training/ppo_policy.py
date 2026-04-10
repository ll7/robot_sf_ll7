"""PPO policy variants used by staged PPO improvement work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch as th
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from torch import nn

from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor

if TYPE_CHECKING:
    from gymnasium import spaces
    from stable_baselines3.common.type_aliases import Schedule

_PRIVILEGED_STATE_KEY = "critic_privileged_state"


class AsymmetricGridSocNavPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy with a separate privileged critic feature extractor.

    The actor consumes the standard SocNav + grid observation surface. The critic
    receives the same surface plus a `critic_privileged_state` vector that carries
    extra simulator state used only for value estimation.
    """

    def __init__(  # noqa: PLR0913
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[GridSocNavExtractor] = GridSocNavExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        critic_features_extractor_kwargs: dict[str, Any] | None = None,
        asymmetric_critic: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an asymmetric actor-critic policy for grid + SocNav observations."""
        self._asymmetric_critic = bool(asymmetric_critic)
        self._critic_features_extractor_kwargs = dict(critic_features_extractor_kwargs or {})
        self._features_extractor_call_count = 0
        share_features_extractor = not self._asymmetric_critic
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_features_extractor(self):  # type: ignore[override]
        """Create actor and critic extractors, using privileged state only for the critic.

        Returns:
            BaseFeaturesExtractor: Actor or critic features extractor instance.
        """
        if not self._asymmetric_critic:
            return super().make_features_extractor()

        self._features_extractor_call_count += 1
        if self._features_extractor_call_count == 1:
            actor_kwargs = dict(self.features_extractor_kwargs)
            actor_kwargs.setdefault("privileged_state_key", _PRIVILEGED_STATE_KEY)
            actor_kwargs.setdefault("include_privileged_state", False)
            return self.features_extractor_class(self.observation_space, **actor_kwargs)

        critic_kwargs = dict(self.features_extractor_kwargs)
        critic_kwargs.update(self._critic_features_extractor_kwargs)
        critic_kwargs.setdefault("privileged_state_key", _PRIVILEGED_STATE_KEY)
        critic_kwargs.setdefault("include_privileged_state", True)
        return self.features_extractor_class(self.observation_space, **critic_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Persist asymmetric critic wiring when saving/loading the policy.

        Returns:
            dict[str, Any]: Constructor parameters including asymmetric-critic metadata.
        """
        data = super()._get_constructor_parameters()
        data.update(
            {
                "critic_features_extractor_kwargs": dict(self._critic_features_extractor_kwargs),
                "asymmetric_critic": bool(self._asymmetric_critic),
            }
        )
        return data
