"""PPO policy variants used by staged PPO improvement work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.type_aliases import Schedule
    from torch import nn

    from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor

_PRIVILEGED_STATE_KEY = "critic_privileged_state"


def _init_classes() -> dict[str, Any]:
    import torch as th  # noqa: PLC0415
    from stable_baselines3.common.policies import (  # noqa: PLC0415
        MultiInputActorCriticPolicy,
    )
    from torch import nn  # noqa: PLC0415

    from robot_sf.feature_extractors.grid_socnav_extractor import (  # noqa: PLC0415
        GridSocNavExtractor,
    )

    class AsymmetricGridSocNavPolicy(MultiInputActorCriticPolicy):
        """SB3 PPO policy with a separate privileged critic feature extractor."""

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

        def make_features_extractor(self):
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
            data = super()._get_constructor_parameters()
            data.update(
                {
                    "critic_features_extractor_kwargs": dict(
                        self._critic_features_extractor_kwargs
                    ),
                    "asymmetric_critic": bool(self._asymmetric_critic),
                }
            )
            return data

    return {"AsymmetricGridSocNavPolicy": AsymmetricGridSocNavPolicy}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"AsymmetricGridSocNavPolicy"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
