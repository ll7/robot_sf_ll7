"""Compatibility tests for external-dataset exceptions under ``RobotSfError``."""

import pytest

from robot_sf.data.external.atc import AtcDataError
from robot_sf.data.external.crowdbot import CrowdBotDataError
from robot_sf.data.external.eth_ucy import EthUcyDataError
from robot_sf.data.external.ind import IndDataError
from robot_sf.data.external.recording_shape_contract import RecordingDatasetError
from robot_sf.data.external.scand import ScandDataError
from robot_sf.data.external.socnavbench_eth import SocNavBenchEthDataError
from robot_sf.errors import RobotSfError

_EXTERNAL_DATA_ERROR_TYPES = (
    AtcDataError,
    CrowdBotDataError,
    EthUcyDataError,
    IndDataError,
    RecordingDatasetError,
    ScandDataError,
    SocNavBenchEthDataError,
)


@pytest.mark.parametrize("error_type", _EXTERNAL_DATA_ERROR_TYPES)
def test_external_data_errors_support_shared_and_builtin_catches(
    error_type: type[Exception],
) -> None:
    """Each public error remains a RuntimeError while gaining the shared base."""

    assert issubclass(error_type, RobotSfError)
    assert issubclass(error_type, RuntimeError)

    with pytest.raises(RobotSfError):
        raise error_type("shared catch")

    with pytest.raises(RuntimeError):
        raise error_type("builtin catch")
