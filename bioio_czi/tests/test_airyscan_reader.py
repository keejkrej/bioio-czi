"""Tests for reading Airyscan raw detector planes (CZI H dimension).

Related: https://github.com/bioio-devs/bioio-czi/issues/72
Test file: https://zenodo.org/records/17423258
"""

from pathlib import Path
from typing import Tuple

import pytest
import requests

from bioio_czi import Reader

AIRYSCAN_ZENODO_URL = (
    "https://zenodo.org/api/records/17423258/files/8bit_hs-airy.czi/content"
)


@pytest.fixture(scope="module", name="airyscan_czi_path")
def _airyscan_czi_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("airyscan") / "8bit_hs-airy.czi"
    if not path.exists():
        response = requests.get(AIRYSCAN_ZENODO_URL, timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


@pytest.mark.parametrize(
    "use_aicspylibczi, expected_dims_order, expected_shape",
    [
        pytest.param(
            False,
            "HCYX",
            (32, 6, 1119, 1119),
            id="pylibczirw",
        ),
        pytest.param(
            True,
            "HTCZYX",
            (32, 1, 6, 1, 1119, 1119),
            id="aicspylibczi",
        ),
    ],
)
def test_airyscan_reader_exposes_h_dimension(
    airyscan_czi_path: Path,
    use_aicspylibczi: bool,
    expected_dims_order: str,
    expected_shape: Tuple[int, ...],
) -> None:
    reader = Reader(airyscan_czi_path, use_aicspylibczi=use_aicspylibczi)

    assert reader.dims.order == expected_dims_order
    assert reader.shape == expected_shape
    assert reader.xarray_dask_data.dims == tuple(expected_dims_order)
    assert reader.xarray_dask_data.shape == expected_shape
    assert "H" in reader.dims.order
    assert reader.dims.H == 32


@pytest.mark.parametrize(
    "use_aicspylibczi",
    [False, True],
    ids=["pylibczirw", "aicspylibczi"],
)
def test_airyscan_detector_and_sum_channels(
    airyscan_czi_path: Path, use_aicspylibczi: bool
) -> None:
    reader = Reader(airyscan_czi_path, use_aicspylibczi=use_aicspylibczi)
    da = reader.xarray_dask_data

    if use_aicspylibczi:
        detectors = da.isel(C=0, T=0, Z=0).compute().values
        sum_channel = da.isel(C=1, T=0, Z=0, H=0).compute().values
    else:
        detectors = da.isel(C=0).compute().values
        sum_channel = da.isel(C=1, H=0).compute().values
        nonzero_sum_h = [
            h
            for h in range(da.sizes["H"])
            if da.isel(C=1, H=h).compute().values.max() > 0
        ]
        assert nonzero_sum_h == [0]

    assert detectors.shape == (32, da.sizes["Y"], da.sizes["X"])
    assert sum_channel.shape == (da.sizes["Y"], da.sizes["X"])
    assert float(sum_channel.max()) > 0
    assert detectors.max(axis=(1, 2)).min() > 0
