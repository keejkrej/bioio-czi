from pathlib import Path

import pytest
from bioio_base.dimensions import DimensionNames

from bioio_czi import Reader

LNP_LSM_CZI = Path.home() / "data" / "lnp_lsm" / "lnp_lsm.czi"


@pytest.mark.parametrize(
    "path, expected",
    [
        pytest.param(
            LNP_LSM_CZI,
            (565.0, 422.0),
            marks=pytest.mark.skipif(
                not LNP_LSM_CZI.is_file() or LNP_LSM_CZI.stat().st_size < 1_000_000,
                reason="local lnp_lsm.czi sample not available",
            ),
        ),
    ],
)
def test_channel_emission_wavelengths(path: Path, expected: tuple[float, ...]) -> None:
    reader = Reader(path)
    assert reader.channel_emission_wavelengths == expected

    xarr = reader.xarray_dask_data
    assert xarr.coords[DimensionNames.Channel].attrs["emission_wavelength_nm"] == list(
        expected
    )

    assert reader.dimension_properties.C.type == "emission_wavelength"
    assert str(reader.dimension_properties.C.unit) == "nanometer"
