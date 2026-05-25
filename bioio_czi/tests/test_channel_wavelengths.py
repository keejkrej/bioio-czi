from pathlib import Path

import pytest
from bioio_base.dimensions import DimensionNames

from bioio_czi import Reader

LNP_LSM_CZI = Path.home() / "data" / "lnp_lsm" / "lnp_lsm.czi"


@pytest.mark.parametrize(
    "path, expected_emission, expected_excitation",
    [
        pytest.param(
            LNP_LSM_CZI,
            (565.0, 422.0),
            (543.0, 401.0),
            marks=pytest.mark.skipif(
                not LNP_LSM_CZI.is_file() or LNP_LSM_CZI.stat().st_size < 1_000_000,
                reason="local lnp_lsm.czi sample not available",
            ),
        ),
    ],
)
def test_channel_wavelengths(
    path: Path,
    expected_emission: tuple[float, ...],
    expected_excitation: tuple[float, ...],
) -> None:
    reader = Reader(path)
    assert reader.channel_emission_wavelengths == expected_emission
    assert reader.channel_excitation_wavelengths == expected_excitation

    xarr = reader.xarray_dask_data
    c_coord = xarr.coords[DimensionNames.Channel]
    assert c_coord.attrs["emission_wavelength_nm"] == list(expected_emission)
    assert c_coord.attrs["excitation_wavelength_nm"] == list(expected_excitation)

    assert reader.dimension_properties.C.type == "channel"
    assert str(reader.dimension_properties.C.unit) == "nanometer"
