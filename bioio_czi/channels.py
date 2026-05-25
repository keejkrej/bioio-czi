import logging
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET

import xarray as xr
from bioio_base.dimensions import DimensionNames

from bioio_czi.bounding_box import size

from .metadata import generate_ome_channel_id

log = logging.getLogger(__name__)


def get_channel_names(
    xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
) -> Optional[list[str]]:
    """
    Get the channel names for the given scene index.

    Parameters
    ----------
    metadata: xml.etree.ElementTree.Element
        The metadata to search for channel names.
    scene_index: int
    """
    # Get all images
    img_sets = xml.findall(".//Image/Dimensions/Channels")

    if len(img_sets) == 0:
        return None

    # Select the current scene
    img = img_sets[0]
    if scene_index < len(img_sets):
        img = img_sets[scene_index]

    # Construct channel name list
    scene_channel_list = []
    channels = img.findall("./Channel")
    number_of_channels_in_data = size(dims_shape, DimensionNames.Channel)

    # There may be more channels in the metadata than in the data
    # if so, we will just use the first N channels and log
    # a warning to the user
    if len(channels) > number_of_channels_in_data:
        log.warning(
            "More channels in metadata than in data "
            f"({len(channels)} vs. {number_of_channels_in_data})"
        )

    for i, channel in enumerate(channels[:number_of_channels_in_data]):
        # Id is required, Name is not.
        # But we prefer to use Name if it is present
        channel_name = channel.attrib.get("Name")
        channel_id = channel.attrib.get("Id")
        if channel_name is None:
            # Idea: we could try to find a channel name from
            # DisplaySetting/Channels/Channel
            channel_name = channel_id
        if channel_name is None:
            # This is actually an error because Id was required by the spec
            channel_name = generate_ome_channel_id(str(scene_index), str(i))

        scene_channel_list.append(channel_name)
    return scene_channel_list


def get_channel_emission_wavelengths(
    xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
) -> Optional[list[Optional[float]]]:
    """
    Get per-channel emission wavelengths (nm) for the given scene index.

    Wavelengths are read from ``EmissionWavelength`` on each channel in
    ``Image/Dimensions/Channels``, in the same order as :func:`get_channel_names`.
    """
    img_sets = xml.findall(".//Image/Dimensions/Channels")
    if len(img_sets) == 0:
        return None

    img = img_sets[0]
    if scene_index < len(img_sets):
        img = img_sets[scene_index]

    channels = img.findall("./Channel")
    number_of_channels_in_data = size(dims_shape, DimensionNames.Channel)

    if len(channels) > number_of_channels_in_data:
        log.warning(
            "More channels in metadata than in data "
            f"({len(channels)} vs. {number_of_channels_in_data})"
        )

    scene_wavelength_list: list[Optional[float]] = []
    for channel in channels[:number_of_channels_in_data]:
        emission = channel.find("EmissionWavelength")
        if emission is not None and emission.text is not None:
            try:
                scene_wavelength_list.append(float(emission.text))
            except ValueError:
                log.warning(
                    "Invalid EmissionWavelength %r for channel %s",
                    emission.text,
                    channel.attrib.get("Id"),
                )
                scene_wavelength_list.append(None)
        else:
            scene_wavelength_list.append(None)

    return scene_wavelength_list


def attach_channel_emission_wavelength_coord_attrs(
    data_array: xr.DataArray,
    emission_wavelengths_nm: Optional[list[Optional[float]]],
) -> xr.DataArray:
    """
    Attach ``emission_wavelength_nm`` to the channel coordinate when present.
    """
    if (
        emission_wavelengths_nm is None
        or DimensionNames.Channel not in data_array.coords
        or not any(w is not None for w in emission_wavelengths_nm)
    ):
        return data_array

    data_array.coords[DimensionNames.Channel].attrs["emission_wavelength_nm"] = (
        emission_wavelengths_nm
    )
    return data_array
