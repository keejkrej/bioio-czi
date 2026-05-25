import logging
from typing import Any, Dict, Literal, Optional
from xml.etree import ElementTree as ET

import xarray as xr
from bioio_base.dimensions import DimensionNames

from bioio_czi.bounding_box import size

from .metadata import generate_ome_channel_id

log = logging.getLogger(__name__)

ChannelWavelengthKind = Literal["emission", "excitation"]

_WAVELENGTH_TAGS: dict[ChannelWavelengthKind, str] = {
    "emission": "EmissionWavelength",
    "excitation": "ExcitationWavelength",
}

_COORD_ATTRS: dict[ChannelWavelengthKind, str] = {
    "emission": "emission_wavelength_nm",
    "excitation": "excitation_wavelength_nm",
}


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


def get_channel_wavelengths(
    xml: ET.Element,
    scene_index: int,
    dims_shape: Dict[str, Any],
    kind: ChannelWavelengthKind,
) -> Optional[list[Optional[float]]]:
    """
    Get per-channel wavelengths (nm) for the given scene index.

    Wavelengths are read from ``EmissionWavelength`` or ``ExcitationWavelength``
    on each channel in ``Image/Dimensions/Channels``, in the same order as
    :func:`get_channel_names`.
    """
    tag = _WAVELENGTH_TAGS[kind]
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
        element = channel.find(tag)
        if element is not None and element.text is not None:
            try:
                scene_wavelength_list.append(float(element.text))
            except ValueError:
                log.warning(
                    "Invalid %s %r for channel %s",
                    tag,
                    element.text,
                    channel.attrib.get("Id"),
                )
                scene_wavelength_list.append(None)
        else:
            scene_wavelength_list.append(None)

    return scene_wavelength_list


def get_channel_emission_wavelengths(
    xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
) -> Optional[list[Optional[float]]]:
    """Per-channel emission wavelengths (nm); see :func:`get_channel_wavelengths`."""
    return get_channel_wavelengths(xml, scene_index, dims_shape, "emission")


def get_channel_excitation_wavelengths(
    xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
) -> Optional[list[Optional[float]]]:
    """Per-channel excitation wavelengths (nm); see :func:`get_channel_wavelengths`."""
    return get_channel_wavelengths(xml, scene_index, dims_shape, "excitation")


def attach_channel_wavelength_coord_attrs(
    data_array: xr.DataArray,
    emission_wavelengths_nm: Optional[list[Optional[float]]] = None,
    excitation_wavelengths_nm: Optional[list[Optional[float]]] = None,
) -> xr.DataArray:
    """
    Attach OME-aligned ``emission_wavelength_nm`` and ``excitation_wavelength_nm``
    to the channel coordinate when present.
    """
    if DimensionNames.Channel not in data_array.coords:
        return data_array

    for kind, wavelengths in (
        ("emission", emission_wavelengths_nm),
        ("excitation", excitation_wavelengths_nm),
    ):
        if wavelengths is None or not any(w is not None for w in wavelengths):
            continue
        data_array.coords[DimensionNames.Channel].attrs[_COORD_ATTRS[kind]] = (
            wavelengths
        )

    return data_array
