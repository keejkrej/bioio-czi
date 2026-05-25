import logging
from datetime import timedelta
from typing import Optional
from xml.etree.ElementTree import Element

log = logging.getLogger(__name__)


def position_index(scene: str) -> Optional[int]:
    """
    Extracts the numeric position index from the current scene name.
    Returns
    -------
    Optional[int]
        The numeric part of the scene name.
        Returns None if parsing fails.
    """
    try:
        prefix = scene.split("-")[0]
        if prefix.startswith("Image:"):
            return int(prefix.removeprefix("Image:"))
        if len(prefix) > 1 and prefix[0].isalpha():
            return int(prefix[1:])
        return int(prefix)
    except (IndexError, ValueError) as exc:
        log.debug(
            "Could not parse position index from scene name '%s': %s",
            scene,
            exc,
        )
    except Exception as exc:
        log.warning("Unexpected error parsing position index: %s", exc, exc_info=True)

    return None


def _row_or_column(
    metadata: Element, current_scene_index: int, row_or_column: str
) -> Optional[str]:
    """
    Extracts the well row or index for the current scene.
    Returns
    -------
    Optional[str]
        The column index as a string. Returns None if not found.
    """
    try:
        scenes = metadata.findall(
            "Metadata/Information/Image/Dimensions/S/Scenes/Scene"
        )
        for scene in scenes:
            scene_index = scene.get("Index")
            if scene_index is not None and int(scene_index) == current_scene_index:
                shape = scene.find("Shape")
                if shape is not None:
                    index = shape.find(
                        "RowIndex" if row_or_column == "row" else "ColumnIndex"
                    )
                    if index is not None:
                        return index.text
    except Exception as exc:
        log.warning(
            f"Failed to extract well {row_or_column} index: %s", exc, exc_info=True
        )

    return None


def column(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well column index for the current scene.
    Returns
    -------
    Optional[str]
        The column index as a string. Returns None if not found.
    """
    return _row_or_column(metadata, current_scene_index, "column")


def time_interval(metadata: Element) -> Optional[timedelta]:
    """
    Extract the timelapse interval from ``Dimensions.T.Positions.Interval.Increment``.

    Returns
    -------
    Optional[timedelta]
        Timelapse interval in seconds when present in the metadata XML.
    """
    increment_paths = (
        "Metadata/Information/Image/Dimensions/T/Positions/Interval/Increment",
        "Information/Image/Dimensions/T/Positions/Interval/Increment",
    )

    try:
        for path in increment_paths:
            increment = metadata.find(path)
            if increment is not None and increment.text is not None:
                return timedelta(seconds=float(increment.text))
    except Exception as exc:
        log.warning("Failed to extract timelapse interval: %s", exc, exc_info=True)

    return None


def row(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well row index for the current scene.
    Returns
    -------
    Optional[str]
        The row index as a string. Returns None if not found.
    """
    return _row_or_column(metadata, current_scene_index, "row")
