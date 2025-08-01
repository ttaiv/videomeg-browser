"""Contains helper for video and audio browser GUI."""

import logging
from importlib.resources import files

from qtpy.QtGui import QPixmap

logger = logging.getLogger(__name__)


def load_icon_pixmap(file_name: str) -> QPixmap | None:
    """Load an icon pixmap from icons directory.

    Parameters
    ----------
    file_name : str
        The name of the icon file to load.
        This should be just the file name with file extension, e.g., "icon.png".

    Returns
    -------
    QPixmap | None
        The loaded QPixmap if successful, or None if the file was not found.
    """
    icons_module = "videomeg_browser.icons"
    pixmap = QPixmap()
    resource = files(icons_module).joinpath(file_name)
    try:
        with resource.open("rb") as file:
            pixmap.loadFromData(file.read())
        return pixmap
    except (FileNotFoundError, ModuleNotFoundError) as e:
        logger.warning(f"Icon '{file_name}' not found from '{icons_module}': {e}")
        return None
