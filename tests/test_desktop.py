"""Test desktop preflight without opening a window or reading credentials."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/reflex'))
from reflex.utils.desktop import check_desktop


class DesktopTests(unittest.TestCase):
    def test_missing_display_stops_before_loading_x11(self):
        with patch.dict(os.environ, {}, clear=True), patch('reflex.utils.desktop.ctypes.CDLL') as load:
            with self.assertRaisesRegex(RuntimeError, 'DISPLAY is missing'):
                check_desktop()
            load.assert_not_called()

    def test_rejected_connection_fails_early(self):
        library = Mock()
        library.XOpenDisplay.return_value = None
        with patch.dict(os.environ, {'DISPLAY': ':1'}), \
                patch('reflex.utils.desktop.ctypes.CDLL', return_value=library):
            with self.assertRaisesRegex(RuntimeError, 'authorization may have expired'):
                check_desktop()
        library.XCloseDisplay.assert_not_called()

    def test_connection_is_closed_after_check(self):
        library = Mock()
        library.XOpenDisplay.return_value = 123
        with patch.dict(os.environ, {'DISPLAY': ':1'}), \
                patch('reflex.utils.desktop.ctypes.CDLL', return_value=library):
            check_desktop()
        library.XCloseDisplay.assert_called_once_with(123)


if __name__ == '__main__':
    unittest.main()
