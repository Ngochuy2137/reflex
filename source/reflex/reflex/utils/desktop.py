"""Check desktop access before starting Isaac Sim, without reading X11 cookies."""

import ctypes
import os
import sys


def check_desktop():
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('DISPLAY is missing in this shell')
    library = ctypes.CDLL('libX11.so.6')
    library.XOpenDisplay.argtypes = [ctypes.c_char_p]
    library.XOpenDisplay.restype = ctypes.c_void_p
    display = library.XOpenDisplay(None)
    if not display:
        raise RuntimeError('Cannot connect to the X11 desktop; its authorization may have expired')
    library.XCloseDisplay.argtypes = [ctypes.c_void_p]
    library.XCloseDisplay(display)


if __name__ == '__main__':
    try:
        check_desktop()
    except (RuntimeError, OSError) as error:
        print(f'[GUI] {error}', file=sys.stderr)
        print('[GUI] On the host, from the Reflex root, run: bash docker/run.sh --gui', file=sys.stderr)
        print('[GUI] Then run the demo again inside that container shell.', file=sys.stderr)
        sys.exit(1)
    print('[GUI] Desktop connection is ready.', flush=True)
