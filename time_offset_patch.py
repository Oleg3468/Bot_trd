from pybit import _helpers
import time as _time

_patched = False


def _get_server_offset_ms(client):
    try:
        resp = client.get_server_time()
        server_sec = int(resp["result"]["timeSecond"])
        server_ms = server_sec * 1000
        local_ms = int(_time.time() * 1000)
        return server_ms - local_ms
    except Exception:
        return 0


def apply_time_offset_patch(client):
    global _patched
    offset = _get_server_offset_ms(client)
    if not _patched:
        _orig = _helpers.generate_timestamp
        def _patched_timestamp():
            return _orig() + offset
        _helpers.generate_timestamp = _patched_timestamp
        _patched = True
    return offset
