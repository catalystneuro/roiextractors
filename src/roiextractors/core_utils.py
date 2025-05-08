"""Class for some core utilities."""


def _convert_seconds_to_str(seconds):
    """Convert seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _convert_bytes_to_str(size_in_bytes):
    """
    Convert bytes to a human-readable string.

    Convert bytes to a human-readable string using IEC binary prefixes (KiB, MiB, GiB).
    Note that RAM memory is typically measured in IEC binary prefixes  while disk storage is typically
    measured in SI binary prefixes.
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes}B"
    elif size_in_bytes < 1024 * 1024:
        size_kb = size_in_bytes / 1024
        return f"{size_kb:.1f}KiB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        size_mb = size_in_bytes / (1024 * 1024)
        return f"{size_mb:.1f}MiB"
    else:
        size_gb = size_in_bytes / (1024 * 1024 * 1024)
        return f"{size_gb:.1f}GiB"
