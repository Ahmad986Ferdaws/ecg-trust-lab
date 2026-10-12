"""No-replace publication of staged archive/sidecar artifact pairs."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path


def publish_new_pair(
    archive_temporary: Path,
    archive_destination: Path,
    sidecar_temporary: Path,
    sidecar_destination: Path,
) -> None:
    """Publish complete files without replacing any existing destination.

    A hard link reserves each filename atomically, so concurrent cooperating
    writers cannot replace one another's evidence. A sidecar failure rolls back
    this writer's archive when it still owns that destination. Filesystems
    without hard-link support fail closed. Callers always clean up staging files.

    The pair is not a filesystem transaction: an abrupt process/host failure can
    leave an archive without a sidecar. Readers must continue verifying both.
    """

    os.link(archive_temporary, archive_destination)
    try:
        os.link(sidecar_temporary, sidecar_destination)
    except BaseException:
        with suppress(OSError):
            if archive_destination.samefile(archive_temporary):
                archive_destination.unlink()
        raise
