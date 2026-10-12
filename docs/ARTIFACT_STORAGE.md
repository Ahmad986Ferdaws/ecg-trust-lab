# Publishing prediction and audit artifacts

The prediction and audit-array writers publish a compressed NPZ archive and a
JSON sidecar. The sidecar records archive size, array signatures and hashes.
Readers verify both files before returning an artifact. The data formats and
scientific identities are unchanged by the publication mechanism.

## Filesystem requirements

Writers stage complete files in the destination directory and publish each
through a hard link that fails if the filename is occupied. Use a filesystem
that supports local hard links. An unsupported filesystem raises an error;
the writer does not fall back to replacing an existing file or streaming into
a final filename. See Python's [os.link documentation](https://docs.python.org/3.12/library/os.html#os.link).

Existing archives, sidecars, directories and dangling destination symlinks are
refused. If two producers race for the same pair, only one can claim the archive
name. The loser receives `FileExistsError` and must not remove or retry over the
winner's files. Verify the existing pair using its expected protocol and
identities before deciding whether it is the intended output.

## Failure and recovery

Each file appears complete, but the two-file pair is **not one atomic filesystem
transaction**. Readers can observe an archive before the sidecar exists and
must refuse it until both verify. An abrupt process or host failure may leave
an incomplete pair. This mechanism does not promise power-loss durability or
provide a backup service.

On an ordinary staging/publication exception, the writer cleans its temporary
files and attempts to remove an archive it just published if it still owns that
filename. It preserves a conflicting sidecar. If cleanup cannot complete,
retain the error and inspect the directory; do not interpret leftover output as
successful evidence.

For an incomplete or conflicting output, preserve the existing files and record
the attempted command and expected identities. Have the responsible operator
determine their provenance. Choose a new authorized destination when appropriate.
Never delete a sealed run or rerun a one-shot cohort merely to clear a storage
error. Follow the relevant protocol's recovery and ledger rules.

These guarantees concern cooperating producers using these APIs in a controlled
directory. Hash verification still matters: filesystem permissions, trusted
input identities, backups and protection from other local processes remain
separate operational responsibilities.
