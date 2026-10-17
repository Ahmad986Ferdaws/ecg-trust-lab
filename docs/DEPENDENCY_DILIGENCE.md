# Inventory dependencies for distribution review

Run the inventory in the environment that will actually be distributed:

```bash
python scripts/dependency_inventory.py > dependency-inventory.json
```

It lists installed Python distributions, versions, declared license expressions,
short legacy license descriptions, license classifiers and whether license files
are declared. It does not import package code or read source-URL records. Output
is sorted by normalized package name; conflicting duplicate distributions stop
the command instead of producing an incomplete-looking success.

The report includes development tools if they are installed. It is **not** a
complete software bill of materials or a license approval. No package archive
hashes, dependency graph, transitive native libraries, JavaScript packages,
model weights or dataset rights are inventoried. Review the inventory before
sharing it; it reveals the environment's package names and versions.

The [Python metadata interface](https://docs.python.org/3.12/library/importlib.metadata.html)
exposes publisher-declared metadata. Under the
[PyPA core metadata specification](https://packaging.python.org/en/latest/specifications/core-metadata/),
`License-Expression` describes the containing distribution, and legacy `License`
is mutually exclusive with it in metadata 2.4. This tool preserves conflicting
fields for review rather than choosing a legal interpretation. Expressions are
not parsed or validated as SPDX, and declared files are not opened.

Before shipping a package, review the actual license texts and notices in the
exact archives, resolve absent/conflicting metadata, and identify native or
bundled components separately. Produce the required distribution notices from
that review. Record reviewer, source archive hashes, license determination and
any unresolved restriction outside the generated report. A permissive dependency
license does not determine the license or ownership of ECG Trust Lab itself.
