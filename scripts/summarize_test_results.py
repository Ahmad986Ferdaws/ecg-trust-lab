"""Publish CI outcome metadata without copying test data or diagnostics."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _test_identities() -> set[tuple[str, str]]:
    """Read identifiers from tracked source, never from report parameter values."""
    root = Path(__file__).resolve().parents[1]
    try:
        files = subprocess.run(
            ["git", "ls-files", "-z", "--", "tests"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        identities = set()
        for name in files:
            if not name.endswith(".py"):
                continue
            module = name.removesuffix(".py").replace("/", ".")
            if not re.fullmatch(r"tests(?:\.[A-Za-z_][A-Za-z0-9_]*)+", module):
                continue
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and re.fullmatch(
                    r"test_[A-Za-z0-9_]+", node.name
                ):
                    identities.add((module, node.name))
        return identities
    except (OSError, subprocess.CalledProcessError, SyntaxError, UnicodeError):
        return set()


def _skip_category(reason: str) -> str:
    if re.search(r"\bprivate\b", reason, re.IGNORECASE):
        return "private artifacts"
    if re.search(r"\b(cuda|gpu)\b", reason, re.IGNORECASE):
        return "GPU"
    if re.search(r"\b(windows|win32|symlinks?)\b", reason, re.IGNORECASE):
        return "platform"
    return "other"


def summarize(source: Path) -> tuple[dict[str, int], ET.Element]:
    root = ET.parse(source).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("expected a JUnit test suite")
    for reported_suite in root.iter("testsuite"):
        if "tests" in reported_suite.attrib and int(reported_suite.attrib["tests"]) != len(
            list(reported_suite.iter("testcase"))
        ):
            raise ValueError("JUnit case count does not match its suite")
    counts = dict.fromkeys(("tests", "passed", "failures", "errors", "skipped"), 0)
    identities = _test_identities()
    suite = ET.Element("testsuite", name="CPU engineering tests")
    for index, case in enumerate(root.iter("testcase"), start=1):
        # Rebuild from an allowlist. Only identifiers confirmed in tracked test
        # source survive; parameter IDs and all diagnostics are discarded.
        clean_case = ET.SubElement(suite, "testcase", name=f"case-{index:06d}")
        module = case.get("classname", "")
        name = case.get("name", "").split("[", 1)[0]
        if (module, name) in identities:
            clean_case.attrib.update(classname=module, name=name)
        counts["tests"] += 1
        for tag, count in (("error", "errors"), ("failure", "failures"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                counts[count] += 1
                outcome = ET.SubElement(clean_case, tag)
                if tag == "skipped":
                    outcome.set("message", _skip_category(case.find(tag).get("message", "")))
                break
        else:
            counts["passed"] += 1
    suite.attrib.update(
        {key: str(counts[key]) for key in ("tests", "failures", "errors", "skipped")}
    )
    return counts, suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--test-outcome", required=True, choices=("success", "failure"))
    args = parser.parse_args(argv)

    counts = None
    suite = None
    try:
        counts, suite = summarize(args.junit)
    except (OSError, ET.ParseError, ValueError):
        # Parser errors may contain source data or local paths. Keep the public
        # failure message fixed, and never disguise a missing report as zero tests.
        print("error: JUnit results are unavailable or invalid", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.output_dir / "junit-metadata.xml"
    if suite is not None:
        ET.ElementTree(suite).write(xml_path, encoding="utf-8", xml_declaration=True)
    else:
        xml_path.unlink(missing_ok=True)
    metadata = {
        "report_available": counts is not None,
        "test_step_outcome": args.test_outcome,
        "counts": counts,
    }
    skips = dict.fromkeys(("private artifacts", "platform", "GPU", "other"), 0)
    if suite is not None:
        for skipped in suite.iter("skipped"):
            skips[skipped.attrib["message"]] += 1
    metadata["skip_categories"] = skips if counts is not None else None
    (args.output_dir / "test-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["## CPU test results", "", f"Test step: **{args.test_outcome}**.", ""]
    if counts is None:
        lines.append("JUnit results are unavailable or invalid; counts are unknown.")
    else:
        lines.extend(["| Outcome | Count |", "| --- | ---: |"])
        lines.extend(f"| {key.capitalize()} | {value} |" for key, value in counts.items())
        if counts["skipped"]:
            lines.extend(
                [
                    "",
                    "Skip categories: "
                    + ", ".join(f"{key}: {value}" for key, value in skips.items() if value)
                    + ".",
                ]
            )
        failed = [
            ".".join(filter(None, (case.get("classname"), case.attrib["name"])))
            for case in suite
            if case.find("failure") is not None or case.find("error") is not None
        ]
        if failed:
            lines.extend(["", "Failed/error cases (parameter values omitted):", ""])
            lines.extend(f"- `{name}`" for name in failed[:20])
            if len(failed) > 20:
                lines.append(f"- {len(failed) - 20} more cases in the metadata artifact.")
    lines.extend(
        [
            "",
            "Artifacts contain outcome metadata and known test identifiers; "
            "diagnostics are omitted.",
        ]
    )
    with args.summary.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return 0 if counts is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
