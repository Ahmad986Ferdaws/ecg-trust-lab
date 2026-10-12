"""Safe configuration decoding that refuses silently overwritten mapping keys."""

from __future__ import annotations

import json
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

_MERGE_KEY = object()


class DuplicateConfigKeyError(ValueError):
    """A JSON configuration declares the same key more than once."""


class _UniqueKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[object] = set()

    def flatten_mapping(self, node: Any) -> None:
        # Validate before PyYAML expands merges. Explicit overrides of inherited
        # keys are valid, so do not recheck the resulting flattened mapping.
        # The parent implementation recursively calls this method on merge sources.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            seen: set[object] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    key: object = _MERGE_KEY
                elif key_node.tag == "tag:yaml.org,2002:value":
                    key = "="
                else:
                    key = self.construct_object(key_node, deep=False)
                try:
                    duplicate = key in seen
                    seen.add(key)
                except TypeError as error:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found unhashable mapping key",
                        key_node.start_mark,
                    ) from error
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found duplicate mapping key",
                        key_node.start_mark,
                    )
        super().flatten_mapping(node)


def load_yaml_unique_keys(serialized: str) -> object:
    """Keep SafeLoader semantics, including merges, while rejecting repeated keys."""
    return cast(object, yaml.load(serialized, Loader=_UniqueKeySafeLoader))


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise DuplicateConfigKeyError("JSON configuration contains a duplicate mapping key")
        mapping[key] = value
    return mapping


def load_json_unique_keys(serialized: str) -> object:
    """Decode JSON with the same explicit-key rule at every object depth."""
    return cast(object, json.loads(serialized, object_pairs_hook=_unique_json_object))
