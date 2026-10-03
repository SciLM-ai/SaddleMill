"""INI parsing and backward-compatible key migration for SaddleMill configuration."""

from __future__ import annotations

import configparser
from typing import Any, Mapping, MutableMapping

from saddlemill.config_defaults import RENAMED_KEYS


def parse_value(val: Any):
    """Interpret strings using the historical ``ConfigManager`` coercion rules."""
    if isinstance(val, str):
        val = val.strip()
        if len(val) >= 2 and val[0] in ("'", '"') and val[-1] == val[0]:
            return val[1:-1]

    if str(val).lower() == "true":
        return True
    if str(val).lower() == "false":
        return False

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    if isinstance(val, str) and " " in val:
        return [parse_value(part) for part in val.split()]

    return val


def migrate_renamed_keys(config: MutableMapping[str, dict[str, Any]]) -> None:
    """Apply the existing config-key migrations in place, including their messages."""
    neb = config.get("ourNEB", {})
    for section, old_key, new_key in RENAMED_KEYS:
        sec = config.get(section, {})
        if old_key in sec:
            sec[new_key] = sec.pop(old_key)
            print(f"Note: [{section}] '{old_key}' renamed to '{new_key}'.")

    if "intermediate_minima" in neb:
        enabled = neb.pop("intermediate_minima")
        if enabled and neb.get("intermediate_minima_check_step", 0) == 0:
            neb["intermediate_minima_check_step"] = 100
            print(
                "Note: [ourNEB] 'intermediate_minima=True' converted to "
                "'intermediate_minima_check_step=100'."
            )


def merge_config_file(
    config: MutableMapping[str, dict[str, Any]],
    config_file: str,
    defaults: Mapping[str, Mapping[str, Any]],
) -> None:
    """Merge one INI file into an existing default-filled config mapping."""
    parser = configparser.ConfigParser(inline_comment_prefixes="#")
    parser.optionxform = str
    parser.read(config_file)

    for section in parser.sections():
        if section not in config:
            config[section] = {}
        for key, value in parser.items(section):
            config[section][key] = parse_value(value)

    migrate_renamed_keys(config)

    for section, section_defaults in defaults.items():
        if section in config:
            unknown = set(config[section]) - set(section_defaults)
            for key in sorted(unknown):
                print(
                    f"Warning: Unrecognized key '{key}' in [{section}]. "
                    f"Valid keys: {sorted(section_defaults)}"
                )


__all__ = ["merge_config_file", "migrate_renamed_keys", "parse_value"]
