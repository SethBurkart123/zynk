#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$ROOT" <<'PY'
from __future__ import annotations

import glob
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(sys.argv[1])

CODEGEN_LINE = {
    "zynk-codegen",
    "zynk-gen-ts",
    "zynk-gen-effect",
    "zynk-cli",
}

CODEGEN_FORBIDDEN_DEPS = {
    "actix",
    "actix-web",
    "axum",
    "hyper",
    "poem",
    "reqwest",
    "rocket",
    "salvo",
    "tonic",
    "tokio",
    "tower",
    "warp",
    "zynk-axum",
    "zynk-cli",
    "zynk-gen-effect",
    "zynk-gen-ts",
    "zynk-runtime",
}

RUNTIME_FORBIDDEN_DEPS = {
    "actix",
    "actix-web",
    "axum",
    "hyper",
    "poem",
    "reqwest",
    "rocket",
    "salvo",
    "tonic",
    "tokio",
    "tower",
    "tower-http",
    "warp",
    "zynk-codegen",
    "zynk-gen-effect",
    "zynk-gen-ts",
    "zynk-axum",
}

RUNTIME_ALLOWED_DEPS = {
    "inventory",
    "serde",
    "serde_json",
    "thiserror",
    "zynk-schema",
}

CLI_FORBIDDEN_DEPS = {
    "actix",
    "actix-web",
    "axum",
    "hyper",
    "poem",
    "reqwest",
    "rocket",
    "salvo",
    "tonic",
    "tower",
    "tower-http",
    "warp",
    "zynk-axum",
}

MACROS_ALLOWED_DEPS = {
    "proc-macro2",
    "quote",
    "syn",
    "zynk-runtime",
}

MACROS_FORBIDDEN_DEPS = {
    "actix",
    "actix-web",
    "axum",
    "hyper",
    "poem",
    "reqwest",
    "rocket",
    "salvo",
    "tonic",
    "tokio",
    "tower",
    "tower-http",
    "warp",
    "zynk-axum",
    "zynk-cli",
    "zynk-codegen",
    "zynk-gen-effect",
    "zynk-gen-ts",
}

PYTHON_FORBIDDEN_RUNTIME_DEPS = {
    "maturin",
    "maturin-runtime",
    "zynk-cli",
    "zynk-codegen",
    "zynk-schema",
}

DEPENDENCY_TABLES = {
    "dependencies",
    "dev-dependencies",
    "build-dependencies",
}


def load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        print(f"error: failed to parse {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def workspace_members() -> list[Path]:
    root_manifest = load_toml(ROOT / "Cargo.toml")
    members = root_manifest.get("workspace", {}).get("members", [])
    manifests: list[Path] = []

    for member in members:
        matches = sorted(glob.glob(str(ROOT / member)))
        if not matches:
            candidate = ROOT / member / "Cargo.toml"
            if candidate.exists():
                manifests.append(candidate)
            continue

        for match in matches:
            manifest = Path(match) / "Cargo.toml"
            if manifest.exists():
                manifests.append(manifest)

    return sorted(set(manifests))


def dependency_name(raw_name: str, value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("package"), str):
        return value["package"]
    return raw_name


def iter_dependency_entries(
    manifest: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    entries: list[tuple[str, str, Any]] = []

    for table_name in DEPENDENCY_TABLES:
        table = manifest.get(table_name, {})
        if isinstance(table, dict):
            for raw_name, value in table.items():
                entries.append((table_name, dependency_name(raw_name, value), value))

    target_tables = manifest.get("target", {})
    if isinstance(target_tables, dict):
        for target_name, target_config in target_tables.items():
            if not isinstance(target_config, dict):
                continue
            for table_name in DEPENDENCY_TABLES:
                table = target_config.get(table_name, {})
                if isinstance(table, dict):
                    section = f"target.{target_name}.{table_name}"
                    for raw_name, value in table.items():
                        entries.append((section, dependency_name(raw_name, value), value))

    return entries


def normalized_python_dependency_name(dependency: str) -> str:
    requirement = dependency.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        return ""
    return match.group(1).lower().replace("_", "-")


def main() -> int:
    violations: list[str] = []
    manifests = workspace_members()

    for manifest_path in manifests:
        manifest = load_toml(manifest_path)
        package = manifest.get("package", {})
        crate_name = package.get("name")
        if not isinstance(crate_name, str):
            continue

        relative_path = manifest_path.relative_to(ROOT)
        dependency_entries = iter_dependency_entries(manifest)

        if crate_name not in CODEGEN_LINE:
            for section, dependency, _value in dependency_entries:
                if dependency == "zynk-codegen":
                    violations.append(
                        f"{relative_path}: crate '{crate_name}' may not depend on "
                        f"'zynk-codegen' in [{section}]; only "
                        "zynk-codegen, zynk-gen-ts, zynk-gen-effect, and zynk-cli "
                        "are in the codegen line"
                    )

        if crate_name == "zynk-codegen":
            for section, dependency, _value in dependency_entries:
                if dependency in CODEGEN_FORBIDDEN_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-codegen may not depend on "
                        f"'{dependency}' in [{section}]; it must stay free of HTTP "
                        "frameworks, runtime crates, CLI crates, and specific generators"
                    )

        if crate_name == "zynk-runtime":
            for section, dependency, _value in dependency_entries:
                if dependency in RUNTIME_FORBIDDEN_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-runtime may not depend on "
                        f"'{dependency}' in [{section}]; it must stay free of HTTP "
                        "frameworks, generators, and server-binding crates"
                    )
                if dependency not in RUNTIME_ALLOWED_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-runtime dependency '{dependency}' in "
                        f"[{section}] is not in the allowed set "
                        f"{sorted(RUNTIME_ALLOWED_DEPS)}"
                    )

        if crate_name == "zynk-cli":
            for section, dependency, _value in dependency_entries:
                if dependency in CLI_FORBIDDEN_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-cli may not depend on "
                        f"'{dependency}' in [{section}]; it must stay free of HTTP "
                        "frameworks and server-binding crates"
                    )

        if crate_name == "zynk-macros":
            lib_config = manifest.get("lib", {})
            if not isinstance(lib_config, dict) or lib_config.get("proc-macro") is not True:
                violations.append(
                    f"{relative_path}: zynk-macros must declare [lib] proc-macro = true"
                )

            for section, dependency, _value in dependency_entries:
                if section != "dependencies":
                    continue
                if dependency in MACROS_FORBIDDEN_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-macros may not depend on "
                        f"'{dependency}' in [{section}]; macros must stay free of "
                        "servers, generators, codegen, CLI, and async runtimes"
                    )
                if dependency not in MACROS_ALLOWED_DEPS:
                    violations.append(
                        f"{relative_path}: zynk-macros dependency '{dependency}' in "
                        f"[{section}] is not in the allowed set "
                        f"{sorted(MACROS_ALLOWED_DEPS)}"
                    )

    python_manifest = ROOT / "bindings/python/pyproject.toml"
    if python_manifest.exists():
        pyproject = load_toml(python_manifest)
        project = pyproject.get("project", {})
        dependencies: list[str] = []
        if isinstance(project, dict):
            runtime_dependencies = project.get("dependencies", [])
            if isinstance(runtime_dependencies, list):
                dependencies.extend(str(dep) for dep in runtime_dependencies)

        for dependency in dependencies:
            dependency_name = normalized_python_dependency_name(dependency)
            if dependency_name in PYTHON_FORBIDDEN_RUNTIME_DEPS:
                violations.append(
                    "bindings/python/pyproject.toml: [project.dependencies] may not "
                    "include Rust crate or build-tool runtime dependency "
                    f"'{dependency}'"
                )

    if violations:
        print("module separation audit failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    checked = ", ".join(str(path.relative_to(ROOT)) for path in manifests)
    print("module separation audit passed")
    print(f"checked workspace manifests: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
