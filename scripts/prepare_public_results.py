"""Create a privacy-preserving, portable archive of experiment text artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
SUBJECT_PATTERN = re.compile(r"(?<![A-Za-z0-9])S\d{2}(?![A-Za-z0-9])")
FORBIDDEN_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])S\d{2}(?![A-Za-z0-9])"),
    re.compile(r"/vepfs/"),
    re.compile(r"/cpfs/"),
    re.compile(r"[A-Za-z]:\\"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--redact-root",
        action="append",
        default=[],
        help="Private absolute root to replace; repeat for multiple roots.",
    )
    parser.add_argument(
        "--private-salt-file",
        type=Path,
        help="Defaults to a hidden file beside the run directory and is never exported.",
    )
    return parser.parse_args()


def text_artifacts(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and ".publication_subject_salt" not in path.name
    )


def load_or_create_salt(path: Path) -> bytes:
    if path.exists():
        value = path.read_text(encoding="ascii").strip()
        return bytes.fromhex(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    path.write_text(salt.hex() + "\n", encoding="ascii")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return salt


def public_subject_id(salt: bytes, subject_id: str) -> str:
    digest = hmac.new(salt, subject_id.encode("ascii"), hashlib.sha256).hexdigest()
    return f"P-{digest[:8].upper()}"


def rewrite_text(
    text: str,
    mapping: dict[str, str],
    redact_roots: list[str] | None = None,
) -> str:
    for index, root in enumerate(redact_roots or [], start=1):
        text = text.replace(root.rstrip("/\\"), f"${{PRIVATE_ROOT_{index}}}")
    for source, target in mapping.items():
        text = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])",
            target,
            text,
        )
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_public_text(path: Path, text: str) -> None:
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"privacy audit failed for {path}: {pattern.pattern}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("output directory must not be inside the source run directory")

    paths = text_artifacts(run_dir)
    texts = {path: path.read_text(encoding="utf-8", errors="replace") for path in paths}
    subject_ids = sorted({value for text in texts.values() for value in SUBJECT_PATTERN.findall(text)})
    salt_file = args.private_salt_file or run_dir.parent / ".publication_subject_salt"
    salt = load_or_create_salt(salt_file.resolve())
    mapping = {subject_id: public_subject_id(salt, subject_id) for subject_id in subject_ids}

    output_dir.mkdir(parents=True)
    for source, text in texts.items():
        relative = source.relative_to(run_dir)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        rewritten = rewrite_text(text, mapping, args.redact_root)
        assert_public_text(destination, rewritten)
        destination.write_text(rewritten, encoding="utf-8", newline="\n")

    if args.config is not None:
        config_text = rewrite_text(
            args.config.resolve(strict=True).read_text(encoding="utf-8"),
            mapping,
            args.redact_root,
        )
        destination = output_dir / "experiment_config.yaml"
        assert_public_text(destination, config_text)
        destination.write_text(config_text, encoding="utf-8", newline="\n")

    checkpoints = sorted(run_dir.rglob("*.pt"))
    checkpoint_lines = [
        f"{sha256_file(path)}  {path.stat().st_size}  {path.relative_to(run_dir).as_posix()}"
        for path in checkpoints
    ]
    (output_dir / "CHECKPOINTS.sha256").write_text(
        "\n".join(checkpoint_lines) + ("\n" if checkpoint_lines else ""),
        encoding="ascii",
    )

    metadata = {
        "source_run": run_dir.name,
        "text_artifact_count": len(paths),
        "subject_count": len(subject_ids),
        "subject_id_method": "HMAC-SHA256 with a private, non-exported salt",
        "redacted_root_count": len(args.redact_root),
        "excluded_binary_artifacts": ["*.pt checkpoints"],
        "checkpoint_count": len(checkpoints),
    }
    (output_dir / "publication_export.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACTS.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "ARTIFACTS.sha256").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="ascii",
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
