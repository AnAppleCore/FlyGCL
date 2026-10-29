"""Validate newcomer navigation and local Markdown links in dongkyu_work.

Pure standard library; deliberately excludes immutable upstream snapshots.
From repository root:
  python dongkyu_work/misa/scripts/verify_docs.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "dongkyu_work"
MISA = WORKSPACE / "misa"
LINK = re.compile(r"!?(?<!\\)\[[^\]]*\]\(([^)]+)\)")
START = [
    WORKSPACE / "README.md",
    MISA / "README.md",
    MISA / "implementation/README.md",
    MISA / "scripts/README.md",
    MISA / "checkpoints/README.md",
    MISA / "original/README.md",
    MISA / "notes/README.md",
    MISA / "notes/FINAL_REPORT.md",
]
MAX_LINES = [40, 90, 45, 65, 40, 40, 60, 85]


def main():
    for path, limit in zip(START, MAX_LINES):
        assert path.is_file(), f"Missing entry-point README: {path}"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= limit, f"Too much detail in entry-point README: {path} ({len(lines)} > {limit})"

    all_docs = sorted(
        p for p in WORKSPACE.rglob("*.md")
        if "original" not in p.relative_to(WORKSPACE).parts
    )
    issues = []
    checked_links = 0
    # Keep shared research documentation free of audience-specific report drafts.
    audience_specific = re.compile(r"박사님|교수님|보고용")

    for doc in all_docs:
        txt = doc.read_text(encoding="utf-8")
        if audience_specific.search(txt):
            issues.append(f"{doc.relative_to(ROOT)} -> audience-specific wording found")
        for match in LINK.finditer(txt):
            target = match.group(1).strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
            if not target or target.startswith(("https://", "http://", "mailto:", "tel:", "//", "data:")):
                continue
            # These are relative Markdown links in our own docs; absolute
            # paths and links in official upstream source are excluded.
            resolved = (doc.parent / target).resolve()
            checked_links += 1
            if not resolved.is_relative_to(ROOT):
                issues.append(f"{doc.relative_to(ROOT)} -> path escapes repo: {target}")
            elif not resolved.exists():
                issues.append(f"{doc.relative_to(ROOT)} -> missing {target}")
    assert not issues, "Broken workspace documentation links:\n" + "\n".join(issues)

    intro = START[0].read_text(encoding="utf-8")
    project = START[1].read_text(encoding="utf-8")
    docs_map = (MISA / "notes/README.md").read_text(encoding="utf-8")
    assert "(misa/README.md)" in intro
    for name in ("implementation/", "scripts/", "checkpoints/", "original/", "notes/"):
        assert name in project, name
    assert "(notes/FINAL_REPORT.md)" in project
    for name in ("FINAL_REPORT.md", "MISA_vs_FlyGCL.md", "ENTRYPOINT_VERIFICATION.md"):
        assert name in docs_map

    print(f"PASS workspace documentation: {len(all_docs)} files, {checked_links} valid local links")
    print("PASS first-visit path: workspace -> MISA -> folders -> result/evidence")
    print("PASS concise directory README limits; official source snapshot excluded")
    print("PASS no audience-specific reporting text in workspace Markdown")


if __name__ == "__main__":
    main()
