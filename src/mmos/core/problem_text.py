from mmos.core.project import ProjectPaths, load_manifest
from mmos.schemas.project import SourceRole

TEXT_PROBLEM_SUFFIXES = {".md", ".markdown", ".txt"}


def collect_problem_text(paths: ProjectPaths) -> str:
    manifest = load_manifest(paths)
    problem_files = [
        paths.root / record.path
        for record in manifest.inputs
        if record.source_role is SourceRole.PROBLEM
    ]
    if not problem_files:
        raise ValueError(
            "no problem files ingested; run: mmos ingest <project> <file> --role problem"
        )

    sections: list[str] = []
    for problem_file in problem_files:
        if problem_file.suffix.lower() not in TEXT_PROBLEM_SUFFIXES:
            raise ValueError(
                f"unsupported problem file type: {problem_file.name} (accepts .md, .markdown, .txt)"
            )
        sections.append(f"## {problem_file.name}\n\n{problem_file.read_text(encoding='utf-8')}")
    return "\n\n".join(sections)
