"""PRD-286 forced-skill validation regressions."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def profile_board(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    def add_profile(name, skills=()):
        profile = root / "profiles" / name
        (profile / "skills").mkdir(parents=True)
        (profile / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
        for skill in skills:
            skill_dir = profile / "skills" / skill
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {skill}\ndescription: synthetic {skill}\n---\n",
                encoding="utf-8",
            )
        return profile

    return add_profile


def test_frontmatter_name_accepted_by_same_resolver_as_worker(profile_board):
    profile = profile_board("reviewer")
    skill_dir = profile / "skills" / "directory-alias"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: resolver-visible\ndescription: synthetic alias\n---\n",
        encoding="utf-8",
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="frontmatter alias",
            assignee="reviewer",
            skills=["resolver-visible"],
        )

    assert task_id.startswith("t_")


def test_disabled_skill_is_not_force_loadable(profile_board):
    profile = profile_board("reviewer", ("sdlc-review",))
    (profile / "config.yaml").write_text(
        "skills:\n  disabled:\n    - sdlc-review\n",
        encoding="utf-8",
    )

    with kb.connect() as conn:
        with pytest.raises(ValueError, match="sdlc-review"):
            kb.create_task(
                conn,
                title="disabled",
                assignee="reviewer",
                skills=["sdlc-review"],
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("assignee", "incident_skill"),
    [
        ("hugoreviewer", "github-code-review"),
        ("hugomacops", "messaging-gateway-diagnostics"),
    ],
)
def test_prd_285_identifiers_rejected_before_insert(
    profile_board, assignee, incident_skill
):
    profile_board(assignee, ("requesting-code-review", "sdlc-review"))
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="must not persist",
                assignee=assignee,
                skills=[incident_skill],
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert assignee in str(excinfo.value)
    assert incident_skill in str(excinfo.value)


def test_multiple_invalid_skills_are_atomic(profile_board):
    profile_board("reviewer", ("sdlc-review",))
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="bad",
                assignee="reviewer",
                skills=["missing-one", "missing-two"],
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert "reviewer" in str(excinfo.value)
    assert "missing-one" in str(excinfo.value)
    assert "missing-two" in str(excinfo.value)


def test_missing_profile_reports_assignee_and_every_invalid_name(profile_board):
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="missing profile",
                assignee="absent-profile",
                skills=["missing-one", "missing-two"],
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    message = str(excinfo.value)
    assert "absent-profile" in message
    assert "missing-one" in message
    assert "missing-two" in message


def test_mixed_valid_invalid_reports_only_invalid(profile_board):
    profile_board("reviewer", ("sdlc-review",))
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="mixed",
                assignee="reviewer",
                skills=["sdlc-review", "missing-review"],
            )
    message = str(excinfo.value)
    assert "missing-review" in message
    assert message.count("'missing-review'") == 1
    assert "invalid: 'sdlc-review'" not in message


def test_valid_skills_preserved(profile_board):
    profile_board("reviewer", ("sdlc-review", "requesting-code-review"))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="valid",
            assignee="reviewer",
            skills=[" sdlc-review ", "requesting-code-review", "sdlc-review"],
        )
        assert kb.get_task(conn, task_id).skills == [
            "sdlc-review",
            "requesting-code-review",
        ]


def test_skills_null_skips_validation(profile_board, monkeypatch):
    profile_board("reviewer")

    def unexpected(*args, **kwargs):
        raise AssertionError("skills=None scanned a profile")

    monkeypatch.setattr(kb, "_resolver_visible_skill_names", unexpected)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="reviewer", skills=None)
        assert kb.get_task(conn, task_id).skills is None


def test_idempotent_retry_returns_existing_task_after_profile_drift(profile_board):
    profile = profile_board("reviewer", ("sdlc-review",))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="original",
            assignee="reviewer",
            skills=["sdlc-review"],
            idempotency_key="stable-request",
        )
        shutil.rmtree(profile / "skills" / "sdlc-review")

        retried_id = kb.create_task(
            conn,
            title="retried webhook",
            assignee="reviewer",
            skills=["sdlc-review"],
            idempotency_key="stable-request",
        )

        assert retried_id == task_id
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_suggestions_are_close_and_bounded(profile_board):
    profile_board(
        "reviewer",
        (
            "github-code-reviewer",
            "github-code-reviews",
            "github-codes-review",
            "github-code-reviewed",
            "unrelated-private-skill",
        ),
    )
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="suggest",
                assignee="reviewer",
                skills=["github-code-review"],
            )
    message = str(excinfo.value)
    assert "unrelated-private-skill" not in message
    assert message.count("github-code-review") <= (
        kb._FORCED_SKILL_SUGGESTION_LIMIT + 1
    )


def test_suggestion_limit_is_global_across_invalid_names(profile_board):
    candidates = (
        "alpha-review-one",
        "alpha-review-two",
        "alpha-review-three",
        "beta-review-one",
        "beta-review-two",
        "beta-review-three",
    )
    profile_board("reviewer", candidates)
    with kb.connect() as conn:
        with pytest.raises(ValueError) as excinfo:
            kb.create_task(
                conn,
                title="globally bounded suggestions",
                assignee="reviewer",
                skills=["alpha-review", "beta-review"],
            )

    message = str(excinfo.value)
    assert "alpha-review" in message
    assert "beta-review" in message
    assert sum(name in message for name in candidates) <= (
        kb._FORCED_SKILL_SUGGESTION_LIMIT
    )


def test_dispatcher_revalidates_stale_profile(profile_board, monkeypatch, tmp_path):
    profile = profile_board("reviewer", ("sdlc-review",))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="stale", assignee="reviewer", skills=["sdlc-review"]
        )
        task = kb.get_task(conn, task_id)
    shutil.rmtree(profile / "skills" / "sdlc-review")

    def unexpected(*args, **kwargs):
        raise AssertionError("stale task reached subprocess setup")

    monkeypatch.setattr("tools.environments.local.build_subprocess_env", unexpected)
    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError) as excinfo:
        kb._default_spawn(task, str(tmp_path))
    assert "reviewer" in str(excinfo.value)
    assert "sdlc-review" in str(excinfo.value)


def test_crash_persists_redacted_bounded_startup_stderr(profile_board, monkeypatch):
    profile_board("reviewer")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    secret = "sk-test-THIS-MUST-NOT-PERSIST-1234567890"
    lines = [f"noise-{i}-" + "x" * 100 for i in range(20)]
    lines += [
        f"OPENAI_API_KEY={secret}",
        "Error: Unknown skill(s): github-code-review",
    ]
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="crash", assignee="reviewer")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, task_id, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, task_id, 987654)
        log = kb.worker_log_path(task_id)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        kb._record_worker_exit(987654, 1 << 8)
        assert task_id in kb.detect_crashed_workers(conn)
        error = kb.get_task(conn, task_id).last_failure_error
    assert "github-code-review" in error
    assert secret not in error
    excerpt = error.split("startup stderr: ", 1)[1]
    assert len(excerpt.encode("utf-8")) <= kb._STARTUP_STDERR_MAX_BYTES
    assert len(excerpt.splitlines()) <= kb._STARTUP_STDERR_MAX_LINES


def test_startup_stderr_drops_partial_leading_line_before_redaction(
    profile_board,
):
    profile_board("reviewer")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="partial", assignee="reviewer")
    log = kb.worker_log_path(task_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    secret_tail = "private-fragment-" * 700
    log.write_text(
        f"OPENAI_API_KEY={secret_tail}\nError: Unknown skill(s): github-code-review\n",
        encoding="utf-8",
    )

    excerpt = kb._startup_stderr_excerpt(task_id)

    assert "private-fragment" not in excerpt
    assert "github-code-review" in excerpt


def test_startup_stderr_redaction_failure_is_fail_closed(profile_board, monkeypatch):
    profile_board("reviewer")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="redactor failure", assignee="reviewer")
    log = kb.worker_log_path(task_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "OPENAI_API_KEY=private-value-must-not-persist\n",
        encoding="utf-8",
    )

    import agent.redact as redact

    monkeypatch.setattr(
        redact,
        "redact_sensitive_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    assert kb._startup_stderr_excerpt(task_id) == (
        "[startup stderr withheld: redaction unavailable]"
    )
