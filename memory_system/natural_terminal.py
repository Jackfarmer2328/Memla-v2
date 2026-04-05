from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote_plus

from .ollama_client import ChatMessage, UniversalLLMClient


SUPPORTED_ACTION_KINDS = {
    "launch_app",
    "open_url",
    "open_path",
    "list_directory",
    "system_info",
    "unsupported",
}

SUPPORTED_SYSTEM_INFO_TOPICS = {"battery", "cpu", "disk", "memory"}

RISKY_TOKENS = {
    "apt",
    "chown",
    "chmod",
    "delete",
    "format",
    "install",
    "kill",
    "pacman",
    "poweroff",
    "reboot",
    "remove",
    "rm ",
    "shutdown",
    "sudo",
    "uninstall",
}

LINUX_OPEN_COMMAND = ["xdg-open"]
MACOS_OPEN_COMMAND = ["open"]


@dataclass(frozen=True)
class TerminalAction:
    kind: str
    target: str
    resolved_target: str = ""
    safe: bool = True
    note: str = ""


@dataclass(frozen=True)
class TerminalPlan:
    prompt: str
    source: str
    actions: list[TerminalAction] = field(default_factory=list)
    needs_confirmation: bool = False
    clarification: str = ""
    residual_constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TerminalExecutionRecord:
    kind: str
    target: str
    status: str
    message: str
    command: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TerminalExecutionResult:
    prompt: str
    plan_source: str
    ok: bool
    records: list[TerminalExecutionRecord] = field(default_factory=list)
    residual_constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TerminalBenchmarkCase:
    case_id: str
    prompt: str
    expected_actions: list[str] = field(default_factory=list)


APP_SPECS: dict[str, dict[str, Any]] = {
    "chrome": {
        "aliases": ("chrome", "google chrome", "chromium", "chromium browser"),
        "linux": ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser"),
        "darwin": ("Google Chrome",),
        "win32": ("chrome.exe",),
    },
    "firefox": {
        "aliases": ("firefox", "mozilla firefox"),
        "linux": ("firefox",),
        "darwin": ("Firefox",),
        "win32": ("firefox.exe",),
    },
    "spotify": {
        "aliases": ("spotify",),
        "linux": ("spotify",),
        "darwin": ("Spotify",),
        "win32": ("spotify.exe",),
    },
    "vscode": {
        "aliases": ("vscode", "visual studio code"),
        "linux": ("code", "code-insiders"),
        "darwin": ("Visual Studio Code",),
        "win32": ("code.exe",),
    },
    "files": {
        "aliases": ("files", "file manager", "nautilus", "dolphin", "thunar"),
        "linux": ("xdg-open", "nautilus", "dolphin", "thunar"),
        "darwin": ("Finder",),
        "win32": ("explorer.exe",),
    },
    "discord": {
        "aliases": ("discord",),
        "linux": ("discord",),
        "darwin": ("Discord",),
        "win32": ("discord.exe",),
    },
    "slack": {
        "aliases": ("slack",),
        "linux": ("slack",),
        "darwin": ("Slack",),
        "win32": ("slack.exe",),
    },
    "steam": {
        "aliases": ("steam",),
        "linux": ("steam",),
        "darwin": ("Steam",),
        "win32": ("steam.exe",),
    },
    "terminal": {
        "aliases": ("terminal", "console"),
        "linux": ("kitty", "gnome-terminal", "konsole", "xfce4-terminal", "xterm"),
        "darwin": ("Terminal", "iTerm"),
        "win32": ("wt.exe", "powershell.exe", "cmd.exe"),
    },
}


PATH_ALIASES = {
    "home": str(Path.home()),
    "desktop": str(Path.home() / "Desktop"),
    "downloads": str(Path.home() / "Downloads"),
    "documents": str(Path.home() / "Documents"),
    "music": str(Path.home() / "Music"),
    "pictures": str(Path.home() / "Pictures"),
    "videos": str(Path.home() / "Videos"),
}


SEARCH_ENGINE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bsearch\s+(?P<engine>google|web|youtube|github|reddit|amazon)\s+for\s+(?P<query>.+)$", flags=re.IGNORECASE),
        "engine_first",
    ),
    (
        re.compile(r"\b(?:search|find)\s+(?P<query>.+?)\s+on\s+(?P<engine>google|web|youtube|github|reddit|amazon)\b", flags=re.IGNORECASE),
        "query_first",
    ),
    (
        re.compile(r"\bopen\s+(?P<engine>youtube|github|google|reddit|amazon)\s+and\s+search\s+(?P<query>.+)$", flags=re.IGNORECASE),
        "engine_first",
    ),
)


def terminal_model_default() -> str:
    return os.environ.get("MEMLA_TERMINAL_MODEL") or os.environ.get("OLLAMA_MODEL") or "phi3:mini"


def _normalize_label(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _alias_hits(prompt: str, alias_map: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = _normalize_label(prompt)
    hits: list[tuple[int, int, str]] = []
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(_normalize_label(alias))}(?![a-z0-9])"
            match = re.search(pattern, normalized)
            if match:
                hits.append((match.start(), -(match.end() - match.start()), canonical))
                break
    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, canonical in sorted(hits):
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(canonical)
    return ordered


def _url_hits(prompt: str) -> list[str]:
    urls = re.findall(r"https?://[^\s]+", prompt, flags=re.IGNORECASE)
    bare_domains = re.findall(r"(?<!@)\b[a-z0-9.-]+\.(?:com|org|net|io|dev|app|ai)\b", prompt, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(urls) + [f"https://{item}" for item in bare_domains]:
        clean = str(raw or "").strip().rstrip(".,)")
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        out.append(clean)
    return out


def _contains_risky_intent(prompt: str) -> bool:
    lower = str(prompt or "").strip().lower()
    return any(token in lower for token in RISKY_TOKENS)


def _search_url(engine: str, query: str) -> str:
    clean_engine = _normalize_label(engine)
    clean_query = " ".join(str(query or "").strip().split())
    if not clean_query:
        return ""
    encoded = quote_plus(clean_query)
    if clean_engine in {"google", "web"}:
        return f"https://www.google.com/search?q={encoded}"
    if clean_engine == "youtube":
        return f"https://www.youtube.com/results?search_query={encoded}"
    if clean_engine == "github":
        return f"https://github.com/search?q={encoded}&type=repositories"
    if clean_engine == "reddit":
        return f"https://www.reddit.com/search/?q={encoded}"
    if clean_engine == "amazon":
        return f"https://www.amazon.com/s?k={encoded}"
    return ""


def _search_hits(prompt: str) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for pattern, _ordering in SEARCH_ENGINE_PATTERNS:
        for match in pattern.finditer(prompt):
            engine = str(match.groupdict().get("engine") or "").strip()
            query = str(match.groupdict().get("query") or "").strip().rstrip(".!?")
            url = _search_url(engine, query)
            if not url or url.lower() in seen:
                continue
            seen.add(url.lower())
            hits.append(url)
    return hits


def _heuristic_plan(prompt: str) -> TerminalPlan | None:
    if _contains_risky_intent(prompt):
        return TerminalPlan(
            prompt=prompt,
            source="heuristic",
            clarification="This V1 terminal only supports safe launch/open/status tasks.",
            residual_constraints=["unsupported_risky_action"],
        )
    app_alias_map = {key: tuple(spec.get("aliases") or ()) for key, spec in APP_SPECS.items()}
    path_alias_map = {key: (key,) for key in PATH_ALIASES}
    app_hits = _alias_hits(prompt, app_alias_map)
    path_hits = _alias_hits(prompt, path_alias_map)
    search_hits = _search_hits(prompt)
    url_hits = _url_hits(prompt)
    normalized = _normalize_label(prompt)

    actions: list[TerminalAction] = []
    if search_hits:
        actions.extend(TerminalAction(kind="open_url", target=url, resolved_target=url) for url in search_hits)
    if url_hits:
        actions.extend(TerminalAction(kind="open_url", target=url, resolved_target=url) for url in url_hits)
    if app_hits:
        actions.extend(TerminalAction(kind="launch_app", target=app, resolved_target=app) for app in app_hits)
    if not actions and path_hits and any(token in normalized for token in {"open", "show", "browse"}):
        actions.extend(
            TerminalAction(kind="open_path", target=key, resolved_target=PATH_ALIASES[key])
            for key in path_hits
        )
    if not actions and path_hits and any(token in normalized for token in {"list", "ls", "what s in", "what is in"}):
        actions.extend(
            TerminalAction(kind="list_directory", target=key, resolved_target=PATH_ALIASES[key])
            for key in path_hits
        )
    if not actions:
        for topic in SUPPORTED_SYSTEM_INFO_TOPICS:
            if topic in normalized and any(token in normalized for token in {"show", "check", "status", "usage", "what s", "what is"}):
                actions.append(TerminalAction(kind="system_info", target=topic, resolved_target=topic))
                break

    if not actions:
        return None
    return TerminalPlan(prompt=prompt, source="heuristic", actions=actions)


def _extract_json_object(response: str) -> dict[str, Any]:
    text = str(response or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _llm_prompt(prompt: str) -> list[ChatMessage]:
    app_ids = ", ".join(sorted(APP_SPECS))
    path_ids = ", ".join(sorted(PATH_ALIASES))
    topics = ", ".join(sorted(SUPPORTED_SYSTEM_INFO_TOPICS))
    schema = {
        "actions": [{"kind": "launch_app", "target": "chrome"}],
        "needs_confirmation": False,
        "clarification": "",
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You translate natural terminal requests into a tiny safe JSON action plan.\n"
                "Allowed action kinds: launch_app, open_url, open_path, list_directory, system_info, unsupported.\n"
                f"Allowed app ids: {app_ids}.\n"
                f"Allowed path ids: {path_ids}.\n"
                f"Allowed system_info targets: {topics}.\n"
                "Never emit shell commands. Never suggest destructive or privileged actions. "
                "If the request is risky or unsupported, return a single unsupported action.\n"
                f"Return JSON only in this shape: {json.dumps(schema)}"
            ),
        ),
        ChatMessage(role="user", content=prompt),
    ]


def _plan_from_model_response(*, prompt: str, response: str, source: str) -> TerminalPlan:
    payload = _extract_json_object(response)
    actions = _normalize_model_actions(payload)
    clarification = str(payload.get("clarification") or "").strip()
    needs_confirmation = bool(payload.get("needs_confirmation"))
    if not actions:
        return TerminalPlan(
            prompt=prompt,
            source=source,
            clarification=clarification or "I could not map that request to a safe bounded terminal action.",
            residual_constraints=["unsupported_or_ambiguous_request"],
        )
    return TerminalPlan(
        prompt=prompt,
        source=source,
        actions=actions,
        needs_confirmation=needs_confirmation,
        clarification=clarification,
        residual_constraints=[],
    )


def _normalize_model_actions(payload: dict[str, Any]) -> list[TerminalAction]:
    actions: list[TerminalAction] = []
    for raw in list(payload.get("actions") or []):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        target = str(raw.get("target") or "").strip()
        if kind not in SUPPORTED_ACTION_KINDS or not target:
            continue
        if kind == "launch_app":
            app_key = _resolve_app_key(target)
            if app_key:
                actions.append(TerminalAction(kind=kind, target=target, resolved_target=app_key))
            continue
        if kind in {"open_path", "list_directory"}:
            path_value = _resolve_path_target(target)
            if path_value:
                actions.append(TerminalAction(kind=kind, target=target, resolved_target=path_value))
            continue
        if kind == "open_url":
            url = _normalize_url(target)
            if url:
                actions.append(TerminalAction(kind=kind, target=target, resolved_target=url))
            continue
        if kind == "system_info":
            topic = _normalize_label(target)
            if topic in SUPPORTED_SYSTEM_INFO_TOPICS:
                actions.append(TerminalAction(kind=kind, target=target, resolved_target=topic))
            continue
        if kind == "unsupported":
            actions.append(TerminalAction(kind=kind, target=target, resolved_target=target, safe=False))
    return actions


def _resolve_app_key(raw: str) -> str:
    normalized = _normalize_label(raw)
    for app_key, spec in APP_SPECS.items():
        aliases = {_normalize_label(app_key), *(_normalize_label(alias) for alias in spec.get("aliases") or ())}
        if normalized in aliases:
            return app_key
    return ""


def _resolve_path_target(raw: str) -> str:
    normalized = _normalize_label(raw)
    if normalized in PATH_ALIASES:
        return PATH_ALIASES[normalized]
    path = Path(str(raw or "").strip()).expanduser()
    if str(path).strip():
        return str(path)
    return ""


def _normalize_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if re.match(r"^https?://", text, flags=re.IGNORECASE):
        return text
    if re.match(r"^[a-z0-9.-]+\.(?:com|org|net|io|dev|app|ai)(?:/.*)?$", text, flags=re.IGNORECASE):
        return f"https://{text}"
    return ""


def _canonical_expected_action(value: str) -> str:
    text = str(value or "").strip()
    if not text or ":" not in text:
        return _normalize_label(text)
    kind, target = text.split(":", 1)
    normalized_kind = str(kind or "").strip().lower().replace(" ", "_")
    normalized_target = str(target or "").strip()
    if normalized_kind == "open_url":
        normalized_target = _normalize_url(normalized_target)
    else:
        normalized_target = _normalize_label(normalized_target)
    return f"{normalized_kind}:{normalized_target}"


def _action_signature(action: TerminalAction) -> str:
    kind = str(action.kind or "").strip().lower().replace(" ", "_")
    if kind in {"open_path", "list_directory"}:
        alias_key = _normalize_label(action.target)
        if alias_key in PATH_ALIASES:
            return f"{kind}:{alias_key}"
        return f"{kind}:{_normalize_label(action.resolved_target or action.target)}"
    if kind == "launch_app":
        app_key = _resolve_app_key(action.resolved_target or action.target)
        return f"{kind}:{app_key or _normalize_label(action.resolved_target or action.target)}"
    if kind == "open_url":
        return f"{kind}:{_normalize_url(action.resolved_target or action.target)}"
    if kind == "system_info":
        return f"{kind}:{_normalize_label(action.resolved_target or action.target)}"
    return f"{kind}:{_normalize_label(action.resolved_target or action.target)}"


def _action_recall(plan: TerminalPlan, expected_actions: list[str]) -> float:
    expected = [_canonical_expected_action(item) for item in expected_actions if str(item or "").strip()]
    if not expected:
        return 1.0 if not plan.actions else 0.0
    predicted = {_action_signature(action) for action in plan.actions}
    hits = sum(1 for item in expected if item in predicted)
    return round(hits / len(expected), 4)


def _terminal_utility(plan: TerminalPlan, expected_actions: list[str]) -> float:
    recall = _action_recall(plan, expected_actions)
    support = 1.0 if plan.actions else 0.0
    return round((0.3 * support) + (0.7 * recall), 4)


def load_terminal_benchmark_cases(path: str) -> list[TerminalBenchmarkCase]:
    cases: list[TerminalBenchmarkCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue
        payload = json.loads(clean)
        case_id = str(payload.get("case_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not case_id or not prompt:
            continue
        expected_actions = [str(item).strip() for item in list(payload.get("expected_actions") or []) if str(item).strip()]
        cases.append(TerminalBenchmarkCase(case_id=case_id, prompt=prompt, expected_actions=expected_actions))
    return cases


def build_terminal_plan(
    *,
    prompt: str,
    model: str = "",
    client: UniversalLLMClient | None = None,
    heuristic_only: bool = False,
    temperature: float = 0.1,
) -> TerminalPlan:
    heuristic = _heuristic_plan(prompt)
    if heuristic is not None:
        return heuristic
    if heuristic_only or client is None or not str(model or "").strip():
        return TerminalPlan(
            prompt=prompt,
            source="fallback",
            clarification="I could not map that request to a safe bounded terminal action.",
            residual_constraints=["unsupported_or_ambiguous_request"],
        )
    try:
        response = client.chat(model=model, messages=_llm_prompt(prompt), temperature=temperature)
    except Exception as exc:
        return TerminalPlan(
            prompt=prompt,
            source="model_error",
            clarification=f"Model fallback unavailable: {str(exc).strip() or 'unknown error'}",
            residual_constraints=["llm_fallback_unavailable"],
        )
    return _plan_from_model_response(prompt=prompt, response=response, source="model")


def build_raw_terminal_plan(
    *,
    prompt: str,
    model: str,
    client: UniversalLLMClient | None,
    temperature: float = 0.1,
) -> TerminalPlan:
    if client is None or not str(model or "").strip():
        return TerminalPlan(
            prompt=prompt,
            source="raw_model",
            clarification="Raw terminal mode requires a model client.",
            residual_constraints=["llm_fallback_unavailable"],
        )
    try:
        response = client.chat(model=model, messages=_llm_prompt(prompt), temperature=temperature)
    except Exception as exc:
        return TerminalPlan(
            prompt=prompt,
            source="raw_model",
            clarification=f"Raw model unavailable: {str(exc).strip() or 'unknown error'}",
            residual_constraints=["llm_fallback_unavailable"],
        )
    return _plan_from_model_response(prompt=prompt, response=response, source="raw_model")


def _platform_key(platform_name: str) -> str:
    lower = str(platform_name or sys.platform).lower()
    if lower.startswith("linux"):
        return "linux"
    if lower.startswith("darwin"):
        return "darwin"
    if lower.startswith("win"):
        return "win32"
    return lower


def _resolve_launch_command(app_key: str, *, platform_name: str) -> list[str]:
    spec = APP_SPECS.get(app_key) or {}
    key = _platform_key(platform_name)
    for candidate in spec.get(key) or ():
        binary = shutil.which(str(candidate))
        if binary:
            return [binary]
    return []


def _open_command_for_platform(platform_name: str) -> list[str]:
    key = _platform_key(platform_name)
    if key == "linux":
        return list(LINUX_OPEN_COMMAND)
    if key == "darwin":
        return list(MACOS_OPEN_COMMAND)
    if key == "win32":
        return ["cmd", "/c", "start", ""]
    return []


def _system_info_record(topic: str) -> TerminalExecutionRecord:
    if topic == "disk":
        usage = shutil.disk_usage(Path.home())
        message = (
            f"Home disk usage: total={round(usage.total / (1024**3), 1)} GiB, "
            f"used={round(usage.used / (1024**3), 1)} GiB, free={round(usage.free / (1024**3), 1)} GiB"
        )
        return TerminalExecutionRecord(kind="system_info", target=topic, status="ok", message=message)
    if topic == "cpu":
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        message = f"CPU load averages: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}"
        return TerminalExecutionRecord(kind="system_info", target=topic, status="ok", message=message)
    if topic == "memory":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            rows = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in meminfo.read_text(encoding="utf-8").splitlines() if ":" in line}
            total_kib = int(rows.get("MemTotal", "0 kB").split()[0] or 0)
            available_kib = int(rows.get("MemAvailable", "0 kB").split()[0] or 0)
            used_gib = (total_kib - available_kib) / (1024**2)
            total_gib = total_kib / (1024**2)
            return TerminalExecutionRecord(
                kind="system_info",
                target=topic,
                status="ok",
                message=f"Memory usage: used={used_gib:.1f} GiB, total={total_gib:.1f} GiB",
            )
        return TerminalExecutionRecord(kind="system_info", target=topic, status="failed", message="Memory info unavailable on this platform.")
    if topic == "battery":
        power_root = Path("/sys/class/power_supply")
        if power_root.exists():
            for candidate in power_root.iterdir():
                capacity = candidate / "capacity"
                status_file = candidate / "status"
                if capacity.exists():
                    level = capacity.read_text(encoding="utf-8").strip()
                    status = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else "unknown"
                    return TerminalExecutionRecord(
                        kind="system_info",
                        target=topic,
                        status="ok",
                        message=f"Battery: {level}% ({status})",
                    )
        return TerminalExecutionRecord(kind="system_info", target=topic, status="failed", message="Battery info unavailable on this platform.")
    return TerminalExecutionRecord(kind="system_info", target=topic, status="failed", message="Unsupported system info topic.")


def execute_terminal_plan(plan: TerminalPlan, *, platform_name: str | None = None) -> TerminalExecutionResult:
    platform_key = _platform_key(platform_name or sys.platform)
    records: list[TerminalExecutionRecord] = []
    residuals = list(plan.residual_constraints)
    for action in plan.actions:
        if action.kind == "launch_app":
            command = _resolve_launch_command(action.resolved_target or action.target, platform_name=platform_key)
            if not command:
                residuals.append(f"app_not_found:{action.resolved_target or action.target}")
                records.append(
                    TerminalExecutionRecord(
                        kind=action.kind,
                        target=action.target,
                        status="failed",
                        message=f"Could not find an installed launcher for {action.resolved_target or action.target}.",
                    )
                )
                continue
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            records.append(
                TerminalExecutionRecord(
                    kind=action.kind,
                    target=action.target,
                    status="ok",
                    message=f"Launched {action.resolved_target or action.target}.",
                    command=command,
                )
            )
            continue
        if action.kind in {"open_url", "open_path"}:
            launcher = _open_command_for_platform(platform_key)
            target = action.resolved_target or action.target
            if action.kind == "open_path":
                path_target = Path(target).expanduser()
                if not path_target.exists():
                    residuals.append(f"path_not_found:{path_target}")
                    records.append(
                        TerminalExecutionRecord(
                            kind=action.kind,
                            target=action.target,
                            status="failed",
                            message=f"{path_target} does not exist.",
                        )
                    )
                    continue
            if not launcher:
                residuals.append(f"open_command_unavailable:{platform_key}")
                records.append(
                    TerminalExecutionRecord(
                        kind=action.kind,
                        target=action.target,
                        status="failed",
                        message=f"No opener command is configured for platform {platform_key}.",
                    )
                )
                continue
            command = list(launcher) + [target]
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            records.append(
                TerminalExecutionRecord(
                    kind=action.kind,
                    target=action.target,
                    status="ok",
                    message=f"Opened {target}.",
                    command=command,
                )
            )
            continue
        if action.kind == "list_directory":
            target = Path(action.resolved_target or action.target).expanduser()
            if not target.exists() or not target.is_dir():
                residuals.append(f"path_not_found:{target}")
                records.append(
                    TerminalExecutionRecord(
                        kind=action.kind,
                        target=action.target,
                        status="failed",
                        message=f"{target} is not an existing directory.",
                    )
                )
                continue
            entries = [item.name + ("/" if item.is_dir() else "") for item in sorted(target.iterdir(), key=lambda item: item.name.lower())[:20]]
            records.append(
                TerminalExecutionRecord(
                    kind=action.kind,
                    target=action.target,
                    status="ok",
                    message=f"{target}: {', '.join(entries) if entries else '(empty)'}",
                )
            )
            continue
        if action.kind == "system_info":
            record = _system_info_record(action.resolved_target or action.target)
            if record.status != "ok":
                residuals.append(f"system_info_unavailable:{action.resolved_target or action.target}")
            records.append(record)
            continue
        residuals.append(f"unsupported_action:{action.kind}")
        records.append(
            TerminalExecutionRecord(
                kind=action.kind,
                target=action.target,
                status="failed",
                message=f"Unsupported action kind: {action.kind}",
            )
        )
    ok = bool(records) and all(record.status == "ok" for record in records)
    return TerminalExecutionResult(
        prompt=plan.prompt,
        plan_source=plan.source,
        ok=ok,
        records=records,
        residual_constraints=residuals,
    )


def run_terminal_benchmark(
    *,
    cases_path: str,
    raw_model: str,
    memla_model: str,
    raw_provider: str = "",
    raw_base_url: str = "",
    memla_provider: str = "",
    memla_base_url: str = "",
    temperature: float = 0.1,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    heuristic_only: bool = False,
) -> dict[str, Any]:
    cases = load_terminal_benchmark_cases(cases_path)
    selected_ids = {str(item).strip() for item in list(case_ids or []) if str(item).strip()}
    if selected_ids:
        cases = [case for case in cases if case.case_id in selected_ids]
    if limit is not None:
        cases = cases[: max(int(limit), 0)]

    raw_client = build_llm_client(provider=raw_provider or None, base_url=raw_base_url or None)
    memla_client = build_llm_client(provider=memla_provider or None, base_url=memla_base_url or None)

    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    raw_model_calls = 0
    memla_model_calls = 0
    memla_heuristic_hits = 0

    for case in cases:
        try:
            raw_start = time.perf_counter()
            raw_plan = build_raw_terminal_plan(
                prompt=case.prompt,
                model=raw_model,
                client=raw_client,
                temperature=temperature,
            )
            raw_latency_ms = round((time.perf_counter() - raw_start) * 1000.0, 2)

            memla_start = time.perf_counter()
            memla_plan = build_terminal_plan(
                prompt=case.prompt,
                model=memla_model,
                client=memla_client,
                heuristic_only=heuristic_only,
                temperature=temperature,
            )
            memla_latency_ms = round((time.perf_counter() - memla_start) * 1000.0, 2)
        except Exception as exc:
            failed_cases.append({"case_id": case.case_id, "error_type": type(exc).__name__, "message": str(exc)})
            continue

        if raw_plan.source == "raw_model":
            raw_model_calls += 1
        if memla_plan.source == "model":
            memla_model_calls += 1
        if memla_plan.source == "heuristic":
            memla_heuristic_hits += 1

        raw_recall = _action_recall(raw_plan, case.expected_actions)
        memla_recall = _action_recall(memla_plan, case.expected_actions)
        raw_utility = _terminal_utility(raw_plan, case.expected_actions)
        memla_utility = _terminal_utility(memla_plan, case.expected_actions)

        rows.append(
            {
                "case_id": case.case_id,
                "prompt": case.prompt,
                "expected_actions": list(case.expected_actions),
                "raw_source": raw_plan.source,
                "raw_latency_ms": raw_latency_ms,
                "raw_actions": [_action_signature(action) for action in raw_plan.actions],
                "raw_action_recall": raw_recall,
                "raw_supported": bool(raw_plan.actions),
                "raw_terminal_utility": raw_utility,
                "raw_residual_constraints": list(raw_plan.residual_constraints),
                "memla_source": memla_plan.source,
                "memla_latency_ms": memla_latency_ms,
                "memla_actions": [_action_signature(action) for action in memla_plan.actions],
                "memla_action_recall": memla_recall,
                "memla_supported": bool(memla_plan.actions),
                "memla_terminal_utility": memla_utility,
                "memla_residual_constraints": list(memla_plan.residual_constraints),
                "latency_delta_ms": round(raw_latency_ms - memla_latency_ms, 2),
            }
        )

    count = len(rows) or 1
    avg_raw_latency_ms = round(sum(float(row["raw_latency_ms"]) for row in rows) / count, 2)
    avg_memla_latency_ms = round(sum(float(row["memla_latency_ms"]) for row in rows) / count, 2)
    avg_raw_recall = round(sum(float(row["raw_action_recall"]) for row in rows) / count, 4)
    avg_memla_recall = round(sum(float(row["memla_action_recall"]) for row in rows) / count, 4)
    avg_raw_utility = round(sum(float(row["raw_terminal_utility"]) for row in rows) / count, 4)
    avg_memla_utility = round(sum(float(row["memla_terminal_utility"]) for row in rows) / count, 4)
    raw_support_rate = round(sum(1.0 for row in rows if row["raw_supported"]) / count, 4)
    memla_support_rate = round(sum(1.0 for row in rows if row["memla_supported"]) / count, 4)
    speedup = round(avg_raw_latency_ms / avg_memla_latency_ms, 4) if avg_memla_latency_ms > 0 else None

    return {
        "generated_ts": int(time.time()),
        "cases_path": str(Path(cases_path).resolve()),
        "case_ids": [case.case_id for case in cases],
        "limit": limit,
        "raw_model": raw_model,
        "memla_model": memla_model,
        "raw_provider": raw_client.provider,
        "memla_provider": memla_client.provider,
        "cases": len(rows),
        "cases_requested": len(cases),
        "failed_case_count": len(failed_cases),
        "avg_raw_latency_ms": avg_raw_latency_ms,
        "avg_memla_latency_ms": avg_memla_latency_ms,
        "memla_vs_raw_speedup": speedup,
        "avg_raw_action_recall": avg_raw_recall,
        "avg_memla_action_recall": avg_memla_recall,
        "avg_raw_terminal_utility": avg_raw_utility,
        "avg_memla_terminal_utility": avg_memla_utility,
        "raw_support_rate": raw_support_rate,
        "memla_support_rate": memla_support_rate,
        "raw_model_call_count": raw_model_calls,
        "memla_model_call_count": memla_model_calls,
        "memla_heuristic_hit_count": memla_heuristic_hits,
        "rows": rows,
        "failed_cases": failed_cases,
    }


def render_terminal_benchmark_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Terminal Benchmark",
        "",
        f"- Raw provider: `{report.get('raw_provider', '')}`",
        f"- Memla provider: `{report.get('memla_provider', '')}`",
        f"- Raw model: `{report.get('raw_model', '')}`",
        f"- Memla model: `{report.get('memla_model', '')}`",
        f"- Cases completed: `{report.get('cases', 0)}` / `{report.get('cases_requested', 0)}`",
        "",
        "## Lane summary",
        "",
        "| Metric | Raw | Memla |",
        "| --- | --- | --- |",
        f"| Avg latency (ms) | `{report.get('avg_raw_latency_ms', 0.0)}` | `{report.get('avg_memla_latency_ms', 0.0)}` |",
        f"| Action recall | `{report.get('avg_raw_action_recall', 0.0)}` | `{report.get('avg_memla_action_recall', 0.0)}` |",
        f"| Support rate | `{report.get('raw_support_rate', 0.0)}` | `{report.get('memla_support_rate', 0.0)}` |",
        f"| Terminal utility | `{report.get('avg_raw_terminal_utility', 0.0)}` | `{report.get('avg_memla_terminal_utility', 0.0)}` |",
        "",
        f"- Raw model calls: `{report.get('raw_model_call_count', 0)}`",
        f"- Memla model calls: `{report.get('memla_model_call_count', 0)}`",
        f"- Memla heuristic hits: `{report.get('memla_heuristic_hit_count', 0)}`",
    ]
    speedup = report.get("memla_vs_raw_speedup")
    if speedup:
        lines.extend(["", f"- Raw-vs-Memla latency speedup: `{speedup}x`"])
    if report.get("failed_cases"):
        lines.extend(["", "## Failed cases", ""])
        for failure in report["failed_cases"]:
            lines.append(f"- `{failure.get('case_id', '')}` [{failure.get('error_type', '')}] {failure.get('message', '')}".rstrip())
    lines.extend(["", "## Case rows", ""])
    for row in report.get("rows", []):
        lines.extend(
            [
                f"### {row.get('case_id', '')}",
                "",
                f"- Prompt: `{row.get('prompt', '')}`",
                f"- Expected actions: `{', '.join(row.get('expected_actions', []))}`",
                f"- Raw source/actions: `{row.get('raw_source', '')}` / `{', '.join(row.get('raw_actions', []))}`",
                f"- Memla source/actions: `{row.get('memla_source', '')}` / `{', '.join(row.get('memla_actions', []))}`",
                f"- Raw latency: `{row.get('raw_latency_ms', 0.0)} ms`",
                f"- Memla latency: `{row.get('memla_latency_ms', 0.0)} ms`",
                f"- Raw utility: `{row.get('raw_terminal_utility', 0.0)}`",
                f"- Memla utility: `{row.get('memla_terminal_utility', 0.0)}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_terminal_plan_text(plan: TerminalPlan) -> str:
    lines = [f"Prompt: {plan.prompt}", f"Plan source: {plan.source}"]
    if plan.actions:
        lines.append("Actions:")
        for action in plan.actions:
            target = action.resolved_target or action.target
            lines.append(f"- {action.kind}: {target}")
    if plan.clarification:
        lines.append(f"Clarification: {plan.clarification}")
    if plan.residual_constraints:
        lines.append(f"Residual constraints: {', '.join(plan.residual_constraints)}")
    return "\n".join(lines)


def render_terminal_execution_text(result: TerminalExecutionResult) -> str:
    lines = [f"Prompt: {result.prompt}", f"Plan source: {result.plan_source}", f"Execution: {'OK' if result.ok else 'FAILED'}"]
    for record in result.records:
        lines.append(f"- {record.kind} {record.target}: {record.status.upper()} {record.message}".rstrip())
    if result.residual_constraints:
        lines.append(f"Residual constraints: {', '.join(result.residual_constraints)}")
    return "\n".join(lines)


def terminal_plan_to_dict(plan: TerminalPlan) -> dict[str, Any]:
    return asdict(plan)


def terminal_execution_to_dict(result: TerminalExecutionResult) -> dict[str, Any]:
    return asdict(result)


def build_llm_client(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> UniversalLLMClient:
    if not any(value for value in (provider, base_url, api_key)):
        return UniversalLLMClient.from_env()
    resolved_provider = str(provider or os.environ.get("LLM_PROVIDER", "ollama")).strip()
    normalized_provider = UniversalLLMClient._normalize_provider(resolved_provider)
    resolved_base_url = str(base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")).strip()
    resolved_api_key = api_key or os.environ.get("LLM_API_KEY")
    if normalized_provider == "github_models" and not resolved_api_key:
        resolved_api_key = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    return UniversalLLMClient(
        provider=normalized_provider,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def _demo_request_supported(prompt: str) -> bool:
    return build_terminal_plan(prompt=prompt, heuristic_only=True).actions != []


__all__ = [
    "TerminalAction",
    "TerminalPlan",
    "TerminalExecutionRecord",
    "TerminalExecutionResult",
    "build_llm_client",
    "build_raw_terminal_plan",
    "build_terminal_plan",
    "execute_terminal_plan",
    "load_terminal_benchmark_cases",
    "render_terminal_benchmark_markdown",
    "render_terminal_execution_text",
    "render_terminal_plan_text",
    "run_terminal_benchmark",
    "terminal_execution_to_dict",
    "terminal_model_default",
    "terminal_plan_to_dict",
]
