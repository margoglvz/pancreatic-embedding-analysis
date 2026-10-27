"""Shared run directories and configuration manifests for CLI and desktop workflows."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs'


def create_run(root=DEFAULT_ROOT):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f') + '_' + uuid4().hex[:6])
    path.mkdir()
    return path


def read_config(run):
    path = Path(run) / 'config.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'created_at': datetime.now(timezone.utc).isoformat(), 'artifacts': {}, 'stages': {}}


def write_config(run, config):
    path = Path(run) / 'config.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def artifact_name(value, suffix):
    name = Path(value).name
    if not name or Path(name).suffix.lower() != suffix:
        raise ValueError(f'Output filename must end in {suffix}: {value}')
    return name


def visualization_run(inp, explicit=None, root=DEFAULT_ROOT):
    inp = Path(inp).expanduser().resolve()
    if explicit:
        run = Path(explicit).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
        return run
    if (inp.parent / 'config.json').exists():
        config = read_config(inp.parent)
        if config['artifacts'].get('pubmed_csv') == inp.name and 'embed_viz' not in config['stages']:
            return inp.parent
    return create_run(root)


def copy_input(run, inp):
    inp = Path(inp).expanduser().resolve()
    config = read_config(run)
    name = config['artifacts'].get('pubmed_csv', 'pubmed_papers.csv')
    target = Path(run) / name
    if inp != target.resolve():
        if target.exists():
            raise FileExistsError(f'Run already contains a different input CSV: {target}')
        shutil.copy2(inp, target)
    return target


@contextmanager
def stage(run, name, settings, artifacts):
    run = Path(run)
    config = read_config(run)
    if name in config['stages']:
        raise FileExistsError(f'{name} already exists in {run}; start a new run.')
    for filename in artifacts.values():
        if (run / filename).exists() and not (name == 'embed_viz' and filename == artifacts.get('pubmed_csv')):
            raise FileExistsError(f'Output already exists: {run / filename}')
    config['artifacts'].update(artifacts)
    # Do not persist NCBI email/API credentials.
    safe = {k: v for k, v in settings.items() if k not in ('email', 'api_key')}
    entry = {'settings': safe, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
    config['stages'][name] = entry
    write_config(run, config)
    print(f'Run folder: {run}', flush=True)
    try:
        yield
    except BaseException as exc:
        entry['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        entry['error_type'] = type(exc).__name__
        raise
    else:
        entry['status'] = 'completed'
    finally:
        entry['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_config(run, config)


def mark_interrupted(run):
    config = read_config(run)
    for entry in config['stages'].values():
        if entry['status'] == 'running':
            entry['status'] = 'interrupted'
            entry['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_config(run, config)
