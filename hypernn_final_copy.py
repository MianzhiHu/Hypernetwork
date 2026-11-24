from pathlib import Path
import shutil
from task_config import selected_config


BASE = Path(__file__).resolve().parent


def copy_selected_model(base=BASE):
    """Replace the exported configuration and verify that it is the only one."""
    base = Path(base).resolve()
    search_root = (base / 'Results' / 'HyperNN_Grid_Search').resolve()
    output = base / 'Results' / 'Final_Models' / 'HyperNN'
    output.mkdir(parents=True, exist_ok=True)

    source = (search_root / selected_config).resolve()
    if source.parent != search_root:
        raise ValueError(f'Configuration must be inside {search_root}')
    if not source.is_dir():
        raise FileNotFoundError(f'Selected configuration not found: {source}')
    if not any(source.rglob('*_model.pt')):
        raise FileNotFoundError(f'No model checkpoints in {source}')

    # Validate the source before deleting any previous export. Never follow an
    # export-folder link to a directory outside this project's Final_Models.
    if output.resolve() != output:
        raise ValueError(f'Export path must not contain directory links: {output}')

    for existing in output.iterdir():
        if existing.is_symlink():
            existing.unlink()
        elif existing.is_dir():
            if existing.resolve().parent != output:
                raise ValueError(f'Refusing to delete outside {output}: {existing}')
            shutil.rmtree(existing)
        else:
            existing.unlink()

    destination = output / selected_config
    shutil.copytree(source, destination)
    exported = list(output.iterdir())
    if exported != [destination] or not destination.is_dir():
        raise RuntimeError(f'Expected exactly one configuration in {output}')
    print(f'Copied {selected_config} to {output}; confirmed one configuration.')
    return destination


if __name__ == '__main__':
    copy_selected_model()
