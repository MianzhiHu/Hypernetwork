"""Preview unwanted configuration folders; pass --delete to permanently remove them."""
from pathlib import Path
import argparse
import re
import shutil
import stat

BASE = Path(__file__).resolve().parent
root = BASE / 'Results' / 'HyperNN_Grid_Search_8_Task'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete', action='store_true', help='Permanently delete the listed configuration folders.')
    args = parser.parse_args()

    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.is_symlink() or getattr(root.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f'Refusing a linked target folder: {root}')
    root = root.resolve()
    if root.parent != (BASE / 'Results').resolve():
        raise ValueError(f'Target is outside the expected Results folder: {root}')

    targets = []
    for folder in sorted(root.iterdir()):
        # Match complete numeric fields: emb_2 must not accidentally match emb_20.
        match = re.fullmatch(r'layers_(\d+)_dims_.+_rank_.+_emb_(\d+)_.+', folder.name)
        if not match or not folder.is_dir():
            continue
        layers, embeddings = map(int, match.groups())
        if layers != 2 and embeddings != 2:
            continue
        if (folder.is_symlink()
                or getattr(folder.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or folder.resolve().parent != root):
            raise ValueError(f'Refusing a linked or out-of-scope configuration: {folder}')
        targets.append(folder)

    for folder in targets:
        print(folder.name)
    print(f'\n{len(targets)} folders match: layers == 2 OR embedding dimensions == 2.')
    print(f'Target: {root}')

    if args.delete:
        for folder in targets:
            # Recheck the absolute target immediately before recursive deletion.
            if (folder.resolve().parent != root or folder.is_symlink()
                    or getattr(folder.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ValueError(f'Target changed; refusing deletion: {folder}')
            shutil.rmtree(folder)
            print(f'Deleted: {folder.name}')
        print(f'Deleted {len(targets)} configuration folders.')
    else:
        print('Preview only. Run with --delete to remove these folders and all their contents.')
    # Shared manifests, split files, participant maps, and evaluation folders are not modified.
