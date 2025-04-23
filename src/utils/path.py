from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def from_root(*path_segments):
    """
    Build absolute path from root directory.
    
    Example:
        from_root("data", "file.txt") 
        => /absolute/path/to/my_project/data/file.txt
    """
    return ROOT_DIR.joinpath(*path_segments)