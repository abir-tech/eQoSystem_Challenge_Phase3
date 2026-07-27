"""Console entry point: `eqosystem-experiments --backend sa` after pip install."""
import runpy, sys, pathlib


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    sys.argv[0] = "run_experiments.py"
    runpy.run_path(str(root / "run_experiments.py"), run_name="__main__")
