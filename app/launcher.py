import multiprocessing
import os
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    from app.gui import main

    main()
