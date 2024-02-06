# directory_maker
"""
create directory, check if it already exists

cs 09oct2023
"""


def directory_maker(directory_path, overwrite=False, verbose=False):
    # Check if data directory (pathlib path) exists, create directory (overwrite if already existing)
    try:
        if directory_path.is_dir():
            if verbose:
                print(
                    "{} directory ({}) already exists!".format(
                        directory_path.stem, directory_path
                    )
                )
            if overwrite:
                for f in directory_path.glob("*.*"):
                    try:
                        f.unlink()
                    except OSError as e:
                        print("Error: %s : %s" % (f, e.strerror))
                print("Directory content deleted!")
            else:
                if verbose:
                    print("Directory content untouched!")
        else:
            # ok, no directory exists.
            directory_path.mkdir(parents=True, exist_ok=False)
            if verbose:
                print(
                    "{} directory ({}) created.".format(
                        directory_path.stem, directory_path
                    )
                )

    except OSError as error:
        print(error)
