# napari-spofi

[![License BSD-3](https://img.shields.io/pypi/l/napari-spofi.svg?color=green)](https://github.com/githubuser/napari-spofi/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-spofi.svg?color=green)](https://pypi.org/project/napari-spofi)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spofi.svg?color=green)](https://python.org)
[![tests](https://github.com/githubuser/napari-spofi/workflows/tests/badge.svg)](https://github.com/githubuser/napari-spofi/actions)
[![codecov](https://codecov.io/gh/githubuser/napari-spofi/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-spofi)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spofi)](https://napari-hub.org/plugins/napari-spofi)

napari plugin to interactively train and test a StarDist model

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started


and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Description

This plugin provides tools for annotating spots in a 3D two-channel image (hdf5 type input file),
submitting tiles for StarDist model generation or model re-training, and refining initial annotations
based on predictions.

The objects of interest in the image are sphere-like spots with a diameter of just a
few pixels and are thus well suited for StarDist instance segmentation. The image 
imensions are typically 1024x1024 pixels in xy and more than 100 sections in z.


## Installation

With python and pip installed (e.g., via miniconda or miniforge),
it is recommended to create a new environment and install `napari-spofi` using pip.

    pip install napari napari-spofi

## Starting `napari-spofi`

Start `napari` and select "spot finder (napari-spofi)" from the "plugin" menu.

In the 'annotation' widget, create a new directory for annotations. Add an image
folder containing at least one h5 file. Select an image file, foreground and background
channels. Load the image file.

Inspect the image for distinct regions. To help locate relevant tile positions, make
the 'checkerboard' layer visible. While the 'tiles' layer is active, double-click a tile
to add it to the list of tiles. This list will be used to generate a set of smaller
images and masks for training purposes.

Switch to napari's 2D view. Navigate to the central section of individual spots
and add points using the 'true' points layer. The heuristic built-in will annotate
pixels that belong to individual spots. Some image enhancement step may be beneficial. 

Annotate tiles in one or a few images.
To prepare training data, use the 'extract spots' button. 



## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari-spofi" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
