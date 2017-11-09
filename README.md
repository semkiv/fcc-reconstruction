# fcc-reconstruction

**DISCONTINUED**, but feel free to fork it.

Reconstruction related scripts for _B<sup>0</sup><sub>d</sub> &rarr; K<sup>*0</sup> &tau;<sup>+</sup> &tau;<sup>-</sup>_ studies.
## merger.py
Script that merges multiple ROOT files containing data suitable for reconstruction algorithm into one file
Usage:
```bash
python merger.py -i [INPUT_FILENAME_1] [NUMBER_OF_EVENTS_1] -i [INPUT_FILENAME_2] [NUMBER_OF_EVENTS_2] ... [-o [OUTPUT_FILENAME=merged.root]]
```
Run `python merger.py --help` for more details

## reconstruction.py
Reconstruction script that implements math algorithm of B0 mass reconstruction
Uses different models for fitting signal and background events
Usage:
```bash
python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-b] [-f] [-v]
```
Run `python reconstruction.py --help` for more details

## reconstruction_composite.py
Reconstruction script that implements math algorithm of B0 mass reconstruction
Uses the composite model for fitting signal and background events simultaneously
Usage:
```bash
python reconstruction_composite.py -i [INPUT_FILENAME] [-n [MAX_EVENTS]] [-f] [-v]
```
Run `python reconstruction_composite.py --help` for more details
