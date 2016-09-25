#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction with a solution selector based on MC truth information

    Uses different models for fitting signal and background events
    Usage: python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-l] [-v]
    Run python reconstruction.py --help for more details
"""

import sys
import argparse
import time
import math
import os

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # To prevent TApplication from capturing command line options and breaking argparse. This line has to be the very first line of access of the ROOT module. If any other variables are touched before it, TApplication will be created, which takes the CLI options
from ROOT import TFile

# This awkward construction serves to suppress the output at RooFit modules import
devnull = open(os.devnull, 'w')
old_stdout_fileno = os.dup(sys.stdout.fileno())
os.dup2(devnull.fileno(), 1)
from ROOT import RooRealVar, RooArgSet, RooDataSet
devnull.close()
os.dup2(old_stdout_fileno, 1)

from utility.common import isclose, reconstruct_mc_truth, show_plot
from utility.UnreconstructableEventError import UnreconstructableEventError
from utility.SignalModel import SignalModel

# few constants
NBINS = 100 # Number of bins in the histogram
XMIN = 4.5 # Left bound of the histogram
XMAX = 6.5 # Right bound of the histogram
PEAK_MIN = 4.7 # Minimum value of the peak
PEAK_MAX = 5.5 # Maximum value of the peak

def process(file_name, tree_name, mc_tree_name, max_events, n_bins, x_min, x_max, peak_x_min, peak_x_max, draw_legend, verbose):
    """
        A function that forms the main logic of the script

        Args:
        file_name (str): the name of the file to process
        tree_name (str): the name of the tree to process
        n_bins (int): the number of bins to be used in the histogram
        x_min (float): the left bound of the histogram
        x_max (float): the right bound of the histogram
        peak_x_min (float): the left bound of the peak
        peak_x_max (float): the right bound of the peak
        draw_legend (bool): the flag that determines whether the histogram legend will be drawn
        max_events (int): the maximum number of events that will be processed
        verbose (bool): the flag that switches inreased verbosity
    """

    start_time = time.time()
    last_timestamp = time.time()

    # Opening the file and getting the branch
    input_file = TFile(file_name, 'read')
    event_tree = input_file.Get(tree_name)
    mc_event_tree = input_file.Get(mc_tree_name)

    # Event counters
    processed_events = 0 # Number of processed events
    reconstructable_events = 0 # Events with valid tau+ and tau- decay vertex
    # Variables for RooFit
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max)
    b_mass_data = RooDataSet('mB_data', 'm_{B} data', RooArgSet(b_mass)) # Storage for reconstructed B mass values

    # Loop through the events
    for counter in xrange(event_tree.GetEntries()): # So we have to use the old one
        if counter < max_events:
            event_tree.GetEntry(counter)
            mc_event_tree.GetEntry(counter)

            processed_events += 1
            if (counter + 1) % 100 == 0 and verbose > 0: # Print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                rec_ev = reconstruct_mc_truth(event_tree, mc_event_tree, verbose)
                reconstructable_events += 1

                b_mass.setVal(rec_ev.m_b)
                b_mass_data.add(RooArgSet(b_mass))

            except UnreconstructableEventError:
                pass

    end_time = time.time()


    # Printing some useful statistics
    if verbose > 0:
        print('{} events have been processed'.format(processed_events))
        print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
        print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    model = SignalModel(name = 'signal_model',
                        title = 'Signal Model',
                        x = b_mass,
                        mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max),
                        width = RooRealVar('width_narrow_gauss', '#sigma', 0.03, 0.01, 0.1),
                        width_wide = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.3, 0.1, 1.),
                        alpha = RooRealVar('alpha', '#alpha', -1., -10., -0.1),
                        n = RooRealVar('n', 'n', 2., 0.1, 10.),
                        narrow_gauss_fraction = RooRealVar('signal_model_narrow_gauss_fraction', 'Fraction of Narrow Gaussian in Signal Model', 0.3, 0.01, 1.),
                        cb_fraction = RooRealVar('signal_model_cb_fraction', 'Fraction of Crystal Ball Shape in Signal Model', 0.3, 0.01, 1.))

    show_plot(b_mass, b_mass_data, 'GeV/#it{c}^{2}', n_bins, fit = True, model = model.pdf, extended = False, components_to_plot = model.components, draw_legend = draw_legend)


def main(argv):
    """The main function. Parses the command line arguments passed to the script and then runs the process function"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required = True, help = 'name of the file to process')
    parser.add_argument('-t', '--tree', type = str, default = 'Events', help = 'name of the event tree')
    parser.add_argument('-m', '--mctree', type = str, default = 'MCTruth', help = 'name of the tree with Monte-Carlo truth events')
    parser.add_argument('-n', '--nevents', type = int, help = 'maximum number of events to process')
    parser.add_argument('-l', '--with-legend', action = 'store_true', help = 'draw legend')
    parser.add_argument('-v', '--verbose', type = int, default = 1, help = 'verbosity level')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else sys.maxint

    process(args.input_file, args.tree, args.mctree, max_events, NBINS, XMIN, XMAX, PEAK_MIN, PEAK_MAX, args.with_legend, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
