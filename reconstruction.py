#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction

    Uses different models for fitting signal and background events
    Usage: python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-b] [-f] [-v]
    Run python reconstruction.py --help for more details
"""

import sys
import argparse
import time

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # to prevent TApplication from capturing command line options and breaking argparse

from ROOT import TFile
from ROOT import RooRealVar, RooArgList, RooArgSet, RooDataSet, RooAddPdf, RooCBShape, RooGaussian

from utility.common import calculate_reconstructed_mass, show_mass_plot
from utility.UnreconstructableEventError import UnreconstructableEventError

# few constants
NBINS = 100 # Number of bins in the histogram
XMIN = 4.5 # Left bound of the histogram
XMAX = 6.5 # Right bound of the histogram
PEAK_MIN = 4.7 # Minimum value of the peak
PEAK_MAX = 5.5 # Maximum value of the peak

def process(file_name, tree_name, max_events, n_bins, x_min, x_max, fit, background, peak_x_min, peak_x_max, verbose):
    """
        A function that forms the main logic of the script

        Args:
        file_name (str): the name of the file to process
        tree_name (str): the name of the tree to process
        max_events (int): the maximum number of events that will be processed
        n_bins (int): the number of bins to be used in the histogram
        x_min (float): the left bound of the histogram
        x_max (float): the right bound of the histogram
        fit (bool): the flag that determines whether the data will be fitted
        background (bool): the flag that determines whether signal or background data is processed
        peak_x_min (float): the left bound of the peak
        peak_x_max (float): the right bound of the peak
    """

    start_time = time.time()
    last_timestamp = time.time()

    # Opening the file and getting the branch
    input_file = TFile(file_name, 'read')
    input_tree = input_file.Get(tree_name)

    # Event countes
    processed_events = 0 # Number of processed events
    reconstructable_events = 0 # Events with valid tau+ and tau- decay vertex

    # Variables for RooFit
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max)
    data = RooDataSet('mB', 'Reconstaructed B mass', RooArgSet(b_mass)) # Storage for reconstructed B mass values

    # Loop through the events
    for counter, event in enumerate(input_tree):
        if counter < max_events:
            processed_events += 1
            if (counter + 1) % 100 == 0: # print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                m_B = calculate_reconstructed_mass(event, verbose)
                reconstructable_events += 1
                b_mass.setVal(m_B)
                data.add(RooArgSet(b_mass))
            except UnreconstructableEventError:
                pass

    end_time = time.time()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    if fit:
        if background:
            # background model is Gaussian + Crystal Ball function

            # defining parameters
            mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max)
            width_right_cb = RooRealVar('width_right_cb', '#sigma_{right CB}', 0.2, 0.02, 1.)
            width_gauss = RooRealVar('width_gauss', '#sigma_{Gauss}', 0.2, 0.02, 1.)
            alpha_right_cb = RooRealVar('alpha_right_cb', '#alpha_{right CB}', -1., -10., -0.1)
            n_right_cb = RooRealVar('n_right_cb', 'n_{right CB}', 1., 0., 10.)

            cb_right = RooCBShape('cb_right','Right CB', b_mass, mean, width_right_cb, alpha_right_cb, n_right_cb)
            gauss = RooGaussian('gauss', 'Gauss', b_mass, mean, width_gauss)

            gauss_fraction = RooRealVar('gauss_fraction', 'Gauss fraction', 0.5, 0.01, 1.)

            model = RooAddPdf('model', 'Model to fit', RooArgList(gauss, cb_right), RooArgList(gauss_fraction))
        else:
            # signal model is narrow Gaussian + wide Gaussian + Crystal Ball shape

            # parameters of the model
            mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max)
            width = RooRealVar('width_narrow_gauss', '#sigma', 0.03, 0.01, 0.1)
            width_wide_gauss = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.3, 0.1, 1.)
            alpha = RooRealVar('alpha', '#alpha', -1, -10., -0.1)
            n = RooRealVar('n', 'n', 2., 0.1, 10.)

            narrow_gauss = RooGaussian('narrow_gauss', 'Narrow Gaussian', b_mass, mean, width)
            wide_gauss = RooGaussian('wide_gauss', 'Wide Gaussian', b_mass, mean, width_wide_gauss)
            cb = RooCBShape('cb','CB', b_mass, mean, width, alpha, n)

            narrow_gauss_fraction = RooRealVar('narrow_gaus_fraction', 'Fraction of narrow Gaussian', 0.3, 0.01, 1.)
            cb_fraction = RooRealVar('cb_fraction', 'Fraction of CB', 0.3, 0.01, 1.)

            model = RooAddPdf('model', 'Model to fit', RooArgList(narrow_gauss, cb, wide_gauss), RooArgList(narrow_gauss_fraction, cb_fraction))

        show_mass_plot(b_mass, data, n_bins, fit, model, True)

    else:
        show_mass_plot(b_mass, data, n_bins)

def main(argv):
    """The main function. Parses the command line arguments passed to the script and then runs the process function"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required = True, help = 'name of the file to process')
    parser.add_argument('-t', '--tree', type = str, default = 'Events', help = 'name of the tree to process')
    parser.add_argument('-n', '--nevents', type = int, help = 'maximum number of events to process')
    parser.add_argument('-f', '--fit', action = 'store_true', help = 'fit the histogram')
    parser.add_argument('-b', '--background', action = 'store_true', help = 'use fit model for background events')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'run with increased verbosity')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else sys.maxint

    process(args.input_file, args.tree, max_events, NBINS, XMIN, XMAX, args.fit, args.background, PEAK_MIN, PEAK_MAX, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
