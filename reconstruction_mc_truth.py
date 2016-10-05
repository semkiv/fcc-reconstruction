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
from ROOT import RooFit, RooRealVar, RooArgSet, RooDataSet, RooAddPdf, RooArgList
devnull.close()
os.dup2(old_stdout_fileno, 1)

from utility.common import isclose, reconstruct_mc_truth, show_plot, add_to_RooDataSet
from utility.ReconstructedEvent import UnreconstructableEventError
from utility.fit import SignalModel, BackgroundModel

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
    b_mass = RooRealVar('b_mass', 'm_{B}', x_min, x_max, 'GeV/#it{c}^{2}')
    b_mass_data = RooDataSet('b_mass_data', 'm_{B} data', RooArgSet(b_mass))
    # b_mass_correct = RooRealVar('mB_correct', 'Correct m_{B}', x_min, x_max, 'GeV/#it{c}^{2}')
    b_mass_correct_data = RooDataSet('mB_correct_data', 'Correct m_{B} data', RooArgSet(b_mass))
    # b_mass_wrong = RooRealVar('mB_wrong', 'Wrong m_{B}', x_min, x_max, 'GeV/#it{c}^{2}')
    b_mass_wrong_data = RooDataSet('mB_wrong_data', 'Wrong m_{B} data', RooArgSet(b_mass))

    # Loop through the events
    for counter in xrange(event_tree.GetEntries()): # So we have to use the old one
        if counter < max_events:
            event_tree.GetEntry(counter)
            mc_event_tree.GetEntry(counter)

            processed_events += 1
            if verbose > 0 and (counter + 1) % 100 == 0: # Print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                rec_ev = reconstruct_mc_truth(event_tree, mc_event_tree, verbose)
                reconstructable_events += 1

                add_to_RooDataSet(b_mass, rec_ev.correct_solution.m_b, b_mass_correct_data)
                add_to_RooDataSet(b_mass, rec_ev.correct_solution.m_b, b_mass_data)
                for wrong_solution in rec_ev.wrong_solutions:
                    add_to_RooDataSet(b_mass, wrong_solution.m_b, b_mass_wrong_data)
                    add_to_RooDataSet(b_mass, wrong_solution.m_b, b_mass_data)

            except UnreconstructableEventError:
                pass

    end_time = time.time()


    # Printing some useful statistics
    if verbose > 0:
        print('{} events have been processed'.format(processed_events))
        print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
        print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    correct_solution_model = SignalModel(name = 'correct_solution_model',
                                         title = 'Correct solution model',
                                         x = b_mass,
                                         mean = RooRealVar('mean_correct', '#mu_{correct}', 5.279, peak_x_min, peak_x_max),
                                         width = RooRealVar('width_narrow_gauss_correct', '#sigma_{correct}', 0.03, 0.01, 0.1),
                                         width_wide = RooRealVar('width_wide_gauss_correct', '#sigma_{wide correct}', 0.3, 0.1, 1.),
                                         alpha = RooRealVar('alpha_correct', '#alpha_{correct}', -1., -10., -0.1),
                                         n = RooRealVar('n_correct', 'n_{correct}', 10., 1., 100.),
                                         narrow_gauss_fraction = RooRealVar('signal_model_narrow_gauss_fraction_correct', 'Fraction of Narrow Gaussian in correct solution model', 0.3, 0., 1.),
                                         cb_fraction = RooRealVar('correct_solution_model_cb_fraction', 'Fraction of Crystal Ball Shape in wrong solution model', 0.3, 0., 1.))

    correct_solution_model.fitTo(b_mass_correct_data, RooFit.Extended(False))
    correct_solution_model.getVariables().Print('v')
    show_plot(b_mass, b_mass_correct_data, n_bins, fit_model = correct_solution_model, components_to_plot = None, draw_legend = draw_legend)

    wrong_solution_model = BackgroundModel(name = 'wrong_solution_model',
                                           title = 'Wrong solution model',
                                           x = b_mass,
                                           mean = RooRealVar('mean_wrong', '#mu_{wrong}', 5.279, peak_x_min, peak_x_max),
                                           width_gauss = RooRealVar('width_gauss_wrong', '#sigma_{Gauss wrong}', 0.2, 0.02, 2.),
                                           width_cb = RooRealVar('width_cb_wrong', '#sigma_{CB wrong}', 0.2, 0.02, 2.),
                                           alpha = RooRealVar('alpha_cb_wrong', '#alpha_{CB wrong}', -1., -10., -0.1),
                                           n = RooRealVar('n_cb_wrong', 'n_{CB wrong}', 10., 1., 100.),
                                           gauss_fraction = RooRealVar('wrong_solution_model_gauss_fraction_wrong', 'Fraction of Gaussian in wrong solution model', 0.5, 0., 1.))

    wrong_solution_model.fitTo(b_mass_wrong_data, RooFit.Extended(False))
    wrong_solution_model.getVariables().Print('v')
    show_plot(b_mass, b_mass_wrong_data, n_bins, fit_model = wrong_solution_model, components_to_plot = None, draw_legend = draw_legend)

    correct_solution_yield = RooRealVar('correct_solution_yield', 'Yield of correct solutions', b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())
    wrong_solution_yield = RooRealVar('wrong_solution_yield', 'Yield of wrong solutions', 3 * b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())

    correct_solution_model.fix()
    wrong_solution_model.fix()
    model = RooAddPdf('model', 'Model to fit', RooArgList(correct_solution_model, wrong_solution_model), RooArgList(correct_solution_yield, wrong_solution_yield))

    model.fitTo(b_mass_data, RooFit.Extended(True))
    model.getVariables().Print('v')
    show_plot(b_mass, b_mass_data, n_bins, fit_model = model, components_to_plot = RooArgList(correct_solution_model, wrong_solution_model), draw_legend = draw_legend)

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
