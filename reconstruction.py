#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction

    Uses different models for fitting signal and background events
    Usage: python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-b] [-f] [-l] [-v]
    Run python reconstruction.py --help for more details
"""

import sys
import argparse
import time

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # to prevent TApplication from capturing command line options and breaking argparse

from ROOT import TFile
from ROOT import RooRealVar, RooArgSet, RooDataSet

from utility.common import reconstruct, show_plot
from utility.UnreconstructableEventError import UnreconstructableEventError
from utility.SignalModel import SignalModel
from utility.BackgroundModel import BackgroundModel

# few constants
NBINS = 100 # Number of bins in the histogram
XMIN = 4.5 # Left bound of the histogram
XMAX = 6.5 # Right bound of the histogram
PEAK_MIN = 4.7 # Minimum value of the peak
PEAK_MAX = 5.5 # Maximum value of the peak

def process(file_name, tree_name, mc_tree_name, max_events, n_bins, x_min, x_max, fit, background, peak_x_min, peak_x_max, draw_legend, verbose):
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
        background (bool): the flag that determines whether signal or background b_mass_data is processed
        peak_x_min (float): the left bound of the peak
        peak_x_max (float): the right bound of the peak
        draw_legend (bool): the flag that determines whether the histogram legend will be drawn
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
    # b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max)
    # b_mass_data = RooDataSet('mB', 'm_{B} data', RooArgSet(b_mass)) # Storage for reconstructed B mass values
    # q_square = RooRealVar('q2', 'q^{2}', 12.5, 17.5)
    # q_square_data = RooDataSet('q2_data', 'q^{2} data', RooArgSet(q_square)) # q^2 values container
    error_p_tauplus_x = RooRealVar('error_p_tauplus_x', '#epsilon_{p_{#tau^{+}x}}', -2., 2.)
    error_p_tauplus_x_data = RooDataSet('error_p_tauplus_x_data', '#epsilon_{p_{#tau^{+}x}} data', RooArgSet(error_p_tauplus_x))
    error_p_tauplus_y = RooRealVar('error_p_tauplus_y', '#epsilon_{p_{#tau^{+}y}}', -2., 2.)
    error_p_tauplus_y_data = RooDataSet('error_p_tauplus_y_data', '#epsilon_{p_{#tau^{+}y}} data', RooArgSet(error_p_tauplus_y))
    error_p_tauplus_z = RooRealVar('error_p_tauplus_z', '#epsilon_{p_{#tau^{+}z}}', -2., 2.)
    error_p_tauplus_z_data = RooDataSet('error_p_tauplus_z_data', '#epsilon_{p_{#tau^{+}z}} data', RooArgSet(error_p_tauplus_z))
    error_p_tauminus_x = RooRealVar('error_p_tauminus_x', '#epsilon_{p_{#tau^{-}x}}', -2., 2.)
    error_p_tauminus_x_data = RooDataSet('error_p_tauminus_x_data', '#epsilon_{p_{#tau^{-}x}} data', RooArgSet(error_p_tauminus_x))
    error_p_tauminus_y = RooRealVar('error_p_tauminus_y', '#epsilon_{p_{#tau^{-}y}}', -2., 2.)
    error_p_tauminus_y_data = RooDataSet('error_p_tauminus_y_data', '#epsilon_{p_{#tau^{-}y}} data', RooArgSet(error_p_tauminus_y))
    error_p_tauminus_z = RooRealVar('error_p_tauminus_z', '#epsilon_{p_{#tau^{-}z}}', -2., 2.)
    error_p_tauminus_z_data = RooDataSet('error_p_tauminus_z_data', '#epsilon_{p_{#tau^{-}z}} data', RooArgSet(error_p_tauminus_z))
    error_p_nu_tauplus_x = RooRealVar('error_p_nu_tauplus_x', '#epsilon_{p_{#nu#tau^{+}x}}', -5., 5.)
    error_p_nu_tauplus_x_data = RooDataSet('error_p_nu_tauplus_x_data', '#epsilon_{p_{#nu#tau^{+}x}} data', RooArgSet(error_p_nu_tauplus_x))
    error_p_nu_tauplus_y = RooRealVar('error_p_nu_tauplus_y', '#epsilon_{p_{#nu#tau^{+}y}}', -5., 5.)
    error_p_nu_tauplus_y_data = RooDataSet('error_p_nu_tauplus_y_data', '#epsilon_{p_{#nu#tau^{+}y}} data', RooArgSet(error_p_nu_tauplus_y))
    error_p_nu_tauplus_z = RooRealVar('error_p_nu_tauplus_z', '#epsilon_{p_{#nu#tau^{+}z}}', -5., 5.)
    error_p_nu_tauplus_z_data = RooDataSet('error_p_nu_tauplus_z_data', '#epsilon_{p_{#nu#tau^{+}z}} data', RooArgSet(error_p_nu_tauplus_z))
    error_p_nu_tauminus_x = RooRealVar('error_p_nu_tauminus_x', '#epsilon_{p_{#nu#tau^{-}x}}', -5., 5.)
    error_p_nu_tauminus_x_data = RooDataSet('error_p_nu_tauminus_x_data', '#epsilon_{p_{#nu#tau^{-}x}} data', RooArgSet(error_p_nu_tauminus_x))
    error_p_nu_tauminus_y = RooRealVar('error_p_nu_tauminus_y', '#epsilon_{p_{#nu#tau^{-}y}}', -5., 5.)
    error_p_nu_tauminus_y_data = RooDataSet('error_p_nu_tauminus_y_data', '#epsilon_{p_{#nu#tau^{-}y}} data', RooArgSet(error_p_nu_tauminus_y))
    error_p_nu_tauminus_z = RooRealVar('error_p_nu_tauminus_z', '#epsilon_{p_{#nu#tau^{-}z}}', -5., 5.)
    error_p_nu_tauminus_z_data = RooDataSet('error_p_nu_tauminus_z_data', '#epsilon_{p_{#nu#tau^{-}z}} data', RooArgSet(error_p_nu_tauminus_z))

    # Loop through the events
    # for counter, (event, mc_event) in enumerate(zip(event_tree, mc_event_tree)): # this finest construction doesn't work
    for counter in xrange(event_tree.GetEntries()): # so we have to use an old one
        if counter < max_events:
            event_tree.GetEntry(counter)
            mc_event_tree.GetEntry(counter)
            processed_events += 1
            if (counter + 1) % 100 == 0: # print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                rec_ev = reconstruct(event_tree, verbose)
                reconstructable_events += 1

                # b_mass.setVal(rec_ev.m_b)
                # b_mass_data.add(RooArgSet(b_mass))
                #
                # q_square.setVal(rec_ev.q_square())
                # q_square_data.add(RooArgSet(q_square))

                error_p_tauplus_x.setVal((rec_ev.p_tauplus.px - mc_event_tree.tauplus_px) / mc_event_tree.tauplus_px)
                error_p_tauplus_x_data.add(RooArgSet(error_p_tauplus_x))
                error_p_tauplus_y.setVal((rec_ev.p_tauplus.py - mc_event_tree.tauplus_py) / mc_event_tree.tauplus_py)
                error_p_tauplus_y_data.add(RooArgSet(error_p_tauplus_y))
                error_p_tauplus_z.setVal((rec_ev.p_tauplus.pz - mc_event_tree.tauplus_pz) / mc_event_tree.tauplus_pz)
                error_p_tauplus_z_data.add(RooArgSet(error_p_tauplus_z))

                error_p_tauminus_x.setVal((rec_ev.p_tauminus.px - mc_event_tree.tauminus_px) / mc_event_tree.tauminus_px)
                error_p_tauminus_x_data.add(RooArgSet(error_p_tauminus_x))
                error_p_tauminus_y.setVal((rec_ev.p_tauminus.py - mc_event_tree.tauminus_py) / mc_event_tree.tauminus_py)
                error_p_tauminus_y_data.add(RooArgSet(error_p_tauminus_y))
                error_p_tauminus_z.setVal((rec_ev.p_tauminus.pz - mc_event_tree.tauminus_pz) / mc_event_tree.tauminus_pz)
                error_p_tauminus_z_data.add(RooArgSet(error_p_tauminus_z))

                error_p_nu_tauplus_x.setVal((rec_ev.p_nu_tauplus.px - mc_event_tree.nu_tauplus_px) / mc_event_tree.nu_tauplus_px)
                error_p_nu_tauplus_x_data.add(RooArgSet(error_p_nu_tauplus_x))
                error_p_nu_tauplus_y.setVal((rec_ev.p_nu_tauplus.py - mc_event_tree.nu_tauplus_py) / mc_event_tree.nu_tauplus_py)
                error_p_nu_tauplus_y_data.add(RooArgSet(error_p_nu_tauplus_y))
                error_p_nu_tauplus_z.setVal((rec_ev.p_nu_tauplus.pz - mc_event_tree.nu_tauplus_pz) / mc_event_tree.nu_tauplus_pz)
                error_p_nu_tauplus_z_data.add(RooArgSet(error_p_nu_tauplus_z))

                error_p_nu_tauminus_x.setVal((rec_ev.p_nu_tauminus.px - mc_event_tree.nu_tauminus_px) / mc_event_tree.nu_tauminus_px)
                error_p_nu_tauminus_x_data.add(RooArgSet(error_p_nu_tauminus_x))
                error_p_nu_tauminus_y.setVal((rec_ev.p_nu_tauminus.py - mc_event_tree.nu_tauminus_py) / mc_event_tree.nu_tauminus_py)
                error_p_nu_tauminus_y_data.add(RooArgSet(error_p_nu_tauminus_y))
                error_p_nu_tauminus_z.setVal((rec_ev.p_nu_tauminus.pz - mc_event_tree.nu_tauminus_pz) / mc_event_tree.nu_tauminus_pz)
                error_p_nu_tauminus_z_data.add(RooArgSet(error_p_nu_tauminus_z))

            except UnreconstructableEventError:
                pass

    end_time = time.time()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    if fit:
        if background:
            model = BackgroundModel(name = 'background_model',
                                    title = 'Background Model',
                                    x = b_mass,
                                    mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max),
                                    width_cb = RooRealVar('width_cb', '#sigma_{CB}', 0.2, 0.02, 1.),
                                    width_gauss = RooRealVar('width_gauss', '#sigma_{Gauss}', 0.2, 0.02, 1.),
                                    alpha = RooRealVar('alpha_cb', '#alpha_{CB}', -1., -10., -0.1),
                                    n = RooRealVar('n_cb', 'n_{CB}', 1., 0., 10.),
                                    gauss_fraction = RooRealVar('background_model_gauss_fraction', 'Fraction of Gaussian in Background Model', 0.3, 0.01, 1.)
                                    )
        else:
            model = SignalModel(name = 'signal_model',
                                title = 'Signal Model',
                                x = b_mass,
                                mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max),
                                width = RooRealVar('width_narrow_gauss', '#sigma', 0.03, 0.01, 0.1),
                                width_wide = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.3, 0.1, 1.),
                                alpha = RooRealVar('alpha', '#alpha', -1, -10., -0.1),
                                n = RooRealVar('n', 'n', 2., 0.1, 10.),
                                narrow_gauss_fraction = RooRealVar('signal_model_narrow_gauss_fraction', 'Fraction of Narrow Gaussian in Signal Model', 0.3, 0.01, 1.),
                                cb_fraction = RooRealVar('signal_model_cb_fraction', 'Fraction of Crystal Ball Shape in Signal Model', 0.3, 0.01, 1.)
                                )

        show_plot(b_mass, b_mass_data, 'GeV/#it{c}^{2}', n_bins, fit, model.pdf, extended = False, components_to_plot = model.components, draw_legend = draw_legend)

    # else:
        # show_plot(b_mass, b_mass_data, 'GeV/#it{c}^{2}', n_bins)

    # show_plot(q_square, q_square_data, 'GeV^{2}/#it{c}^{2}', n_bins)

    show_plot(error_p_tauplus_x, error_p_tauplus_x_data, None, n_bins)
    show_plot(error_p_tauplus_y, error_p_tauplus_y_data, None, n_bins)
    show_plot(error_p_tauplus_z, error_p_tauplus_z_data, None, n_bins)

    show_plot(error_p_tauminus_x, error_p_tauminus_x_data, None, n_bins)
    show_plot(error_p_tauminus_y, error_p_tauminus_y_data, None, n_bins)
    show_plot(error_p_tauminus_z, error_p_tauminus_z_data, None, n_bins)

    show_plot(error_p_nu_tauplus_x, error_p_nu_tauplus_x_data, None, n_bins)
    show_plot(error_p_nu_tauplus_y, error_p_nu_tauplus_y_data, None, n_bins)
    show_plot(error_p_nu_tauplus_z, error_p_nu_tauplus_z_data, None, n_bins)

    show_plot(error_p_nu_tauminus_x, error_p_nu_tauminus_x_data, None, n_bins)
    show_plot(error_p_nu_tauminus_y, error_p_nu_tauminus_y_data, None, n_bins)
    show_plot(error_p_nu_tauminus_z, error_p_nu_tauminus_z_data, None, n_bins)

def main(argv):
    """The main function. Parses the command line arguments passed to the script and then runs the process function"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required = True, help = 'name of the file to process')
    parser.add_argument('-t', '--tree', type = str, default = 'Events', help = 'name of the event tree')
    parser.add_argument('-m', '--mctree', type = str, default = 'MCTruth', help = 'name of the tree with Monte-Carlo truth events')
    parser.add_argument('-n', '--nevents', type = int, help = 'maximum number of events to process')
    parser.add_argument('-f', '--fit', action = 'store_true', help = 'fit the histogram')
    parser.add_argument('-b', '--background', action = 'store_true', help = 'use fit model for background events')
    parser.add_argument('-l', '--with-legend', action = 'store_true', help = 'draw legend')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'run with increased verbosity')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else sys.maxint

    process(args.input_file, args.tree, args.mctree, max_events, NBINS, XMIN, XMAX, args.fit, args.background, PEAK_MIN, PEAK_MAX, args.with_legend, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
