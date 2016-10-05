#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction

    Uses the composite model for fitting signal and background events simultaneously
    Usage: python reconstruction_composite.py -i [INPUT_FILENAME] [-n [MAX_EVENTS]] [-f] [-l] [-v]
    Run python reconstruction_composite.py --help for more details
"""

import os
import sys
import argparse
import time

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # to prevent TApplication from capturing command line options and breaking argparse. It must be placed right after module import

from ROOT import TFile

# This awkward construction serves to suppress the output at RooFit modules import
devnull = open(os.devnull, 'w')
old_stdout_fileno = os.dup(sys.stdout.fileno())
os.dup2(devnull.fileno(), 1)
from ROOT import RooFit, RooRealVar, RooArgList, RooArgSet, RooDataSet, RooAddPdf, RooCBShape, RooGaussian
devnull.close()
os.dup2(old_stdout_fileno, 1)

from utility.common import reconstruct, show_plot, add_to_RooDataSet
from utility.ReconstructedEvent import UnreconstructableEventError
from utility.fit import SignalModel, BackgroundModel

# few constants
NBINS = 100 # Number of bins in the histogram
XMIN = 4.5 # Left bound of the histogram
XMAX = 6.5 # Right bound of the histogram
PEAK_MIN = 4.7 # Minimum value of the peak
PEAK_MAX = 5.5 # Maximum value of the peak

def process(file_name, max_events, n_bins, x_min, x_max, fit, peak_x_min, peak_x_max, draw_legend, verbose):
    """
        A function that forms the main logic of the script

        Args:
        file_name (str): the name of the file to process
        max_events (int): the maximum number of events that will be processed
        n_bins (int): the number of bins to be used in the histogram
        x_min (float): the left bound of the histogram
        x_max (float): the right bound of the histogram
        fit (bool): the flag that determines whether the data will be fitted
        peak_x_min (float): the left bound of the peak
        peak_x_max (float): the right bound of the peak
        draw_legend (bool): the flag that determines whether the histogram legend will be drawn
        verbose (bool): the flag that switches increased verbosity
    """

    start_time = time.time()
    last_timestamp = time.time()

    # Opening the file and getting the branch
    input_file = TFile(file_name, 'read')
    input_tree = input_file.Get('Events')

    # Event counters
    processed_events = 0 # Number of processed events
    reconstructable_events = 0 # Events with valid tau+ and tau- decay vertex

    # Variables for RooFit
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max, 'GeV/#it{c}^{2}')
    b_mass_data = RooDataSet('mB', 'm_{B} data', RooArgSet(b_mass)) # reconstructed B mass values container
    q_square = RooRealVar('q2', 'q^{2}', 12.5, 17.5, 'GeV^{2}/#it{c}^{2}')
    q_square_data = RooDataSet('q2_data', 'q^{2} data', RooArgSet(q_square)) # q^2 values container

    # Loop through the events
    for counter, event in enumerate(input_tree):
        if counter < max_events:
            processed_events += 1
            if (counter + 1) % 100 == 0: # print status message every 100 events
                print('Processing event {:.1f} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                rec_ev = reconstruct(event, verbose)
                reconstructable_events += 1

                add_to_RooDataSet(b_mass, rec_ev.m_b, b_mass_data)
                add_to_RooDataSet(q_square, rec_ev.q_square(), q_square_data)
            except UnreconstructableEventError:
                pass

    end_time = time.time()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    if fit:
        # signal model
        # ILD-like
        # signal_model = SignalModel(name = 'signal_model',
        #                            title = 'Signal Model',
        #                            x = b_mass,
        #                            mean = RooRealVar('mean_signal', '#mu_{signal}', 5.279, peak_x_min, peak_x_max),
        #                            width = RooRealVar('width_signal', '#sigma_{signal}', 0.03, 0.01, 0.1),
        #                            width_wide = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.131),
        #                            alpha = RooRealVar('alpha_signal', '#alpha_{signal}', -0.280),
        #                            n = RooRealVar('n_signal', 'n_{signal}', 99.93),
        #                            narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Fraction of narrow Gaussian in signal', 0.158),
        #                            cb_fraction = RooRealVar('signal_cb_fraction', 'Fraction of CB in signal', 0.404))
        # progressive
        signal_model = SignalModel(name = 'signal_model',
                                   title = 'Signal Model',
                                   x = b_mass,
                                   mean = RooRealVar('mean_signal', '#mu_{signal}', 5.279, peak_x_min, peak_x_max),
                                   width = RooRealVar('width_signal', '#sigma_{signal}', 0.03, 0.01, 0.1),
                                   width_wide = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.110),
                                   alpha = RooRealVar('alpha_signal', '#alpha_{signal}', -0.445),
                                   n = RooRealVar('n_signal', 'n_{signal}', 4.85),
                                   narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Fraction of narrow Gaussian in signal', 0.323),
                                   cb_fraction = RooRealVar('signal_cb_fraction', 'Fraction of CB in signal', 0.285))

        # Wrong signal model
        # ILD-like
        # wrong_signal_model = BackgroundModel(name = 'wrong_signal_model',
        #                                      title = 'Wrong signal model',
        #                                      x = b_mass,
        #                                      mean = RooRealVar('mean_wrong_signal', '#mu_{wrong_signal}', 5.344),
        #                                      width_gauss = RooRealVar('width_wrong_signal', '#sigma_{wrong signal}', 0.371),
        #                                      width_cb = RooRealVar('width_wrong_signal_cb', '#sigma_{wrong signal CB}', 0.267),
        #                                      alpha = RooRealVar('alpha_wrong_signal', '#alpha_{wrong signal}', -0.399),
        #                                      n = RooRealVar('n_wrong_signal', 'n_{wrong signal}', 100.),
        #                                      gauss_fraction = RooRealVar('wrong_signal_gauss_fraction', 'Fraction of Gaussian in wrong signal background', 0.))
        # progressive
        wrong_signal_model = BackgroundModel(name = 'wrong_signal_model',
                                             title = 'Wrong signal model',
                                             x = b_mass,
                                             mean = RooRealVar('mean_wrong_signal', '#mu_{wrong_signal}', 5.327),
                                             width_gauss = RooRealVar('width_wrong_signal', '#sigma_{wrong signal}', 0.193),
                                             width_cb = RooRealVar('width_wrong_signal_cb', '#sigma_{wrong signal CB}', 0.251),
                                             alpha = RooRealVar('alpha_wrong_signal', '#alpha_{wrong signal}', -0.439),
                                             n = RooRealVar('n_wrong_signal', 'n_{wrong signal}', 100.),
                                             gauss_fraction = RooRealVar('wrong_signal_gauss_fraction', 'Fraction of Gaussian in wrong signal background', 0.))

        # Bs -> Ds Ds K* (with Ds -> tau nu) background model
        # ILD-like
        # bs_ds2taunu_model = BackgroundModel(name = 'bs_ds2taunu_model',
        #                                     title = 'Bs (with Ds -> #tau #nu) background model',
        #                                     x = b_mass,
        #                                     mean = RooRealVar('mean_bs_ds2taunu', '#mu_{Bs (with Ds -> #tau #nu)}', 4.970),
        #                                     width_gauss = RooRealVar('width_bs_ds2taunu_gauss', '#sigma_{Bs (with Ds -> #tau #nu) Gauss}', 0.196),
        #                                     width_cb = RooRealVar('width_bs_ds2taunu_cb', '#sigma_{Bs (with Ds -> #tau #nu) CB}', 0.078),
        #                                     alpha = RooRealVar('alpha_bs_ds2taunu', '#alpha_{Bs (with Ds -> #tau #nu)}', -0.746),
        #                                     n = RooRealVar('n_bs_ds2taunu', 'n_{Bs (with Ds -> #tau #nu)}', 1.983),
        #                                     gauss_fraction = RooRealVar('bs_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu) background', 0.243))
        # progressive
        bs_ds2taunu_model = BackgroundModel(name = 'bs_ds2taunu_model',
                                            title = 'Bs (with Ds -> #tau #nu) background model',
                                            x = b_mass,
                                            mean = RooRealVar('mean_bs_ds2taunu', '#mu_{Bs (with Ds -> #tau #nu)}', 4.967),
                                            width_gauss = RooRealVar('width_bs_ds2taunu_gauss', '#sigma_{Bs (with Ds -> #tau #nu) Gauss}', 0.191),
                                            width_cb = RooRealVar('width_bs_ds2taunu_cb', '#sigma_{Bs (with Ds -> #tau #nu) CB}', 0.068),
                                            alpha = RooRealVar('alpha_bs_ds2taunu', '#alpha_{Bs (with Ds -> #tau #nu)}', -1.073),
                                            n = RooRealVar('n_bs_ds2taunu', 'n_{Bs (with Ds -> #tau #nu)}', 1.731),
                                            gauss_fraction = RooRealVar('bs_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu) background', 0.228))

        # Bd -> Ds K* tau nu (with Ds -> tau nu) background model
        # ILD-like
        # bd_ds2taunu_model = BackgroundModel(name = 'bd_ds2taunu_model',
        #                                     title = 'Bd (with Ds -> #tau #nu) background model',
        #                                     x = b_mass,
        #                                     mean = RooRealVar('mean_bd_ds2taunu_ds2taunu', '#mu_{Bd (with Ds -> #tau #nu)}', 4.894),
        #                                     width_gauss = RooRealVar('width_bd_ds2taunu_gauss', '#sigma_{Bd (with Ds -> #tau #nu) Gauss}', 0.331),
        #                                     width_cb = RooRealVar('width_bd_ds2taunu_cb', '#sigma_{Bd (with Ds -> #tau #nu) CB}', 0.169),
        #                                     alpha = RooRealVar('alpha_bd_ds2taunu', '#alpha_{Bd (with Ds -> #tau #nu)}', -0.904),
        #                                     n = RooRealVar('n_bd_ds2taunu', 'n_{Bd (with Ds -> #tau #nu)}', 3.57),
        #                                     gauss_fraction = RooRealVar('bd_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #tau #nu) background', 0.01))
        # progressive
        bd_ds2taunu_model = BackgroundModel(name = 'bd_ds2taunu_model',
                                            title = 'Bd (with Ds -> #tau #nu) background model',
                                            x = b_mass,
                                            mean = RooRealVar('mean_bd_ds2taunu_ds2taunu', '#mu_{Bd (with Ds -> #tau #nu)}', 4.891),
                                            width_gauss = RooRealVar('width_bd_ds2taunu_gauss', '#sigma_{Bd (with Ds -> #tau #nu) Gauss}', 0.565),
                                            width_cb = RooRealVar('width_bd_ds2taunu_cb', '#sigma_{Bd (with Ds -> #tau #nu) CB}', 0.148),
                                            alpha = RooRealVar('alpha_bd_ds2taunu', '#alpha_{Bd (with Ds -> #tau #nu)}', -5.23),
                                            n = RooRealVar('n_bd_ds2taunu', 'n_{Bd (with Ds -> #tau #nu)}', 1.497),
                                            gauss_fraction = RooRealVar('bd_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #tau #nu) background', 0.29))

        signal_yield = RooRealVar('signal_yield', 'Yield of signal', b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())
        wrong_signal_yield = RooRealVar('wrong_signal_yield', 'Yield of wrong signal', b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())
        bs_ds2taunu_yield = RooRealVar('bs_ds2taunu_yield', 'Yield of Bs (with Ds -> #tau #nu) background', b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())
        bd_ds2taunu_yield = RooRealVar('bd_ds2taunu_yield', 'Yield of Bd (with Ds -> #tau #nu) background', b_mass_data.numEntries() / 4., 0, b_mass_data.numEntries())

        # composite model
        model = RooAddPdf('model', 'Model to fit', RooArgList(signal_model, wrong_signal_model, bs_ds2taunu_model, bd_ds2taunu_model), RooArgList(signal_yield, wrong_signal_yield, bs_ds2taunu_yield, bd_ds2taunu_yield))

        model.fitTo(b_mass_data, RooFit.Extended(True))
        model.getVariables().Print('v')
        show_plot(b_mass, b_mass_data, n_bins, model, components_to_plot = RooArgList(signal_model, wrong_signal_model, bs_ds2taunu_model, bd_ds2taunu_model), draw_legend = draw_legend)

    else:
        show_plot(b_mass, b_mass_data, n_bins)

    show_plot(q_square, q_square_data, n_bins)

def main(argv):
    """The main function. Parses the command line arguments passed to the script and then runs the process function"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required = True, help = 'name of the file to process')
    parser.add_argument('-n', '--nevents', type = int, help = 'maximum number of events to process')
    parser.add_argument('-f', '--fit', action = 'store_true', help = 'fit the histogram')
    parser.add_argument('-l', '--with-legend', action = 'store_true', help = 'draw legend')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'run with increased verbosity')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else sys.maxint

    process(args.input_file, max_events, NBINS, XMIN, XMAX, args.fit, PEAK_MIN, PEAK_MAX, args.with_legend, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
