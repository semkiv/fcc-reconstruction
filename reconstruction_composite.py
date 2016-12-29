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

from ROOT import TFile, TH1F, TCanvas

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
    # q_square = RooRealVar('q2', 'q^{2}', 12.5, 22.5, 'GeV^{2}/#it{c}^{2}')
    # q_square_data = RooDataSet('q2_data', 'q^{2} data', RooArgSet(q_square)) # q^2 values container
    q_square_hist = TH1F('q2_hist', 'q^{2} distribution', 40, 12.5, 22.5)

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
                # if rec_ev.m_b >= 5.283 - 3 * 0.042 and rec_ev.m_b <= 5.283 + 3 * 0.042:
                # add_to_RooDataSet(q_square, rec_ev.q_square(), q_square_data)
                q_square_hist.Fill(rec_ev.q_square())
            except UnreconstructableEventError:
                pass

    end_time = time.time()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    if fit:
        # signal model
        signal_model = SignalModel(name = 'signal_model',
                                   title = 'Signal Model',
                                   x = b_mass,
                                   mean = RooRealVar('mean_signal', '#mu_{signal}', 5.279, peak_x_min, peak_x_max),
                                   width = RooRealVar('width_signal', '#sigma_{signal}', 0.03, 0.01, 0.1),
                                   width_wide = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.223),
                                   alpha = RooRealVar('alpha_signal', '#alpha_{signal}', -1.128),
                                   n = RooRealVar('n_signal', 'n_{signal}', 0.773),
                                   narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Fraction of narrow Gaussian in signal', 0.243),
                                   cb_fraction = RooRealVar('signal_cb_fraction', 'Fraction of CB in signal', 0.351))

        # Bd -> Ds K* tau nu (with Ds -> tau nu) background model
        bd_ds2taunu_model = BackgroundModel(name = 'bd_ds2taunu_model',
                                            title = 'Bd (with Ds -> #tau #nu) background model',
                                            x = b_mass,
                                            mean = RooRealVar('mean_bd_ds2taunu', '#mu_{Bd (with Ds -> #tau #nu)}', 4.889),
                                            width_gauss = RooRealVar('width_bd_ds2taunu_gauss', '#sigma_{Bd (with Ds -> #tau #nu) Gauss}', 0.143),
                                            width_cb = RooRealVar('width_bd_ds2taunu_cb', '#sigma_{Bd (with Ds -> #tau #nu) CB}', 0.458),
                                            alpha = RooRealVar('alpha_bd_ds2taunu', '#alpha_{Bd (with Ds -> #tau #nu)}', -1.85),
                                            n = RooRealVar('n_bd_ds2taunu', 'n_{Bd (with Ds -> #tau #nu)}', 2.004),
                                            gauss_fraction = RooRealVar('bd_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #tau #nu) background', 0.658))

        # Bd -> Ds K* tau nu (with Ds -> pi pi pi K)
        bd_ds2pipipik_model = BackgroundModel(name = 'bd_ds2pipipik_model',
                                              title = 'Bd (with Ds -> #pi #pi #pi K^{0}_{L}) background model',
                                              x = b_mass,
                                              mean = RooRealVar('mean_bd_ds2pipipik', '#mu_{Bd (with Ds -> #pi #pi #pi K^{0}_{L})}', 4.846),
                                              width_gauss = RooRealVar('width_bd_ds2pipipik_gauss', '#sigma_{Bd (with Ds -> #pi #pi #pi K^{0}_{L}) Gauss}', 0.195),
                                              width_cb = RooRealVar('width_bd_ds2pipipik_cb', '#sigma_{Bd (with Ds -> #pi #pi #pi K^{0}_{L}) CB}', 0.163),
                                              alpha = RooRealVar('alpha_bd_ds2pipipik', '#alpha_{Bd (with Ds -> #pi #pi #pi K^{0}_{L})}', -0.276),
                                              n = RooRealVar('n_bd_ds2pipipik', 'n_{Bd (with Ds -> #pi #pi #pi K^{0}_{L})}', 4.567),
                                              gauss_fraction = RooRealVar('bd_ds2pipipik_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #pi #pi #pi K^{0}_{L}) background', 0.163))

        # Bd -> Ds K* tau nu (with Ds -> pi pi pi pi0)
        bd_ds2pipipipi_model = BackgroundModel(name = 'bd_ds2pipipipi_model',
                                               title = 'Bd (with Ds -> #pi #pi #pi #pi^{0}) background model',
                                               x = b_mass,
                                               mean = RooRealVar('mean_bd_ds2pipipipi', '#mu_{Bd (with Ds -> #pi #pi #pi #pi^{0}}', 4.844),
                                               width_gauss = RooRealVar('width_bd_ds2pipipipi_gauss', '#sigma_{Bd (with Ds -> #pi #pi #pi #pi^{0}) Gauss}', 0.192),
                                               width_cb = RooRealVar('width_bd_ds2pipipipi_cb', '#sigma_{Bd (with Ds -> #pi #pi #pi #pi^{0}) CB}', 0.150),
                                               alpha = RooRealVar('alpha_bd_ds2pipipipi', '#alpha_{Bd (with Ds -> #pi #pi #pi #pi^{0})}', -0.344),
                                               n = RooRealVar('n_bd_ds2pipipipi', 'n_{Bd (with Ds -> #pi #pi #pi #pi^{0})}', 2.843),
                                               gauss_fraction = RooRealVar('bd_ds2pipipipi_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #pi #pi #pi #pi^{0}) background', 0.161))

        # Bs -> Ds Ds K* (with Ds -> tau nu) background model
        bs_ds2taunu_model = BackgroundModel(name = 'bs_ds2taunu_model',
                                            title = 'Bs (with Ds -> #tau #nu) background model',
                                            x = b_mass,
                                            mean = RooRealVar('mean_bs_ds2taunu', '#mu_{Bs (with Ds -> #tau #nu)}', 4.967),
                                            width_gauss = RooRealVar('width_bs_ds2taunu_gauss', '#sigma_{Bs (with Ds -> #tau #nu) Gauss}', 0.185),
                                            width_cb = RooRealVar('width_bs_ds2taunu_cb', '#sigma_{Bs (with Ds -> #tau #nu) CB}', 0.064),
                                            alpha = RooRealVar('alpha_bs_ds2taunu', '#alpha_{Bs (with Ds -> #tau #nu)}', -1.115),
                                            n = RooRealVar('n_bs_ds2taunu', 'n_{Bs (with Ds -> #tau #nu)}', 1.680),
                                            gauss_fraction = RooRealVar('bs_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu) background', 0.230))

        # Bs -> Ds Ds K* (with Ds -> pi pi pi K) background model
        bs_ds2pipipik_model = BackgroundModel(name = 'bs_ds2pipipik_model',
                                              title = 'Bs (with Ds -> #pi #pi #pi K^{0}_{L}) background model',
                                              x = b_mass,
                                              mean = RooRealVar('mean_bs_ds2pipipik', '#mu_{Bs (with Ds -> #pi #pi #pi K^{0}_{L})}', 4.963),
                                              width_gauss = RooRealVar('width_bs_ds2pipipik_gauss', '#sigma_{Bs (with Ds -> #pi #pi #pi K^{0}_{L}) Gauss}', 0.038),
                                              width_cb = RooRealVar('width_bs_ds2pipipik_cb', '#sigma_{Bs (with Ds -> #pi #pi #pi K^{0}_{L}) CB}', 0.124),
                                              alpha = RooRealVar('alpha_bs_ds2pipipik', '#alpha_{Bs (with Ds -> #pi #pi #pi K^{0}_{L})}', -0.73),
                                              n = RooRealVar('n_bs_ds2pipipik', 'n_{Bs (with Ds -> #pi #pi #pi K^{0}_{L})}', 4.904),
                                              gauss_fraction = RooRealVar('bs_ds2pipipik_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #pi #pi #pi K^{0}_{L}) background', 0.345))

        # Bs -> Ds Ds K* (with Ds -> pi pi pi pi0) background model
        bs_ds2pipipipi_model = BackgroundModel(name = 'bs_ds2pipipipi_model',
                                               title = 'Bs (with Ds -> #pi #pi #pi #pi^{0}) background model',
                                               x = b_mass,
                                               mean = RooRealVar('mean_bs_ds2pipipipi', '#mu_{Bs (with Ds -> #pi #pi #pi #pi^{0})}', 4.999),
                                               width_gauss = RooRealVar('width_bs_ds2pipipipi_gauss', '#sigma_{Bs (with Ds -> #pi #pi #pi #pi^{0}) Gauss}', 0.108),
                                               width_cb = RooRealVar('width_bs_ds2pipipipi_cb', '#sigma_{Bs (with Ds -> #pi #pi #pi #pi^{0}) CB}', 0.539),
                                               alpha = RooRealVar('alpha_bs_ds2pipipipi', '#alpha_{Bs (with Ds -> #pi #pi #pi #pi^{0})}', -1.865),
                                               n = RooRealVar('n_bs_ds2pipipipi', 'n_{Bs (with Ds -> #pi #pi #pi #pi^{0})}', 0.167),
                                               gauss_fraction = RooRealVar('bs_ds2pipipipi_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #pi #pi #pi #pi^{0}) background', 0.737))

        # Bs -> Ds Ds K* (with Ds -> pi pi pi K and Ds -> pi pi pi pi0) background model
        bs_ds2pipipik_ds2pipipipi_model = BackgroundModel(name = 'bs_ds2pipipik_ds2pipipipi_model',
                                                          title = 'B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0}) background model',
                                                          x = b_mass,
                                                          mean = RooRealVar('mean_bs_ds2pipipik_ds2pipipipi', '#mu_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0})}', 4.979),
                                                          width_gauss = RooRealVar('width_bs_ds2pipipik_ds2pipipipi_gauss', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0}) Gauss}', 0.072),
                                                          width_cb = RooRealVar('width_bs_ds2pipipik_ds2pipipipi_cb', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0}) CB}', 0.148),
                                                          alpha = RooRealVar('alpha_bs_ds2pipipik_ds2pipipipi', '#alpha_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0})}', -0.655),
                                                          n = RooRealVar('n_bs_ds2pipipik_ds2pipipipi', 'n_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0})}', 6.408),
                                                          gauss_fraction = RooRealVar('bs_ds2pipipik_ds2pipipipi_gauss_fraction', 'Fraction of Gaussian in B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0}) background', 0.449))

        # Bs -> Ds Ds K* (with Ds -> pi pi pi K and Ds -> tau nu) background model
        bs_ds2pipipik_ds2taunu_model = BackgroundModel(name = 'bs_ds2pipipik_ds2taunu_model',
                                                       title = 'B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu) background model',
                                                       x = b_mass,
                                                       mean = RooRealVar('mean_bs_ds2pipipik_ds2taunu', '#mu_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu)}', 4.969),
                                                       width_gauss = RooRealVar('width_bs_ds2pipipik_ds2taunu_gauss', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu) Gauss}', 0.050),
                                                       width_cb = RooRealVar('width_bs_ds2pipipik_ds2taunu_cb', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu) CB}', 0.135),
                                                       alpha = RooRealVar('alpha_bs_ds2pipipik_ds2taunu', '#alpha_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu)}', -0.655),
                                                       n = RooRealVar('n_bs_ds2pipipik_ds2taunu', 'n_{B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu)}', 6.145),
                                                       gauss_fraction = RooRealVar('bs_ds2pipipik_ds2taunu_gauss_fraction', 'Fraction of Gaussian in B_{s} (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu) background', 0.406))

        # Bs -> Ds Ds K* (with Ds -> pi pi pi pi0 and Ds -> tau nu) background model
        bs_ds2pipipipi_ds2taunu_model = BackgroundModel(name = 'bs_ds2pipipipi_ds2taunu_model',
                                                        title = 'B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu) background model',
                                                        x = b_mass,
                                                        mean = RooRealVar('mean_bs_ds2pipipipi_ds2taunu', '#mu_{B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu)}', 4.986),
                                                        width_gauss = RooRealVar('width_bs_ds2pipipipi_ds2taunu_gauss', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu) Gauss}', 0.097),
                                                        width_cb = RooRealVar('width_bs_ds2pipipipi_ds2taunu_cb', '#sigma_{B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu) CB}', 0.535),
                                                        alpha = RooRealVar('alpha_bs_ds2pipipipi_ds2taunu', '#alpha_{B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu)}', -3.925),
                                                        n = RooRealVar('n_bs_ds2pipipipi_ds2taunu', 'n_{B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu)}', 0.775),
                                                        gauss_fraction = RooRealVar('bs_ds2pipipipi_ds2taunu_gauss_fraction', 'Fraction of Gaussian in B_{s} (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu) background', 0.735))

        signal_yield = RooRealVar('signal_yield', 'Yield of signal', b_mass_data.numEntries() / 10., 0, b_mass_data.numEntries())
        bd_ds2taunu_yield = RooRealVar('bd_ds2taunu_yield', 'Yield of Bd (with D_{s} -> #tau #nu) background', b_mass_data.numEntries() / 10., 0, b_mass_data.numEntries())
        bd_ds2pipipik_yield = RooRealVar('bd_ds2pipipik_yield', 'Yield of Bd (with D_{s} -> #pi #pi #pi K^{0}_{L}) background', 127)
        bd_ds2pipipipi_yield = RooRealVar('bd_ds2pipipipi_yield', 'Yield of Bd (with D_{s} -> #pi #pi #pi #pi^{0}) background', b_mass_data.numEntries() / 10., 0, b_mass_data.numEntries())
        bs_ds2taunu_yield = RooRealVar('bs_ds2taunu_yield', 'Yield of Bs (with D_{s} -> #tau #nu) background', 61)
        bs_ds2pipipik_yield = RooRealVar('bs_ds2pipipik_yield', 'Yield of Bs (with D_{s} -> #pi #pi #pi K^{0}_{L}) background', 25)
        bs_ds2pipipipi_yield = RooRealVar('bs_ds2pipipipi_yield', 'Yield of Bs (with D_{s} -> #pi #pi #pi #pi^{0}) background', b_mass_data.numEntries() / 10., 0, b_mass_data.numEntries())
        bs_ds2pipipik_ds2taunu_yield = RooRealVar('bs_ds2pipipik_ds2taunu_yield', 'Yield of Bs (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #tau #nu) background', 50)
        bs_ds2pipipipi_ds2taunu_yield = RooRealVar('bs_ds2pipipipi_ds2taunu_yield', 'Yield of Bs (with D_{s} -> #pi #pi #pi #pi^{0} and D_{s} -> #tau #nu) background', 111)
        bs_ds2pipipik_ds2pipipipi_yield = RooRealVar('bs_ds2pipipik_ds2pipipipi_yield', 'Yield of Bs (with D_{s} -> #pi #pi #pi K^{0}_{L} and D_{s} -> #pi #pi #pi #pi^{0}) background', 63)

        background_models = RooArgList(bd_ds2taunu_model, bd_ds2pipipik_model, bd_ds2pipipipi_model, bs_ds2taunu_model, bs_ds2pipipik_model, bs_ds2pipipipi_model, bs_ds2pipipik_ds2pipipipi_model, bs_ds2pipipik_ds2taunu_model, bs_ds2pipipipi_ds2taunu_model)
        models = RooArgList(bd_ds2taunu_model, bd_ds2pipipik_model, bd_ds2pipipipi_model, bs_ds2taunu_model, bs_ds2pipipik_model, bs_ds2pipipipi_model, bs_ds2pipipik_ds2pipipipi_model, bs_ds2pipipik_ds2taunu_model, bs_ds2pipipipi_ds2taunu_model)
        models.add(signal_model)
        yields = RooArgList(bd_ds2taunu_yield, bd_ds2pipipik_yield, bd_ds2pipipipi_yield, bs_ds2taunu_yield, bs_ds2pipipik_yield, bs_ds2pipipipi_yield, bs_ds2pipipik_ds2pipipipi_yield, bs_ds2pipipik_ds2taunu_yield, bs_ds2pipipipi_ds2taunu_yield)
        yields.add(signal_yield)
        # composite model
        model = RooAddPdf('model', 'Model to fit', models, yields)

        model.fitTo(b_mass_data, RooFit.Extended(True))
        model.getVariables().Print('v')
        show_plot(b_mass, b_mass_data, n_bins, model, components_to_plot = models, draw_legend = draw_legend)
    else:
        show_plot(b_mass, b_mass_data, n_bins)

    # show_plot(q_square, q_square_data, 40)
    q_square_canvas = TCanvas('q2_canvas', 'q^{2} canvas', 640, 480)
    q_square_hist.Draw()
    raw_input('Press ENTER')

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
