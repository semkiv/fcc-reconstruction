#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction

    Uses the composite model for fitting signal and background events simultaneously
    Usage: python reconstruction_composite.py -i [INPUT_FILENAME] [-n [MAX_EVENTS]] [-f] [-v]
    Run python reconstruction_composite.py --help for more details
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

# Masses of the particles
m_pi = 0.13957018
m_K = 0.493677
m_tau = 1.77684

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
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max)
    data = RooDataSet('mB', 'Reconstaructed B mass', RooArgSet(b_mass)) # reconstructed B mass values container
    # q_square = RooRealVar('q_square', 'q^{2}', 12.5, 17.5)
    # q_square_data = RooDataSet('q_square_data', 'q square data', RooArgSet(q_square)) # q^2 values container

    # Loop through the events
    for counter, event in enumerate(input_tree):
        if counter < max_events:
            processed_events += 1
            if (counter + 1) % 100 == 0: # print status message every 100 events
                print('Processing event {:.1f} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                m_B = calculate_reconstructed_mass(event, verbose)
                reconstructable_events += 1
                b_mass.setVal(m_B)
                data.add(RooArgSet(b_mass))
            except UnreconstructableEventError:
                pass

                    # q_square.setVal(2 * (m_tau ** 2 - numpy.dot(p_tauplus, p_tauminus) + numpy.sqrt((m_tau ** 2 + p_tauplus ** 2) * (m_tau ** 2 + p_tauminus ** 2))))
                    # q_square_data.add(RooArgSet(q_square))

    end_time = time.time()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(end_time - start_time, float(processed_events) / (end_time - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    if fit:
        # signal model: narrow Gaussian + wide Gaussian + Crystal Ball shape

        # free parameters
        mean_signal = RooRealVar('mean_signal', '#mu_{signal}', 5.279, peak_x_min, peak_x_max)
        width_signal = RooRealVar('width_signal', '#sigma_{signal}', 0.03, 0.01, 0.1)

        # fixed parameters
        # ILD-like
        width_wide_gauss = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.165)
        alpha_signal = RooRealVar('alpha_signal', '#alpha_{signal}', -0.206)
        n_signal = RooRealVar('n_signal', 'n_{signal}', 2.056)
        # progressive
        # width_wide_gauss = RooRealVar('width_wide_gauss', '#sigma_{wide}', 0.151)
        # alpha_signal = RooRealVar('alpha_signal', '#alpha_{signal}', -0.133)
        # n_signal = RooRealVar('n_signal', 'n_{signal}', 2.891)

        narrow_gauss = RooGaussian('narrow_gauss', 'Narrow Gaussian part of signal', b_mass, mean_signal, width_signal)
        wide_gauss = RooGaussian('wide_gauss', 'Wide Gaussian part of signal', b_mass, mean_signal, width_wide_gauss)
        signal_cb = RooCBShape('signal_cb','CB part of signal', b_mass, mean_signal, width_signal, alpha_signal, n_signal)

        # ILD-like
        narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Fraction of narrow Gaussian in signal', 0.127)
        signal_cb_fraction = RooRealVar('signal_cb_fraction', 'Fraction of CB in signal', 0.5)
        # progressive
        # narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Fraction of narrow Gaussian in signal', 0.301)
        # signal_cb_fraction = RooRealVar('signal_cb_fraction', 'Fraction of CB in signal', 0.330)

        signal_model = RooAddPdf('signal_model', 'Signal model', RooArgList(narrow_gauss, signal_cb, wide_gauss), RooArgList(narrow_gauss_fraction, signal_cb_fraction))

        # Bs -> Ds Ds K* (with Ds -> tau nu) background model: Gaussian + Crystal Ball shape
        # ILD-like
        mean_bs_ds2taunu = RooRealVar('mean_bs_ds2taunu', '#mu_{Bs (with Ds -> #tau #nu)}', 4.970)
        width_bs_ds2taunu_gauss = RooRealVar('width_bs_ds2taunu_gauss', '#sigma_{Bs (with Ds -> #tau #nu) Gauss}', 0.196)
        width_bs_ds2taunu_cb = RooRealVar('width_bs_ds2taunu_cb', '#sigma_{Bs (with Ds -> #tau #nu) CB}', 0.078)
        alpha_bs_ds2taunu = RooRealVar('alpha_bs_ds2taunu', '#alpha_{Bs (with Ds -> #tau #nu)}', -0.746)
        n_bs_ds2taunu = RooRealVar('n_bs_ds2taunu', 'n_{Bs (with Ds -> #tau #nu)}', 1.983)
        # progressive
        # mean_bs_ds2taunu = RooRealVar('mean_bs_ds2taunu', '#mu_{Bs (with Ds -> #tau #nu)}', 4.967)
        # width_bs_ds2taunu_gauss = RooRealVar('width_bs_ds2taunu_gauss', '#sigma_{Bs (with Ds -> #tau #nu) Gauss}', 0.191)
        # width_bs_ds2taunu_cb = RooRealVar('width_bs_ds2taunu_cb', '#sigma_{Bs (with Ds -> #tau #nu) CB}', 0.068)
        # alpha_bs_ds2taunu = RooRealVar('alpha_bs_ds2taunu', '#alpha_{Bs (with Ds -> #tau #nu)}', -1.073)
        # n_bs_ds2taunu = RooRealVar('n_bs_ds2taunu', 'n_{Bs (with Ds -> #tau #nu)}', 1.731)

        bs_ds2taunu_gauss = RooGaussian('bs_ds2taunu_gauss', 'Bs (with Ds -> #tau #nu) Gaussian', b_mass, mean_bs_ds2taunu, width_bs_ds2taunu_gauss)
        bs_ds2taunu_cb = RooCBShape('bs_ds2taunu_cb', 'Bs (with Ds -> #tau #nu) CB', b_mass, mean_bs_ds2taunu, width_bs_ds2taunu_cb, alpha_bs_ds2taunu, n_bs_ds2taunu)

        # ILD-like
        bs_ds2taunu_gauss_fraction = RooRealVar('bs_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu) background', 0.243)
        # progressive
        # bs_ds2taunu_gauss_fraction = RooRealVar('bs_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu) background', 0.228)

        bs_ds2taunu_model = RooAddPdf('bs_ds2taunu_model', 'Bs (with Ds -> #tau #nu) background model', RooArgList(bs_ds2taunu_gauss, bs_ds2taunu_cb), RooArgList(bs_ds2taunu_gauss_fraction))

        # Bs -> Ds Ds K* (with one Ds -> tau nu and other Ds -> pi pi pi pi) background model: Gaussian + Crystal Ball shape
        # ILD-like
        mean_bs_ds2taunu_ds2pipipipi = RooRealVar('mean_bs_ds2taunu_ds2pipipipi', '#mu_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', 4.990)
        width_bs_ds2taunu_ds2pipipipi_gauss = RooRealVar('width_bs_ds2taunu_ds2pipipipi_gauss', '#sigma_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) Gauss}', 0.068)
        width_bs_ds2taunu_ds2pipipipi_cb = RooRealVar('width_bs_ds2taunu_ds2pipipipi_cb', '#sigma_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) CB}', 0.190)
        alpha_bs_ds2taunu_ds2pipipipi = RooRealVar('alpha_bs_ds2taunu_ds2pipipipi', '#alpha_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', -0.726)
        n_bs_ds2taunu_ds2pipipipi = RooRealVar('n_bs_ds2taunu_ds2pipipipi', 'n_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', 3.171)
        # progressive
        # mean_bs_ds2taunu_ds2pipipipi = RooRealVar('mean_bs_ds2taunu_ds2pipipipi', '#mu_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', 4.979)
        # width_bs_ds2taunu_ds2pipipipi_gauss = RooRealVar('width_bs_ds2taunu_ds2pipipipi_gauss', '#sigma_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) Gauss}', 0.073)
        # width_bs_ds2taunu_ds2pipipipi_cb = RooRealVar('width_bs_ds2taunu_ds2pipipipi_cb', '#sigma_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) CB}', 0.146)
        # alpha_bs_ds2taunu_ds2pipipipi = RooRealVar('alpha_bs_ds2taunu_ds2pipipipi', '#alpha_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', -0.725)
        # n_bs_ds2taunu_ds2pipipipi = RooRealVar('n_bs_ds2taunu_ds2pipipipi', 'n_{Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi)}', 3.752)

        bs_ds2taunu_ds2pipipipi_gauss = RooGaussian('bs_ds2taunu_ds2pipipipi_gauss', 'Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) Gaussian', b_mass, mean_bs_ds2taunu_ds2pipipipi, width_bs_ds2taunu_ds2pipipipi_gauss)
        bs_ds2taunu_ds2pipipipi_cb = RooCBShape('bs_ds2taunu_ds2pipipipi_cb', 'Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) CB', b_mass, mean_bs_ds2taunu_ds2pipipipi, width_bs_ds2taunu_ds2pipipipi_cb, alpha_bs_ds2taunu_ds2pipipipi, n_bs_ds2taunu_ds2pipipipi)

        # ILD-like
        bs_ds2taunu_ds2pipipipi_gauss_fraction = RooRealVar('bs_ds2taunu_ds2pipipipi_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) background', 0.222)
        # progressive
        # bs_ds2taunu_ds2pipipipi_gauss_fraction = RooRealVar('bs_ds2taunu_ds2pipipipi_gauss_fraction', 'Fraction of Gaussian in Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) background', 0.417)

        bs_ds2taunu_ds2pipipipi_model = RooAddPdf('bs_ds2taunu_ds2pipipipi_model', 'Bs (with Ds -> #tau #nu and Ds -> #pi #pi #pi #pi) background model', RooArgList(bs_ds2taunu_ds2pipipipi_gauss, bs_ds2taunu_ds2pipipipi_cb), RooArgList(bs_ds2taunu_ds2pipipipi_gauss_fraction))

        # Bd -> Ds K* tau nu (with Ds -> tau nu) background model: Gaussian + Crystal Ball shape
        # ILD-like
        mean_bd_ds2taunu_ds2taunu = RooRealVar('mean_bd_ds2taunu_ds2taunu', '#mu_{Bd (with Ds -> #tau #nu)}', 4.894)
        width_bd_ds2taunu_gauss = RooRealVar('width_bd_ds2taunu_gauss', '#sigma_{Bd (with Ds -> #tau #nu) Gauss}', 0.331)
        width_bd_ds2taunu_cb = RooRealVar('width_bd_ds2taunu_cb', '#sigma_{Bd (with Ds -> #tau #nu) CB}', 0.169)
        alpha_bd_ds2taunu = RooRealVar('alpha_bd_ds2taunu', '#alpha_{Bd (with Ds -> #tau #nu)}', -0.904)
        n_bd_ds2taunu = RooRealVar('n_bd_ds2taunu', 'n_{Bd (with Ds -> #tau #nu)}', 3.57)
        # progressive
        # mean_bd_ds2taunu_ds2taunu = RooRealVar('mean_bd_ds2taunu_ds2taunu', '#mu_{Bd (with Ds -> #tau #nu)}', 4.891)
        # width_bd_ds2taunu_gauss = RooRealVar('width_bd_ds2taunu_gauss', '#sigma_{Bd (with Ds -> #tau #nu) Gauss}', 0.565)
        # width_bd_ds2taunu_cb = RooRealVar('width_bd_ds2taunu_cb', '#sigma_{Bd (with Ds -> #tau #nu) CB}', 0.148)
        # alpha_bd_ds2taunu = RooRealVar('alpha_bd_ds2taunu', '#alpha_{Bd (with Ds -> #tau #nu)}', -5.23)
        # n_bd_ds2taunu = RooRealVar('n_bd_ds2taunu', 'n_{Bd (with Ds -> #tau #nu)}', 1.497)

        bd_ds2taunu_gauss = RooGaussian('bd_ds2taunu_gauss', 'Bd (with Ds -> #tau #nu) Gaussian', b_mass, mean_bd_ds2taunu_ds2taunu, width_bd_ds2taunu_gauss)
        bd_ds2taunu_cb = RooCBShape('bd_ds2taunu_cb', 'Bd (with Ds -> #tau #nu) CB', b_mass, mean_bd_ds2taunu_ds2taunu, width_bd_ds2taunu_cb, alpha_bd_ds2taunu, n_bd_ds2taunu)

        # ILD-like
        bd_ds2taunu_gauss_fraction = RooRealVar('bd_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #tau #nu) background', 0.01)
        # progressive
        # bd_ds2taunu_gauss_fraction = RooRealVar('bd_ds2taunu_gauss_fraction', 'Fraction of Gaussian in Bd (with Ds -> #tau #nu) background', 0.29)

        bd_ds2taunu_model = RooAddPdf('bd_ds2taunu_model', 'Bd (with Ds -> #tau #nu) background model', RooArgList(bd_ds2taunu_gauss, bd_ds2taunu_cb), RooArgList(bd_ds2taunu_gauss_fraction))

        signal_yield = RooRealVar('signal_yield', 'Yield of signal', reconstructable_events / 4., 0, reconstructable_events)
        bs_ds2taunu_yield = RooRealVar('bs_ds2taunu_yield', 'Yield of Bs (with Ds -> #tau #nu) background', reconstructable_events / 4., 0, reconstructable_events)
        bs_ds2pipipipi_yield = RooRealVar('bs_ds2pipipipi_yield', 'Yield of Bs (with Ds -> #pi #pi #pi #pi) background', reconstructable_events / 4., 0, reconstructable_events)
        bd_ds2taunu_yield = RooRealVar('bd_ds2taunu_yield', 'Yield of Bd (with Ds -> #tau #nu) background', reconstructable_events / 4., 0, reconstructable_events)

        # composite model
        model = RooAddPdf('model', 'Model to fit', RooArgList(signal_model, bs_ds2taunu_model, bs_ds2taunu_ds2pipipipi_model, bd_ds2taunu_model), RooArgList(signal_yield, bs_ds2taunu_yield, bs_ds2pipipipi_yield, bd_ds2taunu_yield))

        show_mass_plot(b_mass, data, n_bins, fit, model, extended = True, components_to_plot = [signal_model, bs_ds2taunu_model, bs_ds2taunu_ds2pipipipi_model, bd_ds2taunu_model], draw_legend = draw_legend)

    else:
        show_mass_plot(b_mass, data, n_bins)

    # q_square_canvas = TCanvas('q_square_canvas', 'q square', 640, 480)
    # q_square_canvas.cd()
    # q_square_frame = q_square.frame(RooFit.Name('q_square'), RooFit.Title('q^{2}'), RooFit.Bins(10))
    # q_square_frame.GetXaxis().SetTitle('q^{2}, GeV^{2}')
    # q_square_frame.GetYaxis().SetTitle('Events / (0.5 GeV^{2})')
    # q_square_data.plotOn(q_square_frame)
    #
    # q_square_frame.Draw()
    # q_square_canvas.Update()


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
