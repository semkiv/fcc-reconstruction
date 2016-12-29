#!/usr/bin/env python

"""
    Reconstruction script that implements math algorithm of B0 mass reconstruction

    Uses different models for fitting signal and background events
    Usage: python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-b] [-f] [-l] [-q] [-r] [-v]
    Run python reconstruction.py --help for more details
"""

import sys
import argparse
import time
import math
import os

import numpy

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # To prevent TApplication from capturing command line options and breaking argparse. This line has to be the very first line of access of the ROOT module. If any other variables are touched before it, TApplication will be created, which takes the CLI options
from ROOT import TFile, TCanvas, TH1F, TH2F

# This awkward construction serves to suppress the output at RooFit modules import
devnull = open(os.devnull, 'w')
old_stdout_fileno = os.dup(sys.stdout.fileno())
os.dup2(devnull.fileno(), 1)
from ROOT import RooFit, RooRealVar, RooArgSet, RooDataSet, RooGaussian
devnull.close()
os.dup2(old_stdout_fileno, 1)

from utility.common import isclose, reconstruct, show_plot, add_to_RooDataSet
from utility.ReconstructedEvent import UnreconstructableEventError
from utility.fit import SignalModel, BackgroundModel, ResolutionModel

# few constants
NBINS = 100 # Number of bins in the histogram
XMIN = 4.5 # Left bound of the histogram
XMAX = 6.5 # Right bound of the histogram
PEAK_MIN = 4.7 # Minimum value of the peak
PEAK_MAX = 5.5 # Maximum value of the peak

M_TAU = 1.77684

def process(file_name, tree_name, mc_tree_name, max_events, n_bins, x_min, x_max, fit, background, peak_x_min, peak_x_max, draw_legend, plot_q_square, plot_momentum_resolution, verbose):
    """
        A function that forms the main logic of the script

        Args:
        file_name (str): the name of the file to process
        tree_name (str): the name of the tree to process
        mc_tree_name (str): the name of the MC tree (used for momentum resolution)
        max_events (int): the maximum number of events that will be processed
        n_bins (int): the number of bins to be used in the histogram
        x_min (float): the left bound of the histogram
        x_max (float): the right bound of the histogram
        fit (bool): the flag that determines whether the data will be fitted
        background (bool): the flag that determines whether signal or background b_mass_data is processed
        peak_x_min (float): the left bound of the peak
        peak_x_max (float): the right bound of the peak
        draw_legend (bool): the flag that determines whether the histogram legend will be drawn
        plot_q_square (bool): the flag that determines whether the q^2 distribution will be plotted
        plot_momentum_resolution (bool): the flag that determines whether the tau and neutrino momentum resolution distributions will be plotted
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
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max, 'GeV/#it{c}^{2}')
    b_mass_data = RooDataSet('mB_data', 'm_{B} data', RooArgSet(b_mass)) # Storage for reconstructed B mass values

    if plot_q_square:
        # q_square = RooRealVar('q2', 'q^{2}', 12.5, 22.5, 'GeV^{2}/#it{c}^{2}')
        # q_square_data = RooDataSet('q2_data', 'q^{2} data', RooArgSet(q_square)) # q^2 values container
        # q_square_true = RooRealVar('q2_square_true', 'q^{2} true', 12.5, 22.5, 'GeV^{2}/#it{c}^{2}')
        # q_square_true_data = RooDataSet('q2_square_true_data', 'q^{2} true data', RooArgSet(q_square_true)) # q^2 values container
        q_square_hist = TH1F('q2_hist', 'q^{2} distribution', 40, 12.5, 22.5)
        q_square_true_hist = TH1F('q2_true_hist', 'q^{2}_{true} distribution', 40, 12.5, 22.5)

    if plot_momentum_resolution:
        error_p_tauplus_x = RooRealVar('error_p_tauplus_x', '#sigma_{p_{#tau^{+}x}}', -2., 2., 'GeV/#it{c}')
        error_p_tauplus_x_data = RooDataSet('error_p_tauplus_x_data', '#sigma_{p_{#tau^{+}x}} data', RooArgSet(error_p_tauplus_x))
        error_p_tauplus_y = RooRealVar('error_p_tauplus_y', '#sigma_{p_{#tau^{+}y}}', -2., 2., 'GeV/#it{c}')
        error_p_tauplus_y_data = RooDataSet('error_p_tauplus_y_data', '#sigma_{p_{#tau^{+}y}} data', RooArgSet(error_p_tauplus_y))
        error_p_tauplus_z = RooRealVar('error_p_tauplus_z', '#sigma_{p_{#tau^{+}z}}', -2., 2., 'GeV/#it{c}')
        error_p_tauplus_z_data = RooDataSet('error_p_tauplus_z_data', '#sigma_{p_{#tau^{+}z}} data', RooArgSet(error_p_tauplus_z))
        error_p_tauminus_x = RooRealVar('error_p_tauminus_x', '#sigma_{p_{#tau^{-}x}}', -2., 2., 'GeV/#it{c}')
        error_p_tauminus_x_data = RooDataSet('error_p_tauminus_x_data', '#sigma_{p_{#tau^{-}x}} data', RooArgSet(error_p_tauminus_x))
        error_p_tauminus_y = RooRealVar('error_p_tauminus_y', '#sigma_{p_{#tau^{-}y}}', -2., 2., 'GeV/#it{c}')
        error_p_tauminus_y_data = RooDataSet('error_p_tauminus_y_data', '#sigma_{p_{#tau^{-}y}} data', RooArgSet(error_p_tauminus_y))
        error_p_tauminus_z = RooRealVar('error_p_tauminus_z', '#sigma_{p_{#tau^{-}z}}', -2., 2., 'GeV/#it{c}')
        error_p_tauminus_z_data = RooDataSet('error_p_tauminus_z_data', '#sigma_{p_{#tau^{-}z}} data', RooArgSet(error_p_tauminus_z))
        # error_p_nu_tauplus_x = RooRealVar('error_p_nu_tauplus_x', '#epsilon_{p_{#nu#tau^{+}x}}', -5., 5.)
        # error_p_nu_tauplus_x_data = RooDataSet('error_p_nu_tauplus_x_data', '#epsilon_{p_{#nu#tau^{+}x}} data', RooArgSet(error_p_nu_tauplus_x))
        # error_p_nu_tauplus_y = RooRealVar('error_p_nu_tauplus_y', '#epsilon_{p_{#nu#tau^{+}y}}', -5., 5.)
        # error_p_nu_tauplus_y_data = RooDataSet('error_p_nu_tauplus_y_data', '#epsilon_{p_{#nu#tau^{+}y}} data', RooArgSet(error_p_nu_tauplus_y))
        # error_p_nu_tauplus_z = RooRealVar('error_p_nu_tauplus_z', '#epsilon_{p_{#nu#tau^{+}z}}', -5., 5.)
        # error_p_nu_tauplus_z_data = RooDataSet('error_p_nu_tauplus_z_data', '#epsilon_{p_{#nu#tau^{+}z}} data', RooArgSet(error_p_nu_tauplus_z))
        # error_p_nu_tauminus_x = RooRealVar('error_p_nu_tauminus_x', '#epsilon_{p_{#nu#tau^{-}x}}', -5., 5.)
        # error_p_nu_tauminus_x_data = RooDataSet('error_p_nu_tauminus_x_data', '#epsilon_{p_{#nu#tau^{-}x}} data', RooArgSet(error_p_nu_tauminus_x))
        # error_p_nu_tauminus_y = RooRealVar('error_p_nu_tauminus_y', '#epsilon_{p_{#nu#tau^{-}y}}', -5., 5.)
        # error_p_nu_tauminus_y_data = RooDataSet('error_p_nu_tauminus_y_data', '#epsilon_{p_{#nu#tau^{-}y}} data', RooArgSet(error_p_nu_tauminus_y))
        # error_p_nu_tauminus_z = RooRealVar('error_p_nu_tauminus_z', '#epsilon_{p_{#nu#tau^{-}z}}', -5., 5.)
        # error_p_nu_tauminus_z_data = RooDataSet('error_p_nu_tauminus_z_data', '#epsilon_{p_{#nu#tau^{-}z}} data', RooArgSet(error_p_nu_tauminus_z))

    # tauplus_selected_correct_counter = 0
    # tauminus_selected_correct_counter = 0
    # tau_selected_correct_counter = 0

    # tauplusdeltapxpxrec = TH2F('tauplusdeltapxpxrec', '', 50, -50, 50, 50, -50, 50)
    # tauplusdeltapypyrec = TH2F('tauplusdeltapypyrec', '', 50, -50, 50, 50, -50, 50)
    # tauplusdeltapzpzrec = TH2F('tauplusdeltapzpzrec', '', 50, -50, 50, 50, -50, 50)
    # tauplusdeltapprec = TH2F('tauplusdeltapprec', '', 60, 0, 60, 60, -20, 40)
    # tauminusdeltapxpxrec = TH2F('tauminusdeltapxpxrec', '', 50, -50, 50, 50, -50, 50)
    # tauminusdeltapypyrec = TH2F('tauminusdeltapypyrec', '', 50, -50, 50, 50, -50, 50)
    # tauminusdeltapzpzrec = TH2F('tauminusdeltapzpzrec', '', 50, -50, 50, 50, -50, 50)
    # tauminusdeltapprec = TH2F('tauminusdeltapprec', '', 60, 0, 60, 60, -20, 40)
    #
    # sigma_bvertex_all = TH1F('sigma_bvertex_all', '', 100, 0, 0.015)
    # sigma_bvertex_correct = TH1F('sigma_bvertex_correct', '', 100, 0, 0.015)

    # Loop through the events
    for counter in xrange(event_tree.GetEntries()): # So we have to use the old one
        if counter < max_events:
            event_tree.GetEntry(counter)

            # if plot_momentum_resolution:
            mc_event_tree.GetEntry(counter)

            processed_events += 1
            if (counter + 1) % 100 == 0 and verbose > 0: # Print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            try:
                rec_ev = reconstruct(event_tree, verbose)
                reconstructable_events += 1
                add_to_RooDataSet(b_mass, rec_ev.m_b, b_mass_data)

                rec_ev_mc_truth = reconstruct(mc_event_tree, verbose)

                # tauplus_p_mc_truth = math.sqrt(mc_event_tree.tauplus_px ** 2 + mc_event_tree.tauplus_py ** 2 + mc_event_tree.tauplus_pz ** 2)
                # tauminus_p_mc_truth = math.sqrt(mc_event_tree.tauminus_px ** 2 + mc_event_tree.tauminus_py ** 2 + mc_event_tree.tauminus_pz ** 2)
                # tauplus_correct_mc_truth = min(rec_ev_mc_truth.p_tauplus_1, rec_ev_mc_truth.p_tauplus_2, key = lambda x: abs(x - tauplus_p_mc_truth))
                # tauminus_correct_mc_truth = min(rec_ev_mc_truth.p_tauminus_1, rec_ev_mc_truth.p_tauminus_2, key = lambda x: abs(x - tauminus_p_mc_truth))
                # tauplus_correct = rec_ev.p_tauplus_1 if isclose(tauplus_correct_mc_truth, rec_ev_mc_truth.p_tauplus_1) else rec_ev.p_tauplus_2
                # tauminus_correct = rec_ev.p_tauminus_1 if isclose(tauminus_correct_mc_truth, rec_ev_mc_truth.p_tauminus_1) else rec_ev.p_tauminus_2
                #
                # tauplus_selected_correct = isclose(tauplus_correct, rec_ev.p_tauplus.absvalue())
                # tauminus_selected_correct = isclose(tauminus_correct, rec_ev.p_tauminus.absvalue())
                #
                # sigma_bvertex_all.Fill(math.sqrt((event_tree.sv_x - mc_event_tree.sv_x) ** 2 + (event_tree.sv_y - mc_event_tree.sv_y) ** 2 + (event_tree.sv_z - mc_event_tree.sv_z) ** 2))
                #
                # if tauplus_selected_correct:
                #     tauplus_selected_correct_counter += 1
                # if tauminus_selected_correct:
                #     tauminus_selected_correct_counter += 1
                # if tauplus_selected_correct and tauminus_selected_correct:
                #     tau_selected_correct_counter += 1
                #     sigma_bvertex_correct.Fill(math.sqrt((event_tree.sv_x - mc_event_tree.sv_x) ** 2 + (event_tree.sv_y - mc_event_tree.sv_y) ** 2 + (event_tree.sv_z - mc_event_tree.sv_z) ** 2))
                #
                # tauplusdeltapxpxrec.Fill(rec_ev.p_tauplus.px, rec_ev.p_tauplus.px - mc_event_tree.tauplus_px)
                # tauplusdeltapypyrec.Fill(rec_ev.p_tauplus.py, rec_ev.p_tauplus.py - mc_event_tree.tauplus_py)
                # tauplusdeltapzpzrec.Fill(rec_ev.p_tauplus.pz, rec_ev.p_tauplus.pz - mc_event_tree.tauplus_pz)
                # tauplusdeltapprec.Fill(rec_ev.p_tauplus.absvalue(), rec_ev.p_tauplus.absvalue() - numpy.linalg.norm([mc_event_tree.tauplus_px, mc_event_tree.tauplus_py, mc_event_tree.tauplus_pz]))
                # tauminusdeltapxpxrec.Fill(rec_ev.p_tauminus.px, rec_ev.p_tauminus.px - mc_event_tree.tauminus_px)
                # tauminusdeltapypyrec.Fill(rec_ev.p_tauminus.py, rec_ev.p_tauminus.py - mc_event_tree.tauminus_py)
                # tauminusdeltapzpzrec.Fill(rec_ev.p_tauminus.pz, rec_ev.p_tauminus.pz - mc_event_tree.tauminus_pz)
                # tauminusdeltapprec.Fill(rec_ev.p_tauminus.absvalue(), rec_ev.p_tauminus.absvalue() - numpy.linalg.norm([mc_event_tree.tauminus_px, mc_event_tree.tauminus_py, mc_event_tree.tauminus_pz]))

                if plot_q_square:
                    # add_to_RooDataSet(q_square, rec_ev.q_square(), q_square_data)
                    # add_to_RooDataSet(q_square_true, rec_ev_mc_truth.q_square(), q_square_true_data)
                    q_square_hist.Fill(rec_ev.q_square())
                    q_square_true_hist.Fill(rec_ev_mc_truth.q_square())

                if plot_momentum_resolution:
                    # add_to_RooDataSet(error_p_tauplus_x, (rec_ev.p_tauplus.px - mc_event_tree.tauplus_px) / mc_event_tree.tauplus_px, error_p_tauplus_x_data)
                    # add_to_RooDataSet(error_p_tauplus_y, (rec_ev.p_tauplus.py - mc_event_tree.tauplus_py) / mc_event_tree.tauplus_py, error_p_tauplus_y_data)
                    # add_to_RooDataSet(error_p_tauplus_z, (rec_ev.p_tauplus.pz - mc_event_tree.tauplus_pz) / mc_event_tree.tauplus_pz, error_p_tauplus_z_data)
                    #
                    # add_to_RooDataSet(error_p_tauminus_x, (rec_ev.p_tauminus.px - mc_event_tree.tauminus_px) / mc_event_tree.tauminus_px, error_p_tauminus_x_data)
                    # add_to_RooDataSet(error_p_tauminus_y, (rec_ev.p_tauminus.py - mc_event_tree.tauminus_py) / mc_event_tree.tauminus_py, error_p_tauminus_y_data)
                    # add_to_RooDataSet(error_p_tauminus_z, (rec_ev.p_tauminus.pz - mc_event_tree.tauminus_pz) / mc_event_tree.tauminus_pz, error_p_tauminus_z_data)

                    add_to_RooDataSet(error_p_tauplus_x, (rec_ev.p_tauplus.px - mc_event_tree.tauplus_px), error_p_tauplus_x_data)
                    add_to_RooDataSet(error_p_tauplus_y, (rec_ev.p_tauplus.py - mc_event_tree.tauplus_py), error_p_tauplus_y_data)
                    add_to_RooDataSet(error_p_tauplus_z, (rec_ev.p_tauplus.pz - mc_event_tree.tauplus_pz), error_p_tauplus_z_data)

                    add_to_RooDataSet(error_p_tauminus_x, (rec_ev.p_tauminus.px - mc_event_tree.tauminus_px), error_p_tauminus_x_data)
                    add_to_RooDataSet(error_p_tauminus_y, (rec_ev.p_tauminus.py - mc_event_tree.tauminus_py), error_p_tauminus_y_data)
                    add_to_RooDataSet(error_p_tauminus_z, (rec_ev.p_tauminus.pz - mc_event_tree.tauminus_pz), error_p_tauminus_z_data)

                    # add_to_RooDataSet(error_p_nu_tauplus_x, (rec_ev.p_nu_tauplus.px - mc_event_tree.nu_tauplus_px) / mc_event_tree.nu_tauplus_px, error_p_nu_tauplus_x_data)
                    # add_to_RooDataSet(error_p_nu_tauplus_y, (rec_ev.p_nu_tauplus.py - mc_event_tree.nu_tauplus_py) / mc_event_tree.nu_tauplus_py, error_p_nu_tauplus_y_data)
                    # add_to_RooDataSet(error_p_nu_tauplus_z, (rec_ev.p_nu_tauplus.pz - mc_event_tree.nu_tauplus_pz) / mc_event_tree.nu_tauplus_pz, error_p_nu_tauplus_z_data)
                    #
                    # add_to_RooDataSet(error_p_nu_tauminus_x, (rec_ev.p_nu_tauminus.px - mc_event_tree.nu_tauminus_px) / mc_event_tree.nu_tauminus_px, error_p_nu_tauminus_x_data)
                    # add_to_RooDataSet(error_p_nu_tauminus_y, (rec_ev.p_nu_tauminus.py - mc_event_tree.nu_tauminus_py) / mc_event_tree.nu_tauminus_py, error_p_nu_tauminus_y_data)
                    # add_to_RooDataSet(error_p_nu_tauminus_z, (rec_ev.p_nu_tauminus.pz - mc_event_tree.nu_tauminus_pz) / mc_event_tree.nu_tauminus_pz, error_p_nu_tauminus_z_data)

            except UnreconstructableEventError:
                pass

    end_time = time.time()


    # Printing some useful statistics
    if verbose > 0:
        # print('tau+ selection efficiency: {}'.format(float(tauplus_selected_correct_counter) / reconstructable_events))
        # print('tau- selection efficiency: {}'.format(float(tauminus_selected_correct_counter) / reconstructable_events))
        # print('tau selection efficiency: {}'.format(float(tau_selected_correct_counter) / reconstructable_events))

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
                                    gauss_fraction = RooRealVar('background_model_gauss_fraction', 'Fraction of Gaussian in Background Model', 0.3, 0.01, 1.))
        else:
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

        model.fitTo(b_mass_data, RooFit.Extended(False))
        show_plot(b_mass, b_mass_data, n_bins, model, components_to_plot = model.components, draw_legend = draw_legend)

    else:
        show_plot(b_mass, b_mass_data, n_bins)

    if plot_q_square:
        # show_plot(q_square, q_square_data, 40)
        # show_plot(q_square_true, q_square_true_data, 40)
        q2_canvas = TCanvas('q2_canvas', 'q^{2}', 640, 480)
        q_square_true_hist.DrawCopy()
        q_square_hist.SetLineColor(ROOT.kRed)
        q_square_hist.Draw('same')
        # q_square_true_hist.Add(q_square_hist, -1)
        # q_square_true_hist.SetLineColor(ROOT.kGreen)
        # q_square_true_hist.Draw('same')
        raw_input('Press ENTER')


    if plot_momentum_resolution:
        mean = RooRealVar('mean', 'mean of gaussian', 0, -1, 1)
        sigma = RooRealVar('sigma', 'width of gaussian', 0.1, 0.01, 0.4)
        sigma_wide = RooRealVar('sigma_wide', 'width of wide gaussian', 1, 0.01, 2)
        narrow_gauss_fraction = RooRealVar('narrow_gauss_fraction', 'Narrow gaussian fraction', 0.5, 0., 1.)

        error_p_tauplus_x_model = ResolutionModel('error_p_tauplus_x_model', '#sigma_{p_{#tau^{+}x}} Model', error_p_tauplus_x, mean, sigma, sigma_wide, narrow_gauss_fraction)
        error_p_tauplus_x_model.fitTo(error_p_tauplus_x_data, RooFit.Extended(False))
        show_plot(error_p_tauplus_x, error_p_tauplus_x_data, n_bins, error_p_tauplus_x_model)
        show_plot(error_p_tauplus_y, error_p_tauplus_y_data, n_bins)
        show_plot(error_p_tauplus_z, error_p_tauplus_z_data, n_bins)

        error_p_tauminus_x_model = ResolutionModel('error_p_tauminus_x_model', '#sigma_{p_{#tau^{-}x}} Model', error_p_tauminus_x, mean, sigma, sigma_wide, narrow_gauss_fraction)
        error_p_tauminus_x_model.fitTo(error_p_tauminus_x_data, RooFit.Extended(False))
        show_plot(error_p_tauminus_x, error_p_tauminus_x_data, n_bins, error_p_tauminus_x_model)
        show_plot(error_p_tauminus_y, error_p_tauminus_y_data, n_bins)
        show_plot(error_p_tauminus_z, error_p_tauminus_z_data, n_bins)

        # show_plot(error_p_nu_tauplus_x, error_p_nu_tauplus_x_data, n_bins)
        # show_plot(error_p_nu_tauplus_y, error_p_nu_tauplus_y_data, n_bins)
        # show_plot(error_p_nu_tauplus_z, error_p_nu_tauplus_z_data, n_bins)
        #
        # show_plot(error_p_nu_tauminus_x, error_p_nu_tauminus_x_data, n_bins)
        # show_plot(error_p_nu_tauminus_y, error_p_nu_tauminus_y_data, n_bins)
        # show_plot(error_p_nu_tauminus_z, error_p_nu_tauminus_z_data, n_bins);

    # canvas_ptaupluspnu_all = TCanvas('ptaupluspnu_all_canvas', 'p_{#tau^{+}} * p_{#nu} distribution', 1200, 320)
    # canvas_ptaupluspnu_all.Divide(2, 1)
    # frame_ptaupluspnu_all = ptaupluspnu_all.frame(100)
    # ptaupluspnu_all_hist = ptaupluspnu_all_data.plotOn(frame_ptaupluspnu_all)
    # frame_ptaupluspnu_mctruth_all = ptaupluspnu_mctruth_all.frame(100)
    # ptaupluspnu_mctruth_all_hist = ptaupluspnu_mctruth_all_data.plotOn(frame_ptaupluspnu_mctruth_all)
    # # diff_hist =ptaupluspnu_all_hist.Add(ptaupluspnu_mctruth_all_hist, -1)
    # canvas_ptaupluspnu_all.cd(1)
    # frame_ptaupluspnu_all.Draw()
    # canvas_ptaupluspnu_all.cd(2)
    # frame_ptaupluspnu_mctruth_all.Draw()
    # canvas_ptaupluspnu_all.cd(3)
    # diff_hist.Draw()

    # canvas_tauplus_deltapprec = TCanvas('canvas_tauplus_deltapprec', '#tau^{+}', 1280, 360)
    # canvas_tauplus_deltapprec.Divide(4, 1)
    # canvas_tauplus_deltapprec.cd(1)
    # tauplusdeltapxpxrec.Draw('COLZ')
    # canvas_tauplus_deltapprec.cd(2)
    # tauplusdeltapypyrec.Draw('COLZ')
    # canvas_tauplus_deltapprec.cd(3)
    # tauplusdeltapzpzrec.Draw('COLZ')
    # canvas_tauplus_deltapprec.cd(4)
    # tauplusdeltapprec.Draw('COLZ')
    #
    # canvas_tauminus_deltapprec = TCanvas('canvas_tauminus_deltapprec', '#tau^{-}', 1280, 360)
    # canvas_tauminus_deltapprec.Divide(4, 1)
    # canvas_tauminus_deltapprec.cd(1)
    # tauminusdeltapxpxrec.Draw('COLZ')
    # canvas_tauminus_deltapprec.cd(2)
    # tauminusdeltapypyrec.Draw('COLZ')
    # canvas_tauminus_deltapprec.cd(3)
    # tauminusdeltapzpzrec.Draw('COLZ')
    # canvas_tauminus_deltapprec.cd(4)
    # tauminusdeltapprec.Draw('COLZ')
    #
    # canvas_eps_bvertex = TCanvas('eps_bvertex', '', 960, 360)
    # canvas_eps_bvertex.Divide(3, 1)
    # canvas_eps_bvertex.cd(1)
    # sigma_bvertex_all.Draw()
    # canvas_eps_bvertex.cd(2)
    # sigma_bvertex_correct.DrawCopy()
    # canvas_eps_bvertex.cd(3)
    # sigma_bvertex_correct.Divide(sigma_bvertex_all)
    # sigma_bvertex_correct.Draw()
    #
    # raw_input('Press ENTER')

    # show_plot(b_fd, b_fd_data, 40);
    # show_plot(b_fd_xy, b_fd_xy_data, 40);
    # show_plot(tauplus_fd, tauplus_fd_data, 40);
    # show_plot(tauplus_fd_xy, tauplus_fd_xy_data, 40)
    # show_plot(tauminus_fd, tauminus_fd_data, 40);
    # show_plot(tauminus_fd_xy, tauminus_fd_xy_data, 40)
    # show_plot(tv_interdistance, tv_interdistance_data, 40)
    # show_plot(tv_interdistance_xy, tv_interdistance_xy_data, 40)

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
    parser.add_argument('-q', '--q-square', action = 'store_true', help = 'plot q^2 distribution')
    parser.add_argument('-r', '--momentum-resolution', action = 'store_true', help = 'plot tau and neutrino momentum resolution distribution')
    parser.add_argument('-v', '--verbose', type = int, default = 1, help = 'verbosity level')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else sys.maxint

    process(args.input_file, args.tree, args.mctree, max_events, NBINS, XMIN, XMAX, args.fit, args.background, PEAK_MIN, PEAK_MAX, args.with_legend, args.q_square, args.momentum_resolution, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
