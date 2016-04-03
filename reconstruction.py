#!/usr/bin/env python

## Reconstruction script that implements math algorithm of B0 mass reconstruction
#  Uses different models for fitting signal and background events
#  Usage: python reconstruction.py -i [INPUT_FILENAME] [-t [TREE_NAME]] [-n [MAX_EVENTS]] [-b] [-f] [-v]
#  Run python reconstruction.py --help for more details

import sys
import os
import argparse
import time

import numpy

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True # to prevent TApplication from capturing command line options and breaking argparse

from ROOT import gROOT
from ROOT import gStyle
from ROOT import TFile
from ROOT import TCanvas
from ROOT import TPaveText
from ROOT import TPad
from ROOT import TLine
from ROOT import RooFit
from ROOT import RooRealVar
from ROOT import RooAddPdf
from ROOT import RooArgList
from ROOT import RooArgSet
from ROOT import RooDataSet
from ROOT import RooCBShape
from ROOT import RooGaussian

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

def process(file_name, tree_name, max_events, n_bins, x_min, x_max, fit, background, peak_x_min, peak_x_max, verbose):
    start_time = time.time()
    last_timestamp = time.time()

    # Nice looking plots
    gROOT.ProcessLine('.x ' + os.environ.get('FCC') + 'lhcbstyle.C')
    gStyle.SetOptStat(0)

    # Opening the file and getting the branch
    input_file = TFile(file_name, 'read')
    input_tree = input_file.Get(tree_name)

    # Event countes
    processed_events = 0 # Number of processed events
    reconstructable_events = 0 # Events with valid tau+ and tau- decay vertex

    # Setting numpy precision
    if verbose:
        numpy.set_printoptions(12)

    # Variables for RooFit
    b_mass = RooRealVar('mB', 'm_{B}', x_min, x_max)
    data = RooDataSet('mB', 'Reconstaructed B mass', RooArgSet(b_mass)) # Storage for reconstructed B mass values

    # Loop through the events
    for counter, event in enumerate(input_tree):
        if max_events == None or counter < max_events:
            processed_events += 1
            if (counter + 1) % 100 == 0: # print status message every 100 events
                print('Processing event {} ({:.1f} events / s)'.format(counter + 1, 100. / (time.time() - last_timestamp)))
                last_timestamp = time.time()

            # Reading data necessary for reconstruction
            p_pi1_tauplus_x = event.pi1_tauplus_px
            p_pi1_tauplus_y = event.pi1_tauplus_py
            p_pi1_tauplus_z = event.pi1_tauplus_pz

            p_pi2_tauplus_x = event.pi2_tauplus_px
            p_pi2_tauplus_y = event.pi2_tauplus_py
            p_pi2_tauplus_z = event.pi2_tauplus_pz

            p_pi3_tauplus_x = event.pi3_tauplus_px
            p_pi3_tauplus_y = event.pi3_tauplus_py
            p_pi3_tauplus_z = event.pi3_tauplus_pz

            p_pi1_tauminus_x = event.pi1_tauminus_px
            p_pi1_tauminus_y = event.pi1_tauminus_py
            p_pi1_tauminus_z = event.pi1_tauminus_pz

            p_pi2_tauminus_x = event.pi2_tauminus_px
            p_pi2_tauminus_y = event.pi2_tauminus_py
            p_pi2_tauminus_z = event.pi2_tauminus_pz

            p_pi3_tauminus_x = event.pi3_tauminus_px
            p_pi3_tauminus_y = event.pi3_tauminus_py
            p_pi3_tauminus_z = event.pi3_tauminus_pz

            p_pi_K_x = event.pi_k_px
            p_pi_K_y = event.pi_k_py
            p_pi_K_z = event.pi_k_pz

            p_K_x = event.k_px
            p_K_y = event.k_py
            p_K_z = event.k_pz

            pv_x = event.pv_x
            pv_y = event.pv_y
            pv_z = event.pv_z
            sv_x = event.sv_x
            sv_y = event.sv_y
            sv_z = event.sv_z
            tv_tauplus_x = event.tv_tauplus_x
            tv_tauplus_y = event.tv_tauplus_y
            tv_tauplus_z = event.tv_tauplus_z
            tv_tauminus_x = event.tv_tauminus_x
            tv_tauminus_y = event.tv_tauminus_y
            tv_tauminus_z = event.tv_tauminus_z
            if verbose:
                print('Primary vertex: ({:.12f}, {:.12f}, {:.12f})'.format(pv_x, pv_y, pv_z))
                print('Secondary vertex: ({:.12f}, {:.12f}, {:.12f})'.format(sv_x, sv_y, sv_z))
                print('Tertiary vertex (tau+): ({:.12f}, {:.12f}, {:.12f})'.format(tv_tauplus_x, tv_tauplus_y, tv_tauplus_z))
                print('Tertiary vertex (tau-): ({:.12f}, {:.12f}, {:.12f})'.format(tv_tauminus_x, tv_tauminus_y, tv_tauminus_z))

            p_pi1_tauplus = numpy.array([p_pi1_tauplus_x, p_pi1_tauplus_y, p_pi1_tauplus_z])
            p_pi2_tauplus = numpy.array([p_pi2_tauplus_x, p_pi2_tauplus_y, p_pi2_tauplus_z])
            p_pi3_tauplus = numpy.array([p_pi3_tauplus_x, p_pi3_tauplus_y, p_pi3_tauplus_z])
            if verbose:
                print('pi1_tau+ momentum:', p_pi1_tauplus)
                print('pi2_tau+ momentum:', p_pi2_tauplus)
                print('pi3_tau+ momentum:', p_pi3_tauplus)

            p_pi1_tauminus = numpy.array([p_pi1_tauminus_x, p_pi1_tauminus_y, p_pi1_tauminus_z])
            p_pi2_tauminus = numpy.array([p_pi2_tauminus_x, p_pi2_tauminus_y, p_pi2_tauminus_z])
            p_pi3_tauminus = numpy.array([p_pi3_tauminus_x, p_pi3_tauminus_y, p_pi3_tauminus_z])
            if verbose:
                print('pi1_tau- momentum:', p_pi1_tauminus)
                print('pi2_tau- momentum:', p_pi2_tauminus)
                print('pi3_tau- momentum:', p_pi3_tauminus)

            p_pi_K = numpy.array([p_pi_K_x, p_pi_K_y, p_pi_K_z])
            p_K = numpy.array([p_K_x, p_K_y, p_K_z])
            if verbose:
                print('pi_k momentum:', p_pi_K)
                print('k momentum:', p_K)

            # here comes just the implementation of kinematic equation
            e_tauplus = numpy.array([tv_tauplus_x - sv_x, tv_tauplus_y - sv_y, tv_tauplus_z - sv_z]) / numpy.linalg.norm(numpy.array([tv_tauplus_x - sv_x, tv_tauplus_y - sv_y, tv_tauplus_z - sv_z]))
            e_tauminus = numpy.array([tv_tauminus_x - sv_x, tv_tauminus_y - sv_y, tv_tauminus_z - sv_z]) / numpy.linalg.norm(numpy.array([tv_tauminus_x - sv_x, tv_tauminus_y - sv_y, tv_tauminus_z - sv_z]))
            e_B = numpy.array([sv_x - pv_x, sv_y - pv_y, sv_z - pv_z]) / numpy.linalg.norm(numpy.array([sv_x - pv_x, sv_y - pv_y, sv_z - pv_z]))
            if verbose:
                print('e_tau+:', e_tauplus)
                print('e_tau-:', e_tauminus)
                print('e_B:', e_B)

            p_pis_tauplus = p_pi1_tauplus + p_pi2_tauplus + p_pi3_tauplus
            if verbose: print('p_pis_tau+:', p_pis_tauplus)

            p_pis_tauplus_par = numpy.dot(p_pis_tauplus, e_tauplus)
            if verbose: print('p_pis_tau+_par:', p_pis_tauplus_par)

            p_pis_tauplus_perp_sqr = numpy.linalg.norm(p_pis_tauplus) ** 2 - p_pis_tauplus_par ** 2
            if verbose: print('p_pis_tau+_perp^2:', p_pis_tauplus_perp_sqr)

            E_pis_tauplus = numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi1_tauplus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi2_tauplus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi3_tauplus) ** 2)
            if verbose: print('E_pis_tau+: {:.12f}'.format(E_pis_tauplus))

            C_tauplus_sqr = (m_tau ** 2 - E_pis_tauplus ** 2 - p_pis_tauplus_perp_sqr + p_pis_tauplus_par ** 2) / 2
            if verbose: print('C_tau+^2: {:.12f}'.format(C_tauplus_sqr))

            alpha_tauplus = C_tauplus_sqr * E_pis_tauplus / (E_pis_tauplus ** 2 - p_pis_tauplus_par ** 2)
            if verbose: print('alpha_tau+: {:.12f}'.format(alpha_tauplus))

            # checking if the expression under the square root is not negative
            if (p_pis_tauplus_perp_sqr * p_pis_tauplus_par ** 2 + C_tauplus_sqr ** 2 - E_pis_tauplus ** 2 * p_pis_tauplus_perp_sqr) >= 0:
                beta_tauplus = p_pis_tauplus_par * numpy.sqrt(p_pis_tauplus_perp_sqr * p_pis_tauplus_par ** 2 + C_tauplus_sqr ** 2 - E_pis_tauplus ** 2 * p_pis_tauplus_perp_sqr) / (E_pis_tauplus ** 2 - p_pis_tauplus_par ** 2)
                if verbose: print('beta_tau+: {:.12f}'.format(beta_tauplus))

                p_nu_tauplus_1 = alpha_tauplus + beta_tauplus
                p_nu_tauplus_2 = alpha_tauplus - beta_tauplus

                p_tauplus_1 = numpy.sqrt(E_pis_tauplus ** 2 + p_nu_tauplus_1 ** 2 + 2 * E_pis_tauplus * p_nu_tauplus_1 - m_tau ** 2)
                if verbose: print('p_tau+_1: {:.12f}'.format(p_tauplus_1))
                p_tauplus_2 = numpy.sqrt(E_pis_tauplus ** 2 + p_nu_tauplus_2 ** 2 + 2 * E_pis_tauplus * p_nu_tauplus_2 - m_tau ** 2)
                if verbose: print('p_tau+_2: {:.12f}'.format(p_tauplus_2))

                p_pis_tauminus = p_pi1_tauminus + p_pi2_tauminus + p_pi3_tauminus
                if verbose: print('p_pis_tau-:', p_pis_tauminus)

                p_pis_tauminus_par = numpy.dot(p_pis_tauminus, e_tauminus)
                if verbose: print('p_pis_tau-_par:', p_pis_tauminus_par)

                p_pis_tauminus_perp_sqr = numpy.linalg.norm(p_pis_tauminus) ** 2 - p_pis_tauminus_par ** 2
                if verbose: print('p_pis_tau-_perp^2:', p_pis_tauminus_perp_sqr)

                E_pis_tauminus = numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi1_tauminus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi2_tauminus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi3_tauminus) ** 2)
                if verbose: print('E_pis_tau-: {:.12f}'.format(E_pis_tauminus))

                C_tauminus_sqr = (m_tau ** 2 - E_pis_tauminus ** 2 - p_pis_tauminus_perp_sqr + p_pis_tauminus_par ** 2) / 2
                if verbose: print 'C_tau-^2: {:.12f}'.format(C_tauminus_sqr)

                alpha_tauminus = C_tauminus_sqr * E_pis_tauminus / (E_pis_tauminus ** 2 - p_pis_tauminus_par ** 2)
                if verbose: print('alpha_tau-: {:.12f}'.format(alpha_tauminus))

                # checking if the expression under the square root is not negative
                if (p_pis_tauminus_perp_sqr * p_pis_tauminus_par ** 2 + C_tauminus_sqr ** 2 - E_pis_tauminus ** 2 * p_pis_tauminus_perp_sqr) >= 0:
                    beta_tauminus = p_pis_tauminus_par * numpy.sqrt(p_pis_tauminus_perp_sqr * p_pis_tauminus_par ** 2 + C_tauminus_sqr ** 2 - E_pis_tauminus ** 2 * p_pis_tauminus_perp_sqr) / (E_pis_tauminus ** 2 - p_pis_tauminus_par ** 2)
                    reconstructable_events += 1
                    if verbose: print('beta_tau-: {:.12f}'.format(beta_tauminus))

                    p_nu_tauminus_1 = alpha_tauminus + beta_tauminus
                    p_nu_tauminus_2 = alpha_tauminus - beta_tauminus

                    p_tauminus_1 = numpy.sqrt(E_pis_tauminus ** 2 + p_nu_tauminus_1 ** 2 + 2 * E_pis_tauminus * p_nu_tauminus_1 - m_tau ** 2)
                    if verbose: print('p_tauminus_1: {:.12f}'.format(p_tauminus_1))
                    p_tauminus_2 = numpy.sqrt(E_pis_tauminus ** 2 + p_nu_tauminus_2 ** 2 + 2 * E_pis_tauminus * p_nu_tauminus_2 - m_tau ** 2)
                    if verbose: print('p_tauminus_2: {:.12f}'.format(p_tauminus_2))

                    A = - (numpy.dot(e_tauplus, e_tauminus) - numpy.dot(e_B, e_tauplus) * numpy.dot(e_B, e_tauminus)) / (1 - numpy.dot(e_B, e_tauminus) ** 2)

                    p_piK_perp = p_pi_K + p_K - numpy.dot((p_pi_K + p_K), e_B) * e_B
                    p_piK_par = numpy.dot((p_pi_K + p_K), e_B)

                    B = - numpy.dot(p_piK_perp, (e_tauminus + numpy.dot(e_tauminus, e_B) * e_B)) / (1 - numpy.dot(e_B, e_tauminus) ** 2)

                    p_tauminus_1_alt = A * p_tauplus_1 + B
                    p_tauminus_2_alt = A * p_tauplus_2 + B

                    # resolving ambiguity
                    min_diff = min(abs(p_tauminus_1 - p_tauminus_1_alt), abs(p_tauminus_1 - p_tauminus_2_alt), abs(p_tauminus_2 - p_tauminus_1_alt), abs(p_tauminus_2 - p_tauminus_2_alt))
                    if min_diff == abs(p_tauminus_1 - p_tauminus_1_alt):
                        p_tauplus = p_tauplus_1
                        p_tauminus = p_tauminus_1
                    elif min_diff == abs(p_tauminus_1 - p_tauminus_2_alt):
                        p_tauplus = p_tauplus_2
                        p_tauminus = p_tauminus_1
                    elif min_diff == abs(p_tauminus_2 - p_tauminus_1_alt):
                        p_tauplus = p_tauplus_1
                        p_tauminus = p_tauminus_2
                    elif min_diff == abs(p_tauminus_2 - p_tauminus_2_alt):
                        p_tauplus = p_tauplus_2
                        p_tauminus = p_tauminus_2

                    p_B = p_tauplus * numpy.dot(e_tauplus, e_B) + p_tauminus * numpy.dot(e_tauminus, e_B) + p_piK_par
                    if verbose: print('B momentum: {:.12f}'.format(p_B))

                    E_piK = numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi_K) ** 2) + numpy.sqrt(m_K ** 2 + numpy.linalg.norm(p_K) ** 2)
                    E_tauplus = numpy.sqrt(m_tau ** 2 + p_tauplus ** 2)
                    E_tauminus = numpy.sqrt(m_tau ** 2 + p_tauminus ** 2)

                    m_B = numpy.sqrt(E_tauplus ** 2 + E_tauminus ** 2 + E_piK ** 2 + 2 * (E_tauplus * E_tauminus + E_tauplus * E_piK + E_tauminus * E_piK) - p_B ** 2)
                    if verbose: print('B mass: {:.12f}'.format(m_B))

                    b_mass.setVal(m_B)
                    data.add(RooArgSet(b_mass))

    # creating canvas the plots to be drawn in
    canvas_m_B = TCanvas('mB_canvas', 'Reconstructed B0 mass distribution', 640, 640 if fit else 480)
    canvas_m_B.cd()

    # creating the pad for the reconstructed B mass distribution histogram
    upper_pad = TPad('upper_pad', 'Upper Pad', 0., 0.25 if  fit else 0., 1., 1.)
    upper_pad.Draw()

    # adding label "FCC-ee"
    label = TPaveText(0.75, 0.8, .9, .9, 'NDC')
    label.AddText('FCC-#it{ee}')

    plot_frame = b_mass.frame(RooFit.Name('B mass'), RooFit.Title('Reconstructed B^{0}_{d} mass'), RooFit.Bins(n_bins))
    plot_frame.GetXaxis().SetTitle('m_{B_{d}^{0}}, GeV/#it{c}^{2}')
    plot_frame.GetYaxis().SetTitle('Events / ({} GeV/#it{{c}}^{{2}})'.format(float(x_max - x_min) / n_bins))
    data.plotOn(plot_frame)

    if fit:
        if background:
            # background model is Gaussian + Crystal Ball function

            # defining parameters
            mean = RooRealVar('mean', '#mu', 5.279, peak_x_min, peak_x_max)
            width_right_cb = RooRealVar('width_right_cb', '#sigma_{right CB}', 0.04, 0.01, 1.)
            width_gauss = RooRealVar('width_gauss', '#sigma_{Gauss}', 0.03, 0.01, 1.)
            alpha_right_cb = RooRealVar('alpha_right_cb', '#alpha_{right CB}', -1, -10., -0.1)
            n_right_cb = RooRealVar('n_right_cb', 'n_{right CB}', 2., 0.1, 10.)

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

        model.fitTo(data)
        if background:
            model.plotOn(plot_frame, RooFit.Components('cb_right'), RooFit.LineColor(ROOT.kRed), RooFit.LineStyle(ROOT.kDashed))
            model.plotOn(plot_frame, RooFit.Components('gauss'), RooFit.LineColor(ROOT.kGreen), RooFit.LineStyle(ROOT.kDashed))
        else:
            model.plotOn(plot_frame, RooFit.Components('narrow_gauss'), RooFit.LineColor(ROOT.kRed), RooFit.LineStyle(ROOT.kDashed))
            model.plotOn(plot_frame, RooFit.Components('cb'), RooFit.LineColor(ROOT.kGreen), RooFit.LineStyle(ROOT.kDashed))
            model.plotOn(plot_frame, RooFit.Components('wide_gauss'), RooFit.LineColor(ROOT.kMagenta), RooFit.LineStyle(ROOT.kDashed))

        model.plotOn(plot_frame)
        params = model.getVariables()
        params.Print('v')

        # prepairing pulls histogram
        pulls_hist = plot_frame.pullHist()
        pulls_hist.GetXaxis().SetTitle('')
        pulls_hist.GetYaxis().SetTitle('')
        pulls_hist.GetXaxis().SetRangeUser(x_min, x_max)
        pulls_hist.GetYaxis().SetRangeUser(-5., 5.)

        upper_line = TLine(x_min, 2, x_max, 2)
        upper_line.SetLineColor(ROOT.kRed)
        lower_line = TLine(x_min, -2, x_max, -2)
        lower_line.SetLineColor(ROOT.kRed)

        # creating the pad the pulls histogram to be drawn in
        lower_pad = TPad('lower_pad', 'Lower Pad', 0., 0., 1., 0.25)
        lower_pad.Draw()
        lower_pad.cd()

        # drawing pulls histogram
        pulls_hist.Draw('ap')
        upper_line.Draw()
        lower_line.Draw()

    upper_pad.cd()
    plot_frame.Draw()
    label.Draw()

    canvas_m_B.Update()

    # printing some useful statistics
    print('{} events have been processed'.format(processed_events))
    print('Elapsed time: {:.1f} s ({:.1f} events / s)'.format(time.time() - start_time, float(processed_events) / (time.time() - start_time)))
    print('Reconstruction efficiency: {} / {} = {:.3f}'.format(reconstructable_events, processed_events, float(reconstructable_events) / processed_events))

    raw_input('Press ENTER when finished')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required = True, help = 'name of the file to process')
    parser.add_argument('-t', '--tree', type = str, default = 'Events', help = 'name of the tree to process')
    parser.add_argument('-n', '--nevents', type = int, help = 'maximum number of events to process')
    parser.add_argument('-f', '--fit', action = 'store_true', help = 'fit the histogram')
    parser.add_argument('-b', '--background', action = 'store_true', help = 'use fit model for background events')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'run with increased verbosity')

    args = parser.parse_args()
    max_events = args.nevents if args.nevents else None

    process(args.input_file, args.tree, max_events, NBINS, XMIN, XMAX, args.fit, args.background, PEAK_MIN, PEAK_MAX, args.verbose)

if __name__ == '__main__':
    main(sys.argv)
