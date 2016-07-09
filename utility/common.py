#!/usr/bin/env python

"""
    Contains a set of utility functions used in reconstruction process as well as some particle masses

    m_pi - the mass of a pi meson
    m_K - the mass of a K meson
    m_tau - the mass of a tau lepton
    reconstruct - a function that reconstructs the event
    show_plot - a function that visualizes reconstruction results by making plots
"""

import os
import numpy
import ROOT

from ROOT import gROOT, gStyle, TCanvas, TPaveText, TPad, TLine, TLegend
from ROOT import RooFit, RooRealVar, RooArgList, RooGaussian, RooAddPdf

from UnreconstructableEventError import UnreconstructableEventError
from ReconstructedEvent import ReconstructedEvent
from heppy_fcc.utility.Momentum import Momentum

# Masses of the particles
m_pi = 0.13957018
m_K = 0.493677
m_tau = 1.77684

# Nice looking plots
gROOT.ProcessLine('.x ' + os.environ.get('FCC') + 'lhcbstyle.C')
gStyle.SetOptStat(0)

def isclose(a, b, rel_tol = 1e-09, abs_tol = 0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def reconstruct(event, mc_truth_event, verbose = False):
    """
        A function that implements the reconstruction algorithm for a given event

        Args:
        event (ROOT.TTree): the event to reconstruct
        verbose (optional, [bool]): the flag that determines whether the function will be run with increased verbosity. Defaults to False

        Returns:
        ReconstructedEvent: reconstructed event information

        Raises:
        UnreconstructableEventError: if the event cannot be reconstructed because of poor smeared values
    """

    rec_ev = ReconstructedEvent()

    # Setting numpy precision
    if verbose:
        numpy.set_printoptions(12)

    pv = numpy.array([event.pv_x, event.pv_y, event.pv_z])
    sv = numpy.array([event.sv_x, event.sv_y, event.sv_z])
    tv_tauplus = numpy.array([event.tv_tauplus_x, event.tv_tauplus_y, event.tv_tauplus_z])
    tv_tauminus = numpy.array([event.tv_tauminus_x, event.tv_tauminus_y, event.tv_tauminus_z])
    if verbose:
        print('Primary vertex: {}'.format(pv))
        print('Secondary vertex: {}'.format(sv))
        print('Tertiary vertex (tau+): {}'.format(tv_tauplus))
        print('Tertiary vertex (tau-): {}'.format(tv_tauminus))

    p_pi1_tauplus = numpy.array([event.pi1_tauplus_px, event.pi1_tauplus_py, event.pi1_tauplus_pz])
    p_pi2_tauplus = numpy.array([event.pi2_tauplus_px, event.pi2_tauplus_py, event.pi2_tauplus_pz])
    p_pi3_tauplus = numpy.array([event.pi3_tauplus_px, event.pi3_tauplus_py, event.pi3_tauplus_pz])
    if verbose:
        print('pi1_tau+ momentum: {}'.format(p_pi1_tauplus))
        print('pi2_tau+ momentum: {}'.format(p_pi2_tauplus))
        print('pi3_tau+ momentum: {}'.format(p_pi3_tauplus))

    p_pi1_tauminus = numpy.array([event.pi1_tauminus_px, event.pi1_tauminus_py, event.pi1_tauminus_pz])
    p_pi2_tauminus = numpy.array([event.pi2_tauminus_px, event.pi2_tauminus_py, event.pi2_tauminus_pz])
    p_pi3_tauminus = numpy.array([event.pi3_tauminus_px, event.pi3_tauminus_py, event.pi3_tauminus_pz])
    if verbose:
        print('pi1_tau- momentum: {}'.format(p_pi1_tauminus))
        print('pi2_tau- momentum: {}'.format(p_pi2_tauminus))
        print('pi3_tau- momentum: {}'.format(p_pi3_tauminus))

    p_pi_K = numpy.array([event.pi_kstar_px, event.pi_kstar_py, event.pi_kstar_pz])
    p_K = numpy.array([event.k_px, event.k_py, event.k_pz])
    if verbose:
        print('pi_kstar momentum: {}'.format(p_pi_K))
        print('k momentum: {}'.format(p_K))

    # here comes just the implementation of kinematic equation
    e_tauplus = (tv_tauplus - sv) / numpy.linalg.norm(tv_tauplus - sv)
    e_tauminus = (tv_tauminus - sv) / numpy.linalg.norm(tv_tauminus - sv)
    e_B = (sv - pv) / numpy.linalg.norm(sv - pv)
    if verbose:
        print('e_tau+: {}'.format(e_tauplus))
        print('e_tau-: {}'.format(e_tauminus))
        print('e_B: {}'.format(e_B))

    p_pis_tauplus = p_pi1_tauplus + p_pi2_tauplus + p_pi3_tauplus
    if verbose: print('p_pis_tau+: {}'.format(p_pis_tauplus))

    p_pis_tauplus_par = numpy.dot(p_pis_tauplus, e_tauplus)
    if verbose: print('p_pis_tau+_par: {}'.format(p_pis_tauplus_par))

    p_pis_tauplus_perp_sqr = numpy.linalg.norm(p_pis_tauplus) ** 2 - p_pis_tauplus_par ** 2
    if verbose: print('p_pis_tau+_perp^2: {}'.format(p_pis_tauplus_perp_sqr))

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
        p_tauplus_2 = numpy.sqrt(E_pis_tauplus ** 2 + p_nu_tauplus_2 ** 2 + 2 * E_pis_tauplus * p_nu_tauplus_2 - m_tau ** 2)
        if verbose:
            print('p_tau+_1: {:.12f}'.format(p_tauplus_1))
            print('p_tau+_2: {:.12f}'.format(p_tauplus_2))

        p_pis_tauminus = p_pi1_tauminus + p_pi2_tauminus + p_pi3_tauminus
        if verbose: print('p_pis_tau-: {}'.format(p_pis_tauminus))

        p_pis_tauminus_par = numpy.dot(p_pis_tauminus, e_tauminus)
        if verbose: print('p_pis_tau-_par: {}'.format(p_pis_tauminus_par))

        p_pis_tauminus_perp_sqr = numpy.linalg.norm(p_pis_tauminus) ** 2 - p_pis_tauminus_par ** 2
        if verbose: print('p_pis_tau-_perp^2: {}'.format(p_pis_tauminus_perp_sqr))

        E_pis_tauminus = numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi1_tauminus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi2_tauminus) ** 2) + numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi3_tauminus) ** 2)
        if verbose: print('E_pis_tau-: {:.12f}'.format(E_pis_tauminus))

        C_tauminus_sqr = (m_tau ** 2 - E_pis_tauminus ** 2 - p_pis_tauminus_perp_sqr + p_pis_tauminus_par ** 2) / 2
        if verbose: print 'C_tau-^2: {:.12f}'.format(C_tauminus_sqr)

        alpha_tauminus = C_tauminus_sqr * E_pis_tauminus / (E_pis_tauminus ** 2 - p_pis_tauminus_par ** 2)
        if verbose: print('alpha_tau-: {:.12f}'.format(alpha_tauminus))

        # checking if the expression under the square root is not negative
        if (p_pis_tauminus_perp_sqr * p_pis_tauminus_par ** 2 + C_tauminus_sqr ** 2 - E_pis_tauminus ** 2 * p_pis_tauminus_perp_sqr) >= 0:
            beta_tauminus = p_pis_tauminus_par * numpy.sqrt(p_pis_tauminus_perp_sqr * p_pis_tauminus_par ** 2 + C_tauminus_sqr ** 2 - E_pis_tauminus ** 2 * p_pis_tauminus_perp_sqr) / (E_pis_tauminus ** 2 - p_pis_tauminus_par ** 2)
            if verbose: print('beta_tau-: {:.12f}'.format(beta_tauminus))

            p_nu_tauminus_1 = alpha_tauminus + beta_tauminus
            p_nu_tauminus_2 = alpha_tauminus - beta_tauminus

            p_tauminus_1 = numpy.sqrt(E_pis_tauminus ** 2 + p_nu_tauminus_1 ** 2 + 2 * E_pis_tauminus * p_nu_tauminus_1 - m_tau ** 2)
            p_tauminus_2 = numpy.sqrt(E_pis_tauminus ** 2 + p_nu_tauminus_2 ** 2 + 2 * E_pis_tauminus * p_nu_tauminus_2 - m_tau ** 2)
            if verbose:
                print('p_tauminus_1: {:.12f}'.format(p_tauminus_1))
                print('p_tauminus_2: {:.12f}'.format(p_tauminus_2))

            A = - (numpy.dot(e_tauplus, e_tauminus) - numpy.dot(e_B, e_tauplus) * numpy.dot(e_B, e_tauminus)) / (1 - numpy.dot(e_B, e_tauminus) ** 2)

            p_piK_perp = p_pi_K + p_K - numpy.dot((p_pi_K + p_K), e_B) * e_B
            p_piK_par = numpy.dot((p_pi_K + p_K), e_B)

            B = - numpy.dot(p_piK_perp, (e_tauminus + numpy.dot(e_tauminus, e_B) * e_B)) / (1 - numpy.dot(e_B, e_tauminus) ** 2)

            p_tauminus_1_alt = A * p_tauplus_1 + B
            p_tauminus_2_alt = A * p_tauplus_2 + B
            if verbose:
                print('p_tauminus_1_alt: {:.12f}'.format(p_tauminus_1_alt))
                print('p_tauminus_2_alt: {:.12f}'.format(p_tauminus_2_alt))


            # resolving ambiguity
            if isclose(abs(p_tauplus_1 - numpy.sqrt(mc_truth_event.tauplus_px ** 2 + mc_truth_event.tauplus_py ** 2 + mc_truth_event.tauplus_pz ** 2)), min(abs(p_tauplus_1 - numpy.sqrt(mc_truth_event.tauplus_px ** 2 + mc_truth_event.tauplus_py ** 2 + mc_truth_event.tauplus_pz ** 2)), abs(p_tauplus_2 - numpy.sqrt(mc_truth_event.tauplus_px ** 2 + mc_truth_event.tauplus_py ** 2 + mc_truth_event.tauplus_pz ** 2)))):
                p_tauplus = p_tauplus_1
            else:
                p_tauplus = p_tauplus_2

            if isclose(abs(p_tauminus_1 - numpy.sqrt(mc_truth_event.tauminus_px ** 2 + mc_truth_event.tauminus_py ** 2 + mc_truth_event.tauminus_pz ** 2)), min(abs(p_tauminus_1 - numpy.sqrt(mc_truth_event.tauminus_px ** 2 + mc_truth_event.tauminus_py ** 2 + mc_truth_event.tauminus_pz ** 2)), abs(p_tauminus_2 - numpy.sqrt(mc_truth_event.tauminus_px ** 2 + mc_truth_event.tauminus_py ** 2 + mc_truth_event.tauminus_pz ** 2)))):
                p_tauminus = p_tauminus_1
            else:
                p_tauminus = p_tauminus_2

            if verbose:
                print('p_tauplus: {:.12f}'.format(p_tauplus))
                print('p_tauminus: {:.12f}'.format(p_tauminus))

            p_B = p_tauplus * numpy.dot(e_tauplus, e_B) + p_tauminus * numpy.dot(e_tauminus, e_B) + p_piK_par
            p_B_11 = p_tauplus_1 * numpy.dot(e_tauplus, e_B) + p_tauminus_1 * numpy.dot(e_tauminus, e_B) + p_piK_par
            p_B_12 = p_tauplus_1 * numpy.dot(e_tauplus, e_B) + p_tauminus_2 * numpy.dot(e_tauminus, e_B) + p_piK_par
            p_B_21 = p_tauplus_2 * numpy.dot(e_tauplus, e_B) + p_tauminus_1 * numpy.dot(e_tauminus, e_B) + p_piK_par
            p_B_22 = p_tauplus_2 * numpy.dot(e_tauplus, e_B) + p_tauminus_2 * numpy.dot(e_tauminus, e_B) + p_piK_par
            rec_ev.p_b = Momentum.fromlist(p_B * e_B)
            rec_ev.p_b_11 = Momentum.fromlist(p_B_11 * e_B)
            rec_ev.p_b_12 = Momentum.fromlist(p_B_12 * e_B)
            rec_ev.p_b_21 = Momentum.fromlist(p_B_21 * e_B)
            rec_ev.p_b_22 = Momentum.fromlist(p_B_22 * e_B)
            if verbose: print('B momentum: {:.12f}'.format(p_B))

            rec_ev.p_tauplus = Momentum.fromlist(p_tauplus * e_tauplus)
            rec_ev.p_tauplus_1 = Momentum.fromlist(p_tauplus_1 * e_tauplus)
            rec_ev.p_tauplus_2 = Momentum.fromlist(p_tauplus_2 * e_tauplus)
            rec_ev.p_nu_tauplus = Momentum.fromlist(p_tauplus * e_tauplus - p_pis_tauplus)
            rec_ev.p_nu_tauplus_1 = Momentum.fromlist(p_tauplus_1 * e_tauplus - p_pis_tauplus)
            rec_ev.p_nu_tauplus_2 = Momentum.fromlist(p_tauplus_2 * e_tauplus - p_pis_tauplus)
            rec_ev.p_tauminus = Momentum.fromlist(p_tauminus * e_tauminus)
            rec_ev.p_tauminus_1 = Momentum.fromlist(p_tauminus_1 * e_tauminus)
            rec_ev.p_tauminus_2 = Momentum.fromlist(p_tauminus_2 * e_tauminus)
            rec_ev.p_nu_tauminus = Momentum.fromlist(p_tauminus * e_tauminus - p_pis_tauminus)
            rec_ev.p_nu_tauminus_1 = Momentum.fromlist(p_tauminus_1 * e_tauminus - p_pis_tauminus)
            rec_ev.p_nu_tauminus_2 = Momentum.fromlist(p_tauminus_2 * e_tauminus - p_pis_tauminus)

            E_piK = numpy.sqrt(m_pi ** 2 + numpy.linalg.norm(p_pi_K) ** 2) + numpy.sqrt(m_K ** 2 + numpy.linalg.norm(p_K) ** 2)
            E_tauplus = numpy.sqrt(m_tau ** 2 + p_tauplus ** 2)
            E_tauplus_1 = numpy.sqrt(m_tau ** 2 + p_tauplus_1 ** 2)
            E_tauplus_2 = numpy.sqrt(m_tau ** 2 + p_tauplus_2 ** 2)
            E_tauminus = numpy.sqrt(m_tau ** 2 + p_tauminus ** 2)
            E_tauminus_1 = numpy.sqrt(m_tau ** 2 + p_tauminus_1 ** 2)
            E_tauminus_2 = numpy.sqrt(m_tau ** 2 + p_tauminus_2 ** 2)

            m_B = numpy.sqrt(E_tauplus ** 2 + E_tauminus ** 2 + E_piK ** 2 + 2 * (E_tauplus * E_tauminus + E_tauplus * E_piK + E_tauminus * E_piK) - p_B ** 2)
            rec_ev.m_b = m_B
            m_B_11 = numpy.sqrt(E_tauplus_1 ** 2 + E_tauminus_1 ** 2 + E_piK ** 2 + 2 * (E_tauplus_1 * E_tauminus_1 + E_tauplus_1 * E_piK + E_tauminus_1 * E_piK) - p_B_11 ** 2)
            rec_ev.m_b_11 = m_B_11
            m_B_12 = numpy.sqrt(E_tauplus_1 ** 2 + E_tauminus_2 ** 2 + E_piK ** 2 + 2 * (E_tauplus_1 * E_tauminus_2 + E_tauplus_1 * E_piK + E_tauminus_2 * E_piK) - p_B_12 ** 2)
            rec_ev.m_b_12 = m_B_12
            m_B_21 = numpy.sqrt(E_tauplus_2 ** 2 + E_tauminus_1 ** 2 + E_piK ** 2 + 2 * (E_tauplus_2 * E_tauminus_1 + E_tauplus_2 * E_piK + E_tauminus_1 * E_piK) - p_B_21 ** 2)
            rec_ev.m_b_21 = m_B_21
            m_B_22 = numpy.sqrt(E_tauplus_2 ** 2 + E_tauminus_2 ** 2 + E_piK ** 2 + 2 * (E_tauplus_2 * E_tauminus_2 + E_tauplus_2 * E_piK + E_tauminus_2 * E_piK) - p_B_22 ** 2)
            rec_ev.m_b_22 = m_B_22
            if verbose: print('B mass: {:.12f}'.format(m_B))

            return rec_ev

        else:
            raise UnreconstructableEventError()
    else:
        raise UnreconstructableEventError()

def show_plot(var, data, units, n_bins = 100, fit = False, model = None, extended = False, components_to_plot = RooArgList(), draw_legend = False):
    """
        A function that visualizes the results of the reconstruction by showing plots

        Args:
        var (ROOT.RooRealVar): the variable the histogram of the diatribution of which will be plotted
        data (ROOT.RooDataSet): the data to be fitted
        units (str or None): X-axis units
        n_bins (optional, [int]): the number of bins in the histogram. Defaults to 100
        fit (optional, [bool]): the flag that determines if the data will be fitted. Defaults to False
        model (optional, required if fit is True, [ROOT.RooAddPdf]): the model to be used for fitting. Defaults to None
        extended (optional, [bool]): determines whether RooFit extended fit will be used. Defaults to False
        components_to_plot (optional, [ROOT.RooArgList]): the components of the model to plot. Defaults to ROOT.RooArgList()
        draw_legend (optional, [bool]): the flag that determines whether the fit legend will be drawn. Defaults to False
    """

    # creating canvas the plots to be drawn in
    canvas = TCanvas(var.GetName() + '_canvas', var.GetTitle() + ' distribution', 640, 640 if fit else 480) # creating bigger canvas if we're going to fit the data (and thus to plot pulls hist)

    # creating the pad for the reconstructed B mass distribution histogram
    upper_pad = TPad('upper_pad', 'Upper Pad', 0., 0.25 if  fit else 0., 1., 1.) # creating a pad that will occupy the top 75% of the canvas (the count starts from the bottom) if we're gooing to fit the data (and thus to plot pulls hist) and the whole canvas otherwise
    upper_pad.Draw()

    # adding label "FCC-ee"
    label = TPaveText(0.75, 0.8, .9, .9, 'NDC') # placing a label; the "NDC" option sets the units to mother container's fraction
    label.AddText('FCC-#it{ee}')

    plot_frame = var.frame(RooFit.Name(var.GetName() + '_frame'), RooFit.Title(var.GetTitle() + ' frame'), RooFit.Bins(n_bins))
    plot_frame.GetXaxis().SetTitle((var.GetTitle() + ', {}'.format(units)) if units else var.GetTitle())
    plot_frame.GetYaxis().SetTitle('Events / ({:g} {})'.format(float(var.getMax() - var.getMin()) / n_bins, units) if units else 'Events / {:g}'.format(float(var.getMax() - var.getMin()) / n_bins))
    data.plotOn(plot_frame)

    if fit:
        model.fitTo(data, RooFit.Extended(extended))

        if draw_legend:
            legend = TLegend(0.175, 0.65, 0.4, 0.9)

        color_index = 2
        for index in xrange(0, components_to_plot.getSize()):
            component = components_to_plot[index]
            model.plotOn(plot_frame, RooFit.Components(component.GetName()), RooFit.LineColor(color_index), RooFit.LineStyle(ROOT.kDashed), RooFit.Name(component.GetName() + '_curve'))
            if draw_legend:
                legend.AddEntry(plot_frame.findObject(component.GetName() + '_curve'), component.GetTitle(), 'l')
            color_index += 2 if color_index == 3 or color_index == 9 else 1 # skip the blue color (4) used for composite model and the white color (10)

        model.plotOn(plot_frame)
        params = model.getVariables()
        params.Print('v')

        # prepairing pulls histogram
        pulls_hist = plot_frame.pullHist()
        pulls_hist.GetXaxis().SetTitle('')
        pulls_hist.GetYaxis().SetTitle('')
        pulls_hist.GetXaxis().SetRangeUser(var.getMin(), var.getMax())
        pulls_hist.GetYaxis().SetRangeUser(-5., 5.)

        # creating the pad the pulls histogram to be drawn in
        lower_pad = TPad('lower_pad', 'Lower Pad', 0., 0., 1., 0.25)
        lower_pad.Draw()
        lower_pad.cd()

        # drawing pulls histogram
        pulls_hist.Draw('ap')

        upper_line = TLine(var.getMin(), 2, var.getMax(), 2)
        upper_line.SetLineColor(ROOT.kRed)
        lower_line = TLine(var.getMin(), -2, var.getMax(), -2)
        lower_line.SetLineColor(ROOT.kRed)
        upper_line.Draw()
        lower_line.Draw()

    upper_pad.cd()
    plot_frame.Draw()

    if fit and draw_legend:
        legend.Draw()

    label.Draw()

    canvas.Update()

    raw_input('Press ENTER to continue')
