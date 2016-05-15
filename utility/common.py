#!/usr/bin/env python

"""
    Contains a set of utility functions used in reconstruction process as well as some particle masses

    m_pi - the mass of a pi meson
    m_K - the mass of a K meson
    m_tau - the mass of a tau lepton
    calculate_reconstructed_mass - a function that calculates a reconstructed mass value
    show_mass_plot - a function that visualizes reconstruction results by making plots
"""

import os
import numpy
import ROOT

from ROOT import gROOT, gStyle, TCanvas, TPaveText, TPad, TLine, TLegend
from ROOT import RooFit

from UnreconstructableEventError import UnreconstructableEventError

# Masses of the particles
m_pi = 0.13957018
m_K = 0.493677
m_tau = 1.77684

def calculate_reconstructed_mass(event, verbose = False):
    """
        A function that implements the reconstruction algorithm for a given event

        Args:
        event (ROOT.TTree): the event to calculate reconstructed mass for
        verbose (optional, [bool]): the flag that determines whether the function will be run with increased verbosity. Defaults to False

        Returns:
        float: the reconstructed mass value

        Raises:
        UnreconstructableEventError: if the event cannot be reconstructed because of poor smeared values
    """

    # Setting numpy precision
    if verbose:
        numpy.set_printoptions(12)

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
        print('pi1_tau+ momentum: {}'.format(p_pi1_tauplus))
        print('pi2_tau+ momentum: {}'.format(p_pi2_tauplus))
        print('pi3_tau+ momentum: {}'.format(p_pi3_tauplus))

    p_pi1_tauminus = numpy.array([p_pi1_tauminus_x, p_pi1_tauminus_y, p_pi1_tauminus_z])
    p_pi2_tauminus = numpy.array([p_pi2_tauminus_x, p_pi2_tauminus_y, p_pi2_tauminus_z])
    p_pi3_tauminus = numpy.array([p_pi3_tauminus_x, p_pi3_tauminus_y, p_pi3_tauminus_z])
    if verbose:
        print('pi1_tau- momentum: {}'.format(p_pi1_tauminus))
        print('pi2_tau- momentum: {}'.format(p_pi2_tauminus))
        print('pi3_tau- momentum: {}'.format(p_pi3_tauminus))

    p_pi_K = numpy.array([p_pi_K_x, p_pi_K_y, p_pi_K_z])
    p_K = numpy.array([p_K_x, p_K_y, p_K_z])
    if verbose:
        print('pi_k momentum: {}'.format(p_pi_K))
        print('k momentum: {}'.format(p_K))

    # here comes just the implementation of kinematic equation
    e_tauplus = numpy.array([tv_tauplus_x - sv_x, tv_tauplus_y - sv_y, tv_tauplus_z - sv_z]) / numpy.linalg.norm(numpy.array([tv_tauplus_x - sv_x, tv_tauplus_y - sv_y, tv_tauplus_z - sv_z]))
    e_tauminus = numpy.array([tv_tauminus_x - sv_x, tv_tauminus_y - sv_y, tv_tauminus_z - sv_z]) / numpy.linalg.norm(numpy.array([tv_tauminus_x - sv_x, tv_tauminus_y - sv_y, tv_tauminus_z - sv_z]))
    e_B = numpy.array([sv_x - pv_x, sv_y - pv_y, sv_z - pv_z]) / numpy.linalg.norm(numpy.array([sv_x - pv_x, sv_y - pv_y, sv_z - pv_z]))
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
        if verbose: print('p_tau+_1: {:.12f}'.format(p_tauplus_1))
        p_tauplus_2 = numpy.sqrt(E_pis_tauplus ** 2 + p_nu_tauplus_2 ** 2 + 2 * E_pis_tauplus * p_nu_tauplus_2 - m_tau ** 2)
        if verbose: print('p_tau+_2: {:.12f}'.format(p_tauplus_2))

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

            return m_B

        else:
            raise UnreconstructableEventError('Event #{} cannot be reconstructed'.format(int(event.event_number)))
    else:
        raise UnreconstructableEventError('Event #{} cannot be reconstructed'.format(int(event.event_number)))

def show_mass_plot(var, data, n_bins = 100, fit = False, model = None, draw_legend = False):
    """
        A function that visualizes the results of the reconstruction by showing plots

        Args:
        var (ROOT.RooRealVar): the variable the histogram of the diatribution of which will be plotted
        data (ROOT.RooDataSet): the data to be fitted
        n_bins (optional, [int]): the number of bins in the histogram. Defaults to 100
        fit (optional, [bool]): the flag that determines if the data will be fitted. Defaults to False
        model (optional, required if fit is True, [ROOT.RooAddPdf]): the model to be used for fitting. Defaults to None
        draw_legend (optional, [bool]): the flag that determines whether the fit legend will be drawn. Defaults to False
    """

    # Nice looking plots
    gROOT.ProcessLine('.x ' + os.environ.get('FCC') + 'lhcbstyle.C')
    gStyle.SetOptStat(0)

    # creating canvas the plots to be drawn in
    canvas_m_B = TCanvas('mB_canvas', 'Reconstructed B0 mass distribution', 640, 640 if fit else 480) # creating bigger canvas if we're going to fit the data (and thus to plot pulls hist)

    # creating the pad for the reconstructed B mass distribution histogram
    upper_pad = TPad('upper_pad', 'Upper Pad', 0., 0.25 if  fit else 0., 1., 1.) # creating a pad that will occupy the top 75% of the canvas (the count starts from the bottom) if we're gooing to fit the data (and thus to plot pulls hist) and the whole canvas otherwise
    upper_pad.Draw()

    # adding label "FCC-ee"
    label = TPaveText(0.75, 0.8, .9, .9, 'NDC') # placing a label; the "NDC" option sets the units to mother container's fraction
    label.AddText('FCC-#it{ee}')

    plot_frame = var.frame(RooFit.Name('B mass'), RooFit.Title('Reconstructed B^{0}_{d} mass'), RooFit.Bins(n_bins))
    plot_frame.GetXaxis().SetTitle('m_{B_{d}^{0}}, GeV/#it{c}^{2}')
    plot_frame.GetYaxis().SetTitle('Events / ({:g} GeV/#it{{c}}^{{2}})'.format(float(var.getMax() - var.getMin()) / n_bins))
    data.plotOn(plot_frame)

    if fit:
        model.fitTo(data)

        if draw_legend:
            legend = TLegend(0.175, 0.65, 0.4, 0.9)

        component_itr = model.getComponents().createIterator()
        color_index = 2

        component_itr.Next() # skip first component (that is, the composite model)
        component = component_itr.Next()
        while component:
            model.plotOn(plot_frame, RooFit.Components(component.GetName()), RooFit.LineColor(color_index), RooFit.LineStyle(ROOT.kDashed), RooFit.Name(component.GetName() + '_curve'))
            if draw_legend:
                legend.AddEntry(plot_frame.findObject(component.GetName() + '_curve'), component.GetTitle(), 'l')
            color_index += 1 if color_index != 3 else 2 # skip blue color used for composite model
            component = component_itr.Next()

        model.plotOn(plot_frame) # this makes the composite model to be drawn twice - first as the first component and now; it is sort of a dirty hack but it is necessary to make pulls work properly (pullHist draws pulls hist for the last plotted component)
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

    canvas_m_B.Update()

    raw_input('Press ENTER to close the plot window')
