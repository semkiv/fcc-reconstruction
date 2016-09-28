#!/usr/bin/env python

"""
    Contains a set of utility functions used in reconstruction process as well as some particle masses

    M_PI - the mass of a pi meson
    M_K - the mass of a K meson
    M_TAU - the mass of a tau lepton
    reconstruct - a function that reconstructs the event
    show_plot - a function that visualizes reconstruction results by making plots
"""

import os
import sys
import numpy
import ROOT

from ROOT import gROOT, gStyle, TCanvas, TPaveText, TPad, TLine, TLegend

# This awkward construction serves to suppress the output at RooFit modules import
devnull = open(os.devnull, 'w')
old_stdout_fileno = os.dup(sys.stdout.fileno())
os.dup2(devnull.fileno(), 1)
from ROOT import RooFit
devnull.close()
os.dup2(old_stdout_fileno, 1)

from UnreconstructableEventError import UnreconstructableEventError
from ReconstructedEvent import ReconstructedEvent
from heppy_fcc.utility.Momentum import Momentum

# Masses of the particles
M_PI = 0.13957018
M_K = 0.493677
M_TAU = 1.77684

# Nicely looking plots
gROOT.ProcessLine('.x ' + os.environ.get('FCC') + 'lhcbstyle.C')
gStyle.SetOptStat(0)

def isclose(a, b, rel_tol = 1e-09, abs_tol = 0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def reconstruct(event, verbose):
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

    pv = numpy.array([event.pv_x, event.pv_y, event.pv_z])
    sv = numpy.array([event.sv_x, event.sv_y, event.sv_z])
    tv_tauplus = numpy.array([event.tv_tauplus_x, event.tv_tauplus_y, event.tv_tauplus_z])
    tv_tauminus = numpy.array([event.tv_tauminus_x, event.tv_tauminus_y, event.tv_tauminus_z])

    p_pi1_tauplus = numpy.array([event.pi1_tauplus_px, event.pi1_tauplus_py, event.pi1_tauplus_pz])
    p_pi2_tauplus = numpy.array([event.pi2_tauplus_px, event.pi2_tauplus_py, event.pi2_tauplus_pz])
    p_pi3_tauplus = numpy.array([event.pi3_tauplus_px, event.pi3_tauplus_py, event.pi3_tauplus_pz])

    p_pi1_tauminus = numpy.array([event.pi1_tauminus_px, event.pi1_tauminus_py, event.pi1_tauminus_pz])
    p_pi2_tauminus = numpy.array([event.pi2_tauminus_px, event.pi2_tauminus_py, event.pi2_tauminus_pz])
    p_pi3_tauminus = numpy.array([event.pi3_tauminus_px, event.pi3_tauminus_py, event.pi3_tauminus_pz])

    p_pi_K = numpy.array([event.pi_kstar_px, event.pi_kstar_py, event.pi_kstar_pz])
    p_K = numpy.array([event.k_px, event.k_py, event.k_pz])

    # here comes just the implementation of kinematic equation
    kin_e_tauplus = (tv_tauplus - sv) / numpy.linalg.norm(tv_tauplus - sv)
    kin_e_tauminus = (tv_tauminus - sv) / numpy.linalg.norm(tv_tauminus - sv)
    kin_e_B = (sv - pv) / numpy.linalg.norm(sv - pv)

    kin_p_pis_tauplus = p_pi1_tauplus + p_pi2_tauplus + p_pi3_tauplus
    kin_p_pis_tauplus_par = numpy.dot(kin_p_pis_tauplus, kin_e_tauplus)
    kin_p_pis_tauplus_perp_sqr = numpy.linalg.norm(kin_p_pis_tauplus) ** 2 - kin_p_pis_tauplus_par ** 2

    kin_E_pis_tauplus = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi1_tauplus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi2_tauplus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi3_tauplus) ** 2)

    kin_C_tauplus_sqr = (M_TAU ** 2 - kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_perp_sqr + kin_p_pis_tauplus_par ** 2) / 2

    kin_alpha_tauplus = kin_C_tauplus_sqr * kin_E_pis_tauplus / (kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_par ** 2)

    # checking if the expression under the square root is not negative
    if (kin_p_pis_tauplus_perp_sqr * kin_p_pis_tauplus_par ** 2 + kin_C_tauplus_sqr ** 2 - kin_E_pis_tauplus ** 2 * kin_p_pis_tauplus_perp_sqr) >= 0:
        kin_beta_tauplus = kin_p_pis_tauplus_par * numpy.sqrt(kin_p_pis_tauplus_perp_sqr * kin_p_pis_tauplus_par ** 2 + kin_C_tauplus_sqr ** 2 - kin_E_pis_tauplus ** 2 * kin_p_pis_tauplus_perp_sqr) / (kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_par ** 2)

        kin_p_nu_tauplus_1 = kin_alpha_tauplus + kin_beta_tauplus
        kin_p_nu_tauplus_2 = kin_alpha_tauplus - kin_beta_tauplus

        kin_p_tauplus_1 = numpy.sqrt(kin_E_pis_tauplus ** 2 + kin_p_nu_tauplus_1 ** 2 + 2 * kin_E_pis_tauplus * kin_p_nu_tauplus_1 - M_TAU ** 2)
        kin_p_tauplus_2 = numpy.sqrt(kin_E_pis_tauplus ** 2 + kin_p_nu_tauplus_2 ** 2 + 2 * kin_E_pis_tauplus * kin_p_nu_tauplus_2 - M_TAU ** 2)

        kin_p_pis_tauminus = p_pi1_tauminus + p_pi2_tauminus + p_pi3_tauminus
        kin_p_pis_tauminus_par = numpy.dot(kin_p_pis_tauminus, kin_e_tauminus)
        kin_p_pis_tauminus_perp_sqr = numpy.linalg.norm(kin_p_pis_tauminus) ** 2 - kin_p_pis_tauminus_par ** 2

        kin_E_pis_tauminus = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi1_tauminus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi2_tauminus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi3_tauminus) ** 2)

        kin_C_tauminus_sqr = (M_TAU ** 2 - kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_perp_sqr + kin_p_pis_tauminus_par ** 2) / 2

        kin_alpha_tauminus = kin_C_tauminus_sqr * kin_E_pis_tauminus / (kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_par ** 2)

        # checking if the expression under the square root is not negative
        if (kin_p_pis_tauminus_perp_sqr * kin_p_pis_tauminus_par ** 2 + kin_C_tauminus_sqr ** 2 - kin_E_pis_tauminus ** 2 * kin_p_pis_tauminus_perp_sqr) >= 0:
            kin_beta_tauminus = kin_p_pis_tauminus_par * numpy.sqrt(kin_p_pis_tauminus_perp_sqr * kin_p_pis_tauminus_par ** 2 + kin_C_tauminus_sqr ** 2 - kin_E_pis_tauminus ** 2 * kin_p_pis_tauminus_perp_sqr) / (kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_par ** 2)

            kin_p_nu_tauminus_1 = kin_alpha_tauminus + kin_beta_tauminus
            kin_p_nu_tauminus_2 = kin_alpha_tauminus - kin_beta_tauminus

            kin_p_tauminus_1 = numpy.sqrt(kin_E_pis_tauminus ** 2 + kin_p_nu_tauminus_1 ** 2 + 2 * kin_E_pis_tauminus * kin_p_nu_tauminus_1 - M_TAU ** 2)
            kin_p_tauminus_2 = numpy.sqrt(kin_E_pis_tauminus ** 2 + kin_p_nu_tauminus_2 ** 2 + 2 * kin_E_pis_tauminus * kin_p_nu_tauminus_2 - M_TAU ** 2)

            kin_A = - (numpy.dot(kin_e_tauplus, kin_e_tauminus) - numpy.dot(kin_e_B, kin_e_tauplus) * numpy.dot(kin_e_B, kin_e_tauminus)) / (1 - numpy.dot(kin_e_B, kin_e_tauminus) ** 2)

            kin_p_piK_perp = p_pi_K + p_K - numpy.dot((p_pi_K + p_K), kin_e_B) * kin_e_B
            kin_p_piK_par = numpy.dot((p_pi_K + p_K), kin_e_B)

            kin_B = - numpy.dot(kin_p_piK_perp, (kin_e_tauminus + numpy.dot(kin_e_tauminus, kin_e_B) * kin_e_B)) / (1 - numpy.dot(kin_e_B, kin_e_tauminus) ** 2)

            kin_p_tauminus_1_alt = kin_A * kin_p_tauplus_1 + kin_B
            kin_p_tauminus_2_alt = kin_A * kin_p_tauplus_2 + kin_B

            # resolving ambiguity
            min_diff = min(abs(kin_p_tauminus_1 - kin_p_tauminus_1_alt), abs(kin_p_tauminus_1 - kin_p_tauminus_2_alt), abs(kin_p_tauminus_2 - kin_p_tauminus_1_alt), abs(kin_p_tauminus_2 - kin_p_tauminus_2_alt))
            if isclose(min_diff, abs(kin_p_tauminus_1 - kin_p_tauminus_1_alt)):
                kin_p_tauplus = kin_p_tauplus_1
                kin_p_tauminus = kin_p_tauminus_1
            elif isclose(min_diff, abs(kin_p_tauminus_1 - kin_p_tauminus_2_alt)):
                kin_p_tauplus = kin_p_tauplus_2
                kin_p_tauminus = kin_p_tauminus_1
            elif isclose(min_diff, abs(kin_p_tauminus_2 - kin_p_tauminus_1_alt)):
                kin_p_tauplus = kin_p_tauplus_1
                kin_p_tauminus = kin_p_tauminus_2
            elif isclose(min_diff, abs(kin_p_tauminus_2 - kin_p_tauminus_2_alt)):
                kin_p_tauplus = kin_p_tauplus_2
                kin_p_tauminus = kin_p_tauminus_2

            kin_p_B = kin_p_tauplus * numpy.dot(kin_e_tauplus, kin_e_B) + kin_p_tauminus * numpy.dot(kin_e_tauminus, kin_e_B) + kin_p_piK_par

            kin_E_piK = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi_K) ** 2) + numpy.sqrt(M_K ** 2 + numpy.linalg.norm(p_K) ** 2)
            kin_E_tauplus = numpy.sqrt(M_TAU ** 2 + kin_p_tauplus ** 2)
            kin_E_tauminus = numpy.sqrt(M_TAU ** 2 + kin_p_tauminus ** 2)
            kin_m_B = numpy.sqrt(kin_E_tauplus ** 2 + kin_E_tauminus ** 2 + kin_E_piK ** 2 + 2 * (kin_E_tauplus * kin_E_tauminus + kin_E_tauplus * kin_E_piK + kin_E_tauminus * kin_E_piK) - kin_p_B ** 2)

            # Printing comprehensive information if needed
            if verbose > 1:
                # Setting numpy precision
                numpy.set_printoptions(12)

                print('Event under the hood:')

                print('Primary vertex: {}'.format(pv))
                print('Secondary vertex: {}'.format(sv))
                print('Tertiary vertex (tau+): {}'.format(tv_tauplus))
                print('Tertiary vertex (tau-): {}'.format(tv_tauminus))
                print('pi1_tau+ momentum: {}'.format(p_pi1_tauplus))
                print('pi2_tau+ momentum: {}'.format(p_pi2_tauplus))
                print('pi3_tau+ momentum: {}'.format(p_pi3_tauplus))
                print('pi1_tau- momentum: {}'.format(p_pi1_tauminus))
                print('pi2_tau- momentum: {}'.format(p_pi2_tauminus))
                print('pi3_tau- momentum: {}'.format(p_pi3_tauminus))
                print('pi_Kstar momentum: {}'.format(p_pi_K))
                print('K momentum: {}'.format(p_K))

                print('e_tau+: {}'.format(kin_e_tauplus))
                print('e_tau-: {}'.format(kin_e_tauminus))
                print('e_B: {}'.format(kin_e_B))
                print('p_pis_tau+: {}'.format(kin_p_pis_tauplus))
                print('p_pis_tau+_par: {:.12f}'.format(kin_p_pis_tauplus_par))
                print('p_pis_tau+_perp^2: {:.12f}'.format(kin_p_pis_tauplus_perp_sqr))
                print('E_pis_tau+: {:.12f}'.format(kin_E_pis_tauplus))
                print('C_tau+^2: {:.12f}'.format(kin_C_tauplus_sqr))
                print('alpha_tau+: {:.12f}'.format(kin_alpha_tauplus))
                print('beta_tau+: {:.12f}'.format(kin_beta_tauplus))
                print('p_nu_tau+_1: {:.12f}'.format(kin_p_nu_tauplus_1))
                print('p_nu_tau+_2: {:.12f}'.format(kin_p_nu_tauplus_2))
                print('p_tau+_1: {:.12f}'.format(kin_p_tauplus_1))
                print('p_tau+_2: {:.12f}'.format(kin_p_tauplus_2))
                print('p_pis_tau-: {}'.format(kin_p_pis_tauminus))
                print('p_pis_tau-_par: {:.12f}'.format(kin_p_pis_tauminus_par))
                print('p_pis_tau-_perp^2: {:.12f}'.format(kin_p_pis_tauminus_perp_sqr))
                print('E_pis_tau-: {:.12f}'.format(kin_E_pis_tauminus))
                print 'C_tau-^2: {:.12f}'.format(kin_C_tauminus_sqr)
                print('alpha_tau-: {:.12f}'.format(kin_alpha_tauminus))
                print('beta_tau-: {:.12f}'.format(kin_beta_tauminus))
                print('p_nu_tau-_1: {:.12f}'.format(kin_p_nu_tauminus_1))
                print('p_nu_tau-_2: {:.12f}'.format(kin_p_nu_tauminus_2))
                print('p_tau-_1: {:.12f}'.format(kin_p_tauminus_1))
                print('p_tau-_2: {:.12f}'.format(kin_p_tauminus_2))
                print('p_tau-_1_alt: {:.12f}'.format(kin_p_tauminus_1_alt))
                print('p_tau-_2_alt: {:.12f}'.format(kin_p_tauminus_2_alt))
                print('p_tau+: {:.12f}'.format(kin_p_tauplus))
                print('p_tau-: {:.12f}'.format(kin_p_tauminus))
                print('B momentum: {:.12f}'.format(kin_p_B))
                print('B mass: {:.12f}'.format(kin_m_B))

            return ReconstructedEvent(kin_m_B, Momentum.fromlist(kin_p_B * kin_e_B), Momentum.fromlist(kin_p_tauplus * kin_e_tauplus), Momentum.fromlist(kin_p_tauminus * kin_e_tauminus), Momentum.fromlist(kin_p_tauplus * kin_e_tauplus - kin_p_pis_tauplus), Momentum.fromlist(kin_p_tauminus * kin_e_tauminus - kin_p_pis_tauminus))

        else:
            raise UnreconstructableEventError("Event cannot be reconstructed because of ill-formed tau- vertex")
    else:
        raise UnreconstructableEventError("Event cannot be reconstructed because of ill-formed tau+ vertex")

def reconstruct_mc_truth(event, mc_truth_event, verbose):
    """
        A function that implements the reconstruction algorithm for a given event using MC truth information

        Args:
        event (ROOT.TTree): the event to reconstruct
        mc_truth_event (ROOT.TTree): the MC truth event
        verbose (optional, [bool]): the flag that determines whether the function will be run with increased verbosity. Defaults to False

        Returns:
        ReconstructedEvent: reconstructed event information

        Raises:
        UnreconstructableEventError: if the event cannot be reconstructed because of poor smeared values
    """

    pv = numpy.array([event.pv_x, event.pv_y, event.pv_z])
    sv = numpy.array([event.sv_x, event.sv_y, event.sv_z])
    tv_tauplus = numpy.array([event.tv_tauplus_x, event.tv_tauplus_y, event.tv_tauplus_z])
    tv_tauminus = numpy.array([event.tv_tauminus_x, event.tv_tauminus_y, event.tv_tauminus_z])

    p_pi1_tauplus = numpy.array([event.pi1_tauplus_px, event.pi1_tauplus_py, event.pi1_tauplus_pz])
    p_pi2_tauplus = numpy.array([event.pi2_tauplus_px, event.pi2_tauplus_py, event.pi2_tauplus_pz])
    p_pi3_tauplus = numpy.array([event.pi3_tauplus_px, event.pi3_tauplus_py, event.pi3_tauplus_pz])

    p_pi1_tauminus = numpy.array([event.pi1_tauminus_px, event.pi1_tauminus_py, event.pi1_tauminus_pz])
    p_pi2_tauminus = numpy.array([event.pi2_tauminus_px, event.pi2_tauminus_py, event.pi2_tauminus_pz])
    p_pi3_tauminus = numpy.array([event.pi3_tauminus_px, event.pi3_tauminus_py, event.pi3_tauminus_pz])

    p_pi_K = numpy.array([event.pi_kstar_px, event.pi_kstar_py, event.pi_kstar_pz])
    p_K = numpy.array([event.k_px, event.k_py, event.k_pz])

    # here comes just the implementation of kinematic equation
    kin_e_tauplus = (tv_tauplus - sv) / numpy.linalg.norm(tv_tauplus - sv)
    kin_e_tauminus = (tv_tauminus - sv) / numpy.linalg.norm(tv_tauminus - sv)
    kin_e_B = (sv - pv) / numpy.linalg.norm(sv - pv)

    kin_p_pis_tauplus = p_pi1_tauplus + p_pi2_tauplus + p_pi3_tauplus
    kin_p_pis_tauplus_par = numpy.dot(kin_p_pis_tauplus, kin_e_tauplus)
    kin_p_pis_tauplus_perp_sqr = numpy.linalg.norm(kin_p_pis_tauplus) ** 2 - kin_p_pis_tauplus_par ** 2

    kin_E_pis_tauplus = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi1_tauplus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi2_tauplus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi3_tauplus) ** 2)

    kin_C_tauplus_sqr = (M_TAU ** 2 - kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_perp_sqr + kin_p_pis_tauplus_par ** 2) / 2

    kin_alpha_tauplus = kin_C_tauplus_sqr * kin_E_pis_tauplus / (kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_par ** 2)

    # checking if the expression under the square root is not negative
    if (kin_p_pis_tauplus_perp_sqr * kin_p_pis_tauplus_par ** 2 + kin_C_tauplus_sqr ** 2 - kin_E_pis_tauplus ** 2 * kin_p_pis_tauplus_perp_sqr) >= 0:
        kin_beta_tauplus = kin_p_pis_tauplus_par * numpy.sqrt(kin_p_pis_tauplus_perp_sqr * kin_p_pis_tauplus_par ** 2 + kin_C_tauplus_sqr ** 2 - kin_E_pis_tauplus ** 2 * kin_p_pis_tauplus_perp_sqr) / (kin_E_pis_tauplus ** 2 - kin_p_pis_tauplus_par ** 2)

        kin_p_nu_tauplus_1 = kin_alpha_tauplus + kin_beta_tauplus
        kin_p_nu_tauplus_2 = kin_alpha_tauplus - kin_beta_tauplus

        kin_p_tauplus_1 = numpy.sqrt(kin_E_pis_tauplus ** 2 + kin_p_nu_tauplus_1 ** 2 + 2 * kin_E_pis_tauplus * kin_p_nu_tauplus_1 - M_TAU ** 2)
        kin_p_tauplus_2 = numpy.sqrt(kin_E_pis_tauplus ** 2 + kin_p_nu_tauplus_2 ** 2 + 2 * kin_E_pis_tauplus * kin_p_nu_tauplus_2 - M_TAU ** 2)

        # resolving ambiguity
        kin_p_tauplus_mc_truth = numpy.sqrt(mc_truth_event.tauplus_px ** 2 + mc_truth_event.tauplus_py ** 2 + mc_truth_event.tauplus_pz ** 2)
        diff_tauplus_1 = abs(kin_p_tauplus_1 - kin_p_tauplus_mc_truth)
        diff_tauplus_2 = abs(kin_p_tauplus_2 - kin_p_tauplus_mc_truth)
        min_diff_tauplus = min(diff_tauplus_1, diff_tauplus_2)
        kin_p_tauplus = kin_p_tauplus_1 if isclose(min_diff_tauplus, diff_tauplus_1) else kin_p_tauplus_2

        kin_p_pis_tauminus = p_pi1_tauminus + p_pi2_tauminus + p_pi3_tauminus
        kin_p_pis_tauminus_par = numpy.dot(kin_p_pis_tauminus, kin_e_tauminus)
        kin_p_pis_tauminus_perp_sqr = numpy.linalg.norm(kin_p_pis_tauminus) ** 2 - kin_p_pis_tauminus_par ** 2

        kin_E_pis_tauminus = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi1_tauminus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi2_tauminus) ** 2) + numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi3_tauminus) ** 2)

        kin_C_tauminus_sqr = (M_TAU ** 2 - kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_perp_sqr + kin_p_pis_tauminus_par ** 2) / 2

        kin_alpha_tauminus = kin_C_tauminus_sqr * kin_E_pis_tauminus / (kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_par ** 2)

        # checking if the expression under the square root is not negative
        if (kin_p_pis_tauminus_perp_sqr * kin_p_pis_tauminus_par ** 2 + kin_C_tauminus_sqr ** 2 - kin_E_pis_tauminus ** 2 * kin_p_pis_tauminus_perp_sqr) >= 0:
            kin_beta_tauminus = kin_p_pis_tauminus_par * numpy.sqrt(kin_p_pis_tauminus_perp_sqr * kin_p_pis_tauminus_par ** 2 + kin_C_tauminus_sqr ** 2 - kin_E_pis_tauminus ** 2 * kin_p_pis_tauminus_perp_sqr) / (kin_E_pis_tauminus ** 2 - kin_p_pis_tauminus_par ** 2)

            kin_p_nu_tauminus_1 = kin_alpha_tauminus + kin_beta_tauminus
            kin_p_nu_tauminus_2 = kin_alpha_tauminus - kin_beta_tauminus

            kin_p_tauminus_1 = numpy.sqrt(kin_E_pis_tauminus ** 2 + kin_p_nu_tauminus_1 ** 2 + 2 * kin_E_pis_tauminus * kin_p_nu_tauminus_1 - M_TAU ** 2)
            kin_p_tauminus_2 = numpy.sqrt(kin_E_pis_tauminus ** 2 + kin_p_nu_tauminus_2 ** 2 + 2 * kin_E_pis_tauminus * kin_p_nu_tauminus_2 - M_TAU ** 2)

            # resolving ambiguity
            kin_p_tauminus_mc_truth = numpy.sqrt(mc_truth_event.tauminus_px ** 2 + mc_truth_event.tauminus_py ** 2 + mc_truth_event.tauminus_pz ** 2)
            diff_tauminus_1 = abs(kin_p_tauminus_1 - kin_p_tauminus_mc_truth)
            diff_tauminus_2 = abs(kin_p_tauminus_2 - kin_p_tauminus_mc_truth)
            min_diff_tauminus = min(diff_tauminus_1, diff_tauminus_2)
            kin_p_tauminus = kin_p_tauminus_1 if isclose(min_diff_tauminus, diff_tauminus_1) else kin_p_tauminus_2

            kin_p_piK_perp = p_pi_K + p_K - numpy.dot((p_pi_K + p_K), kin_e_B) * kin_e_B
            kin_p_piK_par = numpy.dot((p_pi_K + p_K), kin_e_B)
            kin_p_B = kin_p_tauplus * numpy.dot(kin_e_tauplus, kin_e_B) + kin_p_tauminus * numpy.dot(kin_e_tauminus, kin_e_B) + kin_p_piK_par

            kin_E_piK = numpy.sqrt(M_PI ** 2 + numpy.linalg.norm(p_pi_K) ** 2) + numpy.sqrt(M_K ** 2 + numpy.linalg.norm(p_K) ** 2)
            kin_E_tauplus = numpy.sqrt(M_TAU ** 2 + kin_p_tauplus ** 2)
            kin_E_tauminus = numpy.sqrt(M_TAU ** 2 + kin_p_tauminus ** 2)
            kin_m_B = numpy.sqrt(kin_E_tauplus ** 2 + kin_E_tauminus ** 2 + kin_E_piK ** 2 + 2 * (kin_E_tauplus * kin_E_tauminus + kin_E_tauplus * kin_E_piK + kin_E_tauminus * kin_E_piK) - kin_p_B ** 2)

            # Printing comprehensive information if needed
            if verbose > 1:
                # Setting numpy precision
                numpy.set_printoptions(12)

                print('Event under the hood:')

                print('Primary vertex: {}'.format(pv))
                print('Secondary vertex: {}'.format(sv))
                print('Tertiary vertex (tau+): {}'.format(tv_tauplus))
                print('Tertiary vertex (tau-): {}'.format(tv_tauminus))
                print('pi1_tau+ momentum: {}'.format(p_pi1_tauplus))
                print('pi2_tau+ momentum: {}'.format(p_pi2_tauplus))
                print('pi3_tau+ momentum: {}'.format(p_pi3_tauplus))
                print('pi1_tau- momentum: {}'.format(p_pi1_tauminus))
                print('pi2_tau- momentum: {}'.format(p_pi2_tauminus))
                print('pi3_tau- momentum: {}'.format(p_pi3_tauminus))
                print('pi_Kstar momentum: {}'.format(p_pi_K))
                print('K momentum: {}'.format(p_K))

                print('e_tau+: {}'.format(kin_e_tauplus))
                print('e_tau-: {}'.format(kin_e_tauminus))
                print('e_B: {}'.format(kin_e_B))
                print('p_pis_tau+: {}'.format(kin_p_pis_tauplus))
                print('p_pis_tau+_par: {:.12f}'.format(kin_p_pis_tauplus_par))
                print('p_pis_tau+_perp^2: {:.12f}'.format(kin_p_pis_tauplus_perp_sqr))
                print('E_pis_tau+: {:.12f}'.format(kin_E_pis_tauplus))
                print('C_tau+^2: {:.12f}'.format(kin_C_tauplus_sqr))
                print('alpha_tau+: {:.12f}'.format(kin_alpha_tauplus))
                print('beta_tau+: {:.12f}'.format(kin_beta_tauplus))
                print('p_nu_tau+_1: {:.12f}'.format(kin_p_nu_tauplus_1))
                print('p_nu_tau+_2: {:.12f}'.format(kin_p_nu_tauplus_2))
                print('p_tau+_1: {:.12f}'.format(kin_p_tauplus_1))
                print('p_tau+_2: {:.12f}'.format(kin_p_tauplus_2))
                print('p_pis_tau-: {}'.format(kin_p_pis_tauminus))
                print('p_pis_tau-_par: {:.12f}'.format(kin_p_pis_tauminus_par))
                print('p_pis_tau-_perp^2: {:.12f}'.format(kin_p_pis_tauminus_perp_sqr))
                print('E_pis_tau-: {:.12f}'.format(kin_E_pis_tauminus))
                print 'C_tau-^2: {:.12f}'.format(kin_C_tauminus_sqr)
                print('alpha_tau-: {:.12f}'.format(kin_alpha_tauminus))
                print('beta_tau-: {:.12f}'.format(kin_beta_tauminus))
                print('p_nu_tau-_1: {:.12f}'.format(kin_p_nu_tauminus_1))
                print('p_nu_tau-_2: {:.12f}'.format(kin_p_nu_tauminus_2))
                print('p_tau-_1: {:.12f}'.format(kin_p_tauminus_1))
                print('p_tau-_2: {:.12f}'.format(kin_p_tauminus_2))
                print('p_tau+: {:.12f}'.format(kin_p_tauplus))
                print('p_tau-: {:.12f}'.format(kin_p_tauminus))
                print('B momentum: {:.12f}'.format(kin_p_B))
                print('B mass: {:.12f}'.format(kin_m_B))

            return ReconstructedEvent(kin_m_B, Momentum.fromlist(kin_p_B * kin_e_B), Momentum.fromlist(kin_p_tauplus * kin_e_tauplus), Momentum.fromlist(kin_p_tauminus * kin_e_tauminus), Momentum.fromlist(kin_p_tauplus * kin_e_tauplus - kin_p_pis_tauplus), Momentum.fromlist(kin_p_tauminus * kin_e_tauminus - kin_p_pis_tauminus))

        else:
            raise UnreconstructableEventError("Event cannot be reconstructed because of ill-formed tau- vertex")
    else:
        raise UnreconstructableEventError("Event cannot be reconstructed because of ill-formed tau+ vertex")

def show_plot(var, data, units, n_bins = 100, fit_model = None, components_to_plot = None, draw_legend = False):
    """
        A function that visualizes the results of the reconstruction by showing plots

        Args:
        var (ROOT.RooRealVar): the variable the histogram of the diatribution of which will be plotted
        data (ROOT.RooDataSet): the data to be fitted
        units (str or None): X-axis units
        n_bins (optional, [int]): the number of bins in the histogram. Defaults to 100
        fit_model (optional, [ROOT.RooAddPdf]): the model the data has been fitted to. Defaults to None
        components_to_plot (optional, [ROOT.RooArgList]): the components of the model to plot. Defaults to None
        draw_legend (optional, [bool]): the flag that determines whether the fit legend will be drawn. Defaults to False
    """

    # creating canvas the plots to be drawn in
    canvas = TCanvas(var.GetName() + '_canvas', var.GetTitle() + ' distribution', 640, 640 if fit_model else 480) # creating bigger canvas if we're going to plot fits (and thus to plot pulls hist)

    # creating the pad for the reconstructed B mass distribution histogram
    upper_pad = TPad('upper_pad', 'Upper Pad', 0., 0.25 if  fit_model else 0., 1., 1.) # creating a pad that will occupy the top 75% of the canvas (the count starts from the bottom) if we're gooing to plot fits the data (and thus to plot pulls hist) and the whole canvas otherwise
    upper_pad.Draw()

    # adding label "FCC-ee"
    label = TPaveText(0.75, 0.8, .9, .9, 'NDC') # placing a label; the "NDC" option sets the units to mother container's fraction
    label.AddText('FCC-#it{ee}')

    plot_frame = var.frame(RooFit.Name(var.GetName() + '_frame'), RooFit.Title(var.GetTitle() + ' frame'), RooFit.Bins(n_bins))
    plot_frame.GetXaxis().SetTitle((var.GetTitle() + ', {}'.format(units)) if units else var.GetTitle())
    plot_frame.GetYaxis().SetTitle('Events / ({:g} {})'.format(float(var.getMax() - var.getMin()) / n_bins, units) if units else 'Events / {:g}'.format(float(var.getMax() - var.getMin()) / n_bins))
    data.plotOn(plot_frame)

    if fit_model:
        if draw_legend:
            legend = TLegend(0.175, 0.65, 0.4, 0.9)

        color_index = 2
        if components_to_plot:
            for index in xrange(0, components_to_plot.getSize()):
                component = components_to_plot[index]
                fit_model.plotOn(plot_frame, RooFit.Components(component.GetName()), RooFit.LineColor(color_index), RooFit.LineStyle(ROOT.kDashed), RooFit.Name(component.GetName() + '_curve'))
                if draw_legend:
                    legend.AddEntry(plot_frame.findObject(component.GetName() + '_curve'), component.GetTitle(), 'l')
                color_index += 2 if color_index == 3 or color_index == 9 else 1 # skip the blue color (4) used for composite model and the white color (10)

        fit_model.plotOn(plot_frame)

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

    if fit_model and draw_legend:
        legend.Draw()

    label.Draw()

    canvas.Update()

    raw_input('Press ENTER to continue')
