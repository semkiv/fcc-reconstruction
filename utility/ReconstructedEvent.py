#!/usr/bin/env python

"""
    Contains the ReconstructedEvent class definition

    ReconstructedEvent - a class that stores reconstructed event information
"""

m_tau = 1.77684

import numpy

class ReconstructedEvent(object):
    """
        A utility class (more like a C-struct) used to store reconstruction results (i.e. invisible values)

        Attributes:
        m_b (float): reconstructed B0 mass
        p_b (heppy_fcc.utility.Momentum): reconstructed B0 momentum
        p_tauplus (heppy_fcc.utility.Momentum): reconstructed tau+ momentum
        p_nu_tauplus (heppy_fcc.utility.Momentum): reconstructed momentum of the nu from tau+ decay
        p_tauminus (heppy_fcc.utility.Momentum): reconstructed tau- momentum
        p_nu_tauminus (heppy_fcc.utility.Momentum): reconstructed momentum of the nu from tau- decay

        Methods:
        q_square (float): calculates the q^2 of the event
    """

    def __init__(self, m_b = None, p_b = None, p_tauplus = None, p_tauminus = None, p_nu_tauplus = None, p_nu_tauminus = None
    # , tauplus_ok = False, tauminus_ok = False, tau_ok = False
    ):
        """
            Constructor

            Args:
            m_b (optional, [float]): reconstructed B0 mass. Defaults to None
            p_b (optional, [heppy_fcc.utility.Momentum]): reconstructed B0 momentum. Defaults to None
            p_tauplus (optional, [heppy_fcc.utility.Momentum]): reconstructed tau+ momentum. Defaults to None
            p_tauminus (optional, [heppy_fcc.utility.Momentum]): reconstructed tau- momentum. Defaults to None
            p_nu_tauplus (optional, [heppy_fcc.utility.Momentum]): reconstructed momentum of the nu from tau+ decay. Defaults to None
            p_nu_tauminus (optional, [heppy_fcc.utility.Momentum]): reconstructed momentum of the nu from tau- decay. Defaults to None
        """

        super(ReconstructedEvent, self).__init__()

        self.m_b = m_b
        self.p_b = p_b
        self.p_tauplus = p_tauplus
        self.p_nu_tauplus = p_nu_tauplus
        self.p_tauminus = p_tauminus
        self.p_nu_tauminus = p_nu_tauminus

        # self.m_b_11 = 0.
        # self.m_b_12 = 0.
        # self.m_b_21 = 0.
        # self.m_b_22 = 0.
        # self.p_b_11 = 0.
        # self.p_b_12 = 0.
        # self.p_b_21 = 0.
        # self.p_b_22 = 0.
        # self.p_tauplus_1 = 0.
        # self.p_tauplus_2 = 0.
        # self.p_tauminus_1 = 0.
        # self.p_tauminus_2 = 0.
        # self.p_nu_tauplus_1 = 0.
        # self.p_nu_tauplus_2 = 0.
        # self.p_nu_tauminus_1 = 0.
        # self.p_nu_tauminus_2 = 0.

        # self.tauplus_ok = tauplus_ok
        # self.tauminus_ok = tauminus_ok
        # self.tau_ok = tau_ok

    def q_square(self):
        """
            Calculates the q^2 of the event

            Returns:
            float: q^2 value
        """

        return 2 * (m_tau ** 2 - numpy.dot(self.p_tauplus.raw(), self.p_tauminus.raw()) + numpy.sqrt((m_tau ** 2 + numpy.dot(self.p_tauplus.raw(), self.p_tauplus.raw())) * (m_tau ** 2 + numpy.dot(self.p_tauminus.raw(), self.p_tauminus.raw()))))
