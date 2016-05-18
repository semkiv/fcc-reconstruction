#!/usr/bin/env python

m_tau = m_tau = 1.77684

import numpy

class ReconstructedEvent(object):
    """A utility class (more like a C-struct) used to store reconstruction results (i.e. invisible values)"""

    def __init__(self, m_b = None, p_b = None, p_tauplus = None, p_nu_tauplus = None, p_tauminus = None, p_nu_tauminus = None):
        super(ReconstructedEvent, self).__init__()

        self.m_b = m_b
        self.p_b = p_b
        self.p_tauplus = p_tauplus
        self.p_nu_tauplus = p_nu_tauplus
        self.p_tauminus = p_tauminus
        self.p_nu_tauminus = p_nu_tauminus

    def q_square(self):
        return 2 * (m_tau ** 2 - numpy.dot(self.p_tauplus, self.p_tauminus) + numpy.sqrt((m_tau ** 2 + numpy.dot(self.p_tauplus, self.p_tauplus)) * (m_tau ** 2 + numpy.dot(self.p_tauminus, self.p_tauminus))))
