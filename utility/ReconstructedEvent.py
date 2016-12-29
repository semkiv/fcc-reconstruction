#!/usr/bin/env python

"""
    Contains the definitions of the ReconstructedEvent, AllSolutions and UnreconstructableEventError classes

    ReconstructedEvent - a class that stores reconstructed event information
    AllSolutions - a class to store all 4 reconstruction results
    UnreconstructableEventError - a class of the exceptions to be thrown in case of unreconstructable event
"""

M_TAU = 1.77684

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

    def __init__(self, m_b = None, p_b = None, p_tauplus = None, p_tauminus = None, p_nu_tauplus = None, p_nu_tauminus = None, p_tauplus_1 = None, p_tauplus_2 = None, p_tauminus_1 = None, p_tauminus_2 = None):
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
        self.p_tauplus_1 = p_tauplus_1
        self.p_tauplus_2 = p_tauplus_2
        self.p_tauminus_1 = p_tauminus_1
        self.p_tauminus_2 = p_tauminus_2

    def q_square(self):
        """
            Calculates the q^2 of the event

            Returns:
            float: q^2 value
        """

        return 2 * (M_TAU ** 2 - numpy.dot(self.p_tauplus.raw(), self.p_tauminus.raw()) + numpy.sqrt((M_TAU ** 2 + numpy.dot(self.p_tauplus.raw(), self.p_tauplus.raw())) * (M_TAU ** 2 + numpy.dot(self.p_tauminus.raw(), self.p_tauminus.raw()))))

class AllSolutions(object):
    """
        A utility class to store all 4 solutions

        Attributes:
        correct_solution (ReconstructedEvent): the correct solution
        wrong_solutions (list): 3 wrong solutions
    """

    def __init__(self, correct_solution = None, wrong_solutions = None):
        """
            Constructor

            Args:
            correct_solution (optional, [ReconstructedEvent]): the correct solution. Defaults to None
            wrong_solutions (optional, [list]): 3 wrong solutions. Defaults to None
        """

        super(AllSolutions, self).__init__()

        self.correct_solution = correct_solution
        self.wrong_solutions = wrong_solutions if wrong_solutions else []

class UnreconstructableEventError(RuntimeError):
    """Exception to be thrown if the event cannot be reconstructed because of poor smeared values"""

    def __init__(self, error_string = 'Event cannot be reconstructed'):
        """
            Constructor

            Args:
            error_string (optional, [str]): error message. Defaults to 'Event cannot be reconstructed'
        """
        super(UnreconstructableEventError, self).__init__(error_string)
