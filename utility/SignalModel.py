#!/usr/bin/env python

"""
    Contains the SignalModel class definition

    SignalModel - a class that represents the signal fit model
"""

from ROOT import RooGaussian, RooCBShape, RooRealVar, RooArgList, RooAddPdf

class SignalModel(object):
    """
        A class that represents a signal model

        The model is a 'narrow' Gaussian, a Crystal Ball shape and a 'wide' Gaussian. All three share the same mean value. A 'narrow' Gaussian and a Crystal Ball shape share also the width

        Note:
        Please do not modify it's fields

        Attributes:
        name (str): the name of the model
        title (str): the title of the model
        components (ROOT.RooArgList): a RooArgList containing components of the model
        pdf (RooAddPdf): the PDF of the model
    """

    def __init__(self, name, title, x, mean, width, width_wide, alpha, n, narrow_gauss_fraction, cb_fraction):
        """
            Constructor

            Args:
            name (str): the name of the PDF
            title (str): the title of the PDF
            x (ROOT.RooRealVar): the PDF variable
            mean (ROOT.RooRealVar): the mean value
            width (ROOT.RooRealVar): the width shared by the 'narrow' Gaussian and the Crystal Ball shape
            width_wide (ROOT.RooRealVar): the width of the 'wide' Gaussian
            alpha (ROOT.RooRealVar): the alpha parameter of the Crystal Ball shape
            n (ROOT.RooRealVar): the n parameter of the Crystal Ball shape
            narrow_gauss_fraction (ROOT.RooRealVar): the fraction of the 'narrow Gaussian in the model'
            cb_fraction (ROOT.RooRealVar): the fraction of the Crystal Ball shape in the model
        """

        super(SignalModel, self).__init__()

        # Storing variables is necessary because RooArgList elements dangle if they go out of scope (because the RooArgList stores pointers) including when temporaries are used
        self.name = name
        self.title = title
        self.x = x
        self.mean = mean
        self.width = width
        self.width_wide = width_wide
        self.alpha = alpha
        self.n = n
        self.narrow_gauss_fraction = narrow_gauss_fraction
        # RooRealVar(self.name + '_narrow_gauss_fraction', 'Fraction of Narrow Gaussian in ' + self.title, 0.3, 0.01, 1.)
        self.cb_fraction = cb_fraction
        # RooRealVar(self.name + '_cb_fraction', 'Fraction of Crystal Ball Shape in ' + self.title, 0.3, 0.01, 1.)

        self.narrow_gauss = RooGaussian(self.name + '_narrow_gauss', self.title + ' Narrow Gaussian', self.x, self.mean, self.width)
        self.wide_gauss = RooGaussian(self.name + '_wide_gauss', self.title + ' Wide Gaussian', self.x, self.mean, self.width_wide)
        self.cb = RooCBShape(self.name + '_cb', self.title + ' Crystal Ball shape', self.x, self.mean, self.width, self.alpha, self.n)

        self.components = RooArgList(self.narrow_gauss, self.cb, self.wide_gauss)
        self.fractions = RooArgList(self.narrow_gauss_fraction, self.cb_fraction)

        self.pdf = RooAddPdf(self.name, self.title, self.components, self.fractions)
