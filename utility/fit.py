#!/usr/bin/env python

"""
    Contains a set of utility classes used when fitting

    SignalModel - a class that represents the signal fit model
    BackgroundModel - a class that represents the background fit model
"""

from ROOT import RooGaussian, RooCBShape, RooRealVar, RooArgList, RooAddPdf

class SignalModel(RooAddPdf):
    """
        A class that represents a signal model. Derived from ROOT.RooAddPdf

        The model is a 'narrow' Gaussian, a Crystal Ball shape and a 'wide' Gaussian. All three share the same mean value. A 'narrow' Gaussian and a Crystal Ball shape share also the width

        Note:
        Please do not manually modify its attributes

        Attributes:
        name (str): the name of the model
        title (str): the title of the model
        components (ROOT.RooArgList): a RooArgList containing components of the model

        Methods:
        fix: fixes all model parameters
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
        self.cb_fraction = cb_fraction

        self.narrow_gauss = RooGaussian(self.name + '_narrow_gauss', self.title + ' Narrow Gaussian', self.x, self.mean, self.width)
        self.wide_gauss = RooGaussian(self.name + '_wide_gauss', self.title + ' Wide Gaussian', self.x, self.mean, self.width_wide)
        self.cb = RooCBShape(self.name + '_cb', self.title + ' Crystal Ball shape', self.x, self.mean, self.width, self.alpha, self.n)

        self.components = RooArgList(self.narrow_gauss, self.cb, self.wide_gauss)
        self.fractions = RooArgList(self.narrow_gauss_fraction, self.cb_fraction)

        super(SignalModel, self).__init__(self.name, self.title, self.components, self.fractions)

    def fix(self):
        """
            Fixes all model parameters
        """

        self.mean.setConstant(True)
        self.width.setConstant(True)
        self.width_wide.setConstant(True)
        self.alpha.setConstant(True)
        self.n.setConstant(True)
        self.narrow_gauss_fraction.setConstant(True)
        self.cb_fraction.setConstant(True)

class BackgroundModel(RooAddPdf):
    """
        A class that represents a background model. Derived from ROOT.RooAddPdf

        The model is a Gaussian and a Crystal Ball shape sharing mean value

        Note:
        Please do not manually modify its attributes

        Attributes:
        name (str): the name of the model
        title (str): the title of the model
        components (ROOT.RooArgList): a RooArgList containing components of the model

        Methods:
        fix: fixes all model parameters
    """

    def __init__(self, name, title, x, mean, width_gauss, width_cb, alpha, n, gauss_fraction):
        """
            Constructor

            Args:
            name (str): the name of the PDF
            title (str): the title of the PDF
            x (ROOT.RooRealVar): the PDF variable
            mean (ROOT.RooRealVar): the mean value
            width_gauss (ROOT.RooRealVar): the width of the Gaussian
            width_cb (ROOT.RooRealVar): the width of the Crystal Ball shape
            alpha (ROOT.RooRealVar): the alpha parameter of the Crystal Ball shape
            n (ROOT.RooRealVar): the n parameter of the Crystal Ball shape
            gauss_fraction (ROOT.RooRealVar): the fraction of the Gaussian in the model
        """

        # Storing variables is necessary because RooArgList elements dangle if they go out of scope (because the RooArgList stores pointers) including when temporaries are used
        self.name = name
        self.title = title
        self.x = x
        self.mean = mean
        self.width_gauss = width_gauss
        self.width_cb = width_cb
        self.alpha = alpha
        self.n = n
        self.gauss_fraction = gauss_fraction

        self.gauss = RooGaussian(self.name + '_narrow_gauss', self.title + ' Narrow Gaussian', self.x, self.mean, self.width_gauss)
        self.cb = RooCBShape(self.name + '_cb', self.title + ' Crystal Ball shape', self.x, self.mean, self.width_cb, self.alpha, self.n)

        self.components = RooArgList(self.gauss, self.cb)
        self.fractions = RooArgList(self.gauss_fraction)

        super(BackgroundModel, self).__init__(self.name, self.title, self.components, self.fractions)

    def fix(self):
        """
            Fixes all model parameters
        """

        self.mean.setConstant(True)
        self.width_gauss.setConstant(True)
        self.width_cb.setConstant(True)
        self.alpha.setConstant(True)
        self.n.setConstant(True)
        self.gauss_fraction.setConstant(True)
