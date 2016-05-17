#!/usr/bin/env python

from ROOT import RooGaussian, RooCBShape, RooRealVar, RooArgList, RooAddPdf

class BackgroundModel(object):
    """
        A class that represents a background model

        The model is a Gaussian and a Crystal Ball shape sharing mean value

        Note:
        Please do not modify it's fields

        Attributes:
        name (str): the name of the model
        title (str): the title of the model
        components (ROOT.RooArgList): a RooArgList containing components of the model
        pdf (RooAddPdf): the PDF of the model
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

        super(BackgroundModel, self).__init__()

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
        # RooRealVar(self.name + '_gauss_fraction', 'Fraction of Gaussian in ' + self.title, 0.3, 0.01, 1.)

        self.gauss = RooGaussian(self.name + '_narrow_gauss', self.title + ' Narrow Gaussian', self.x, self.mean, self.width_gauss)
        self.cb = RooCBShape(self.name + '_cb', self.title + ' Crystal Ball shape', self.x, self.mean, self.width_cb, self.alpha, self.n)

        self.components = RooArgList(self.gauss, self.cb)
        self.fractions = RooArgList(self.gauss_fraction)

        self.pdf = RooAddPdf(self.name, self.title, self.components, self.fractions)
