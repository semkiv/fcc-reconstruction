#!/usr/bin/env python

"""
    Defines the UnreconstructableEventError class

    UnreconstructableEventError - a class of the exceptions to be thrown in case of unreconstructable event
"""

class UnreconstructableEventError(RuntimeError):
    """Exception to be thrown if the event cannot be reconstructed because of poor smeared values"""

    def __init__(self, error_string = 'Event cannot be reconstructed'):
        super(UnreconstructableEventError, self).__init__(error_string)
