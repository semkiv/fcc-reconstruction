#!/usr/bin/env python
#
# Andrii Semkiv
# 31.03.2016
#

import sys
from array import array

import ROOT
from ROOT import TFile
from ROOT import TTree

def merge(files_and_numbers):
    output_file = TFile('merged.root', 'recreate')
    output_tree = TTree('Events', 'Events')

    pi1_tauplus_px, pi1_tauplus_py, pi1_tauplus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi2_tauplus_px, pi2_tauplus_py, pi2_tauplus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi3_tauplus_px, pi3_tauplus_py, pi3_tauplus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi1_tauminus_px, pi1_tauminus_py, pi1_tauminus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi2_tauminus_px, pi2_tauminus_py, pi2_tauminus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi3_tauminus_px, pi3_tauminus_py, pi3_tauminus_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pi_k_px, pi_k_py, pi_k_pz = array('d', [0]), array('d', [0]), array('d', [0])
    k_px, k_py, k_pz = array('d', [0]), array('d', [0]), array('d', [0])
    pv_x, pv_y, pv_z = array('d', [0]), array('d', [0]), array('d', [0])
    sv_x, sv_y, sv_z = array('d', [0]), array('d', [0]), array('d', [0])
    tv_tauplus_x, tv_tauplus_y, tv_tauplus_z = array('d', [0]), array('d', [0]), array('d', [0])
    tv_tauminus_x, tv_tauminus_y, tv_tauminus_z = array('d', [0]), array('d', [0]), array('d', [0])

    output_tree.Branch('pi1_tauplus_px', pi1_tauplus_px, 'pi1_tauplus_px/D')
    output_tree.Branch('pi1_tauplus_py', pi1_tauplus_py, 'pi1_tauplus_py/D')
    output_tree.Branch('pi1_tauplus_pz', pi1_tauplus_pz, 'pi1_tauplus_pz/D')

    output_tree.Branch('pi2_tauplus_px', pi2_tauplus_px, 'pi2_tauplus_px/D')
    output_tree.Branch('pi2_tauplus_py', pi2_tauplus_py, 'pi2_tauplus_py/D')
    output_tree.Branch('pi2_tauplus_pz', pi2_tauplus_pz, 'pi2_tauplus_pz/D')

    output_tree.Branch('pi3_tauplus_px', pi3_tauplus_px, 'pi3_tauplus_px/D')
    output_tree.Branch('pi3_tauplus_py', pi3_tauplus_py, 'pi3_tauplus_py/D')
    output_tree.Branch('pi3_tauplus_pz', pi3_tauplus_pz, 'pi3_tauplus_pz/D')

    output_tree.Branch('pi1_tauminus_px', pi1_tauminus_px, 'pi1_tauminus_px/D')
    output_tree.Branch('pi1_tauminus_py', pi1_tauminus_py, 'pi1_tauminus_py/D')
    output_tree.Branch('pi1_tauminus_pz', pi1_tauminus_pz, 'pi1_tauminus_pz/D')

    output_tree.Branch('pi2_tauminus_px', pi2_tauminus_px, 'pi2_tauminus_px/D')
    output_tree.Branch('pi2_tauminus_py', pi2_tauminus_py, 'pi2_tauminus_py/D')
    output_tree.Branch('pi2_tauminus_pz', pi2_tauminus_pz, 'pi2_tauminus_pz/D')

    output_tree.Branch('pi3_tauminus_px', pi3_tauminus_px, 'pi3_tauminus_px/D')
    output_tree.Branch('pi3_tauminus_py', pi3_tauminus_py, 'pi3_tauminus_px/D')
    output_tree.Branch('pi3_tauminus_pz', pi3_tauminus_pz, 'pi3_tauminus_px/D')

    output_tree.Branch('pi_k_px', pi_k_px, 'pi_k_px/D')
    output_tree.Branch('pi_k_py', pi_k_py, 'pi_k_py/D')
    output_tree.Branch('pi_k_pz', pi_k_pz, 'pi_k_pz/D')

    output_tree.Branch('k_px', k_px, 'k_px/D')
    output_tree.Branch('k_py', k_py, 'k_py/D')
    output_tree.Branch('k_pz', k_pz, 'k_pz/D')

    output_tree.Branch('pv_x', pv_x, 'pv_x/D')
    output_tree.Branch('pv_y', pv_y, 'pv_y/D')
    output_tree.Branch('pv_z', pv_z, 'pv_z/D')

    output_tree.Branch('sv_x', sv_x, 'sv_x/D')
    output_tree.Branch('sv_y', sv_y, 'sv_y/D')
    output_tree.Branch('sv_z', sv_z, 'sv_z/D')

    output_tree.Branch('tv_tauplus_x', tv_tauplus_x, 'tv_tauplus_x/D')
    output_tree.Branch('tv_tauplus_y', tv_tauplus_y, 'tv_tauplus_y/D')
    output_tree.Branch('tv_tauplus_z', tv_tauplus_z, 'tv_tauplus_z/D')

    output_tree.Branch('tv_tauminus_x', tv_tauminus_x, 'tv_tauminus_x/D')
    output_tree.Branch('tv_tauminus_y', tv_tauminus_y, 'tv_tauminus_y/D')
    output_tree.Branch('tv_tauminus_z', tv_tauminus_z, 'tv_tauminus_z/D')

    for f, n in files_and_numbers.iteritems():
        print 'Processing file', f
        input_file = TFile(f, 'read')
        input_tree = input_file.Get('Events')
        for counter, event in enumerate(input_tree):
            if counter < n:
                if (counter + 1) % 100 == 0:
                    print 'Processing event', counter + 1

                # Reading data necessary for reconstruction
                pi1_tauplus_px[0] = event.pi1_tauplus_px
                pi1_tauplus_py[0] = event.pi1_tauplus_py
                pi1_tauplus_pz[0] = event.pi1_tauplus_pz

                pi2_tauplus_px[0] = event.pi2_tauplus_px
                pi2_tauplus_py[0] = event.pi2_tauplus_py
                pi2_tauplus_pz[0] = event.pi2_tauplus_pz

                pi3_tauplus_px[0] = event.pi3_tauplus_px
                pi3_tauplus_py[0] = event.pi3_tauplus_py
                pi3_tauplus_pz[0] = event.pi3_tauplus_pz

                pi1_tauminus_px[0] = event.pi1_tauminus_px
                pi1_tauminus_py[0] = event.pi1_tauminus_py
                pi1_tauminus_pz[0] = event.pi1_tauminus_pz

                pi2_tauminus_px[0] = event.pi2_tauminus_px
                pi2_tauminus_py[0] = event.pi2_tauminus_py
                pi2_tauminus_pz[0] = event.pi2_tauminus_pz

                pi3_tauminus_px[0] = event.pi3_tauminus_px
                pi3_tauminus_py[0] = event.pi3_tauminus_py
                pi3_tauminus_pz[0] = event.pi3_tauminus_pz

                pi_k_px[0] = event.pi_k_px
                pi_k_py[0] = event.pi_k_py
                pi_k_pz[0] = event.pi_k_pz

                k_px[0] = event.k_px
                k_py[0] = event.k_py
                k_pz[0] = event.k_pz

                pv_x[0] = event.pv_x
                pv_y[0] = event.pv_y
                pv_z[0] = event.pv_z

                sv_x[0] = event.sv_x
                sv_y[0] = event.sv_y
                sv_z[0] = event.sv_z

                tv_tauplus_x[0] = event.tv_tauplus_x
                tv_tauplus_y[0] = event.tv_tauplus_y
                tv_tauplus_z[0] = event.tv_tauplus_z

                tv_tauminus_x[0] = event.tv_tauminus_x
                tv_tauminus_y[0] = event.tv_tauminus_y
                tv_tauminus_z[0] = event.tv_tauminus_z

                output_tree.Fill()

    output_file.Write()
    output_file.Close()

def main(argv):
    argsdict = {}

    try:
        for f, n in zip(argv[1::2], argv[2::2]):
            argsdict[f] = int(n)
    except Exception as e:
        print 'Something went wrong...'
        print e
        print 'Usage: merger.py FILENAME_1 NUMBER_OF_EVENTS_1 FILENAME_2 NUMBER_OF_EVENTS_2 ...'

    merge(argsdict)

if __name__ == '__main__':
    main(sys.argv)
