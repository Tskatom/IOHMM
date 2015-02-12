"""
Created on Dec 15 2014
Tools for Log
"""
__author__ = 'weiwang'
__email__ = 'tskatom@vt.edu'

import os
import logging
import sys
import re

def getLogFile(logfile=None):
    if logfile:
        lgf = logfile
    else:
        lgf = sys.argv[0]

    (d, p) = os.path.split(lgf)
    p = re.sub(r'(\.py)?', '.log', p)

    logfolder = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logfolder):
        logfolder = os.path.join(os.path.realpath(d), 'logs')
        if not os.path.exists(logfolder):
            logfolder = os.path.realpath(d)

    return os.path.join(logfolder, p)

def init(args=None, l=logging.INFO, logfile=None):
    logf = getLogFile(logfile)
    if args and vars(args).get('verbose', False):
        l = logging.DEBUG
        l_format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(funcName)s - %(message)s'
        logging.basicConfig(filename=logf,
                            format=l_format,
                            level=l)

def getLogger(log_name):
    return logging.getLogger(log_name)
