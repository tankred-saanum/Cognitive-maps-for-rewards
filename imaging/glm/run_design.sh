#!/bin/bash
#
# run 1st Level analysis on the condor cluster
#
# (c) Mona Garvert <garvert@cbs.mpg.de>  August 2019


executable = /afs/cbs.mpg.de/software/scripts/envwrap
arguments = /data/pt_02071/choice-maps/imagingData/sub-SUBJID/session_SESSID/1stLevel/design_DESIGNID/run-design_sub-SUBJID
universe = vanilla
output = /data/pt_02071/choice-maps/imagingData/sub-SUBJID/session_SESSID/1stLevel/design_DESIGNID/sub-SUBJID_test.out 
error = /data/pt_02071/choice-maps/imagingData/sub-SUBJID/session_SESSID/1stLevel/design_DESIGNID/sub-SUBJID_test.error
log = /data/pt_02071/choice-maps/imagingData/sub-SUBJID/session_SESSID/1stLevel/design_DESIGNID/sub-SUBJID_test.log
request_memory = 5000
request_cpus = 2 
getenv = True
notification = Error
queue
