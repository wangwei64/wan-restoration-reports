"""Dispatch the unchanged, previously validated selector implementations."""
from cue_controls import intervention as cue
from neutral_controls import intervention as coverage
from random_controls import intervention as uniform
from region_controls import intervention as region
from roi_controls import intervention as roi

def intervention(arm,ref,seed,trace):
    if arm=='roi':return roi(arm,ref,seed,trace)
    if arm.startswith('random_'):return uniform(arm,ref,seed,trace)
    if arm.startswith(('subject_random_','detail_random_')):return region(arm,ref,seed,trace)
    if arm.startswith('uniform_coverage_'):return coverage(arm,ref,seed,trace)
    return cue(arm,ref,seed,trace)
