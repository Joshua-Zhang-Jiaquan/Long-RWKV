import math
from lrwkv_evidence.theory_mvp.exact import scan_chain,predicted_scans,eulerian,optimal_binary_compression,enumerate_scans


def test_adversarial_order_and_cache_counterexample():
    assert scan_chain([3,2,1,0])[0]==4
    assert scan_chain([3,2,1,0],alternating=True)[0]==2
    assert scan_chain([1,0])[0]==2
    assert scan_chain([1,0],cache_records=1)[0]==1


def test_exact_independent_simulation_matches_formulas():
    rows=enumerate_scans(6)
    assert rows[-1]['permutations']==720
    assert rows[-1]['alternating_mean_scans']>rows[-1]['forward_mean_scans']
    assert eulerian(4)==[1,11,11,1]


def test_nontrivial_compression_beats_store_one_bit():
    r=optimal_binary_compression(3,1)
    assert r['optimal_accuracy']==.75
    assert r['optimal_accuracy']>r['naive_store_bits_accuracy']
    assert optimal_binary_compression(3,0)['optimal_accuracy']==.5
    assert optimal_binary_compression(3,3)['optimal_accuracy']==1
