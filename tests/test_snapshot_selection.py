from hbmon.worker import _select_best_detection, calculate_detection_score, Det

def test_select_best_detection_prioritizes_confidence():
    """
    Verify that _select_best_detection prioritizes confidence over area.
    """
    # Det(x1, y1, x2, y2, conf)
    # Area calculation: (x2-x1)*(y2-y1)
    
    # Large box, low confidence: 100x100 = 10000px, conf 0.5
    large_low_conf = Det(0, 0, 100, 100, 0.5)
    
    # Small box, high confidence: 20x20 = 400px, conf 0.9
    small_high_conf = Det(0, 0, 20, 20, 0.9)
    
    # Medium box, medium confidence: 50x50 = 2500px, conf 0.7
    medium_mid_conf = Det(0, 0, 50, 50, 0.7)
    
    entries = [
        (large_low_conf, {}),
        (small_high_conf, {}),
        (medium_mid_conf, {})
    ]
    
    # The winner should be small_high_conf because 0.9 > 0.7 > 0.5
    winner, _ = _select_best_detection(entries)
    
    assert winner == small_high_conf
    assert winner.conf == 0.9
    assert winner.area == 400

def test_select_best_detection_tiebreaker_area():
    """
    Verify that if confidence is equal, larger area wins.
    """
    # Small box, high confidence: 20x20 = 400px, conf 0.9
    small_high_conf = Det(0, 0, 20, 20, 0.9)
    
    # Large box, high confidence: 100x100 = 10000px, conf 0.9
    large_high_conf = Det(0, 0, 100, 100, 0.9)
    
    entries = [
        (small_high_conf, {}),
        (large_high_conf, {})
    ]
    
    winner, _ = _select_best_detection(entries)
    
    # Confidence tied at 0.9, area 10000 > 400
    assert winner == large_high_conf
    assert winner.area == 10000

def test_visit_scoring_logic_simulation():
    """
    Simulate the visit loop scoring logic change.
    Old logic: score = area * conf
    New logic: score = (conf, area)
    """
    # Large box (10000), low conf (0.5)
    # Old score: 5000
    # New score: (0.5, 10000)
    det1_area = 10000
    det1_conf = 0.5
    
    # Small box (400), high conf (0.9)
    # Old score: 360
    # New score: (0.9, 400)
    det2_area = 400
    det2_conf = 0.9
    
    # Verify old logic behavior (undesired)
    old_score1 = det1_area * det1_conf
    old_score2 = det2_area * det2_conf
    assert old_score1 > old_score2
    
    # Verify new logic behavior (desired)
    new_score1 = calculate_detection_score(Det(0, 0, 100, 100, det1_conf)) # Use dummy coords to get area=10000
    new_score2 = calculate_detection_score(Det(0, 0, 20, 20, det2_conf))   # Use dummy coords to get area=400
    assert new_score2 > new_score1
