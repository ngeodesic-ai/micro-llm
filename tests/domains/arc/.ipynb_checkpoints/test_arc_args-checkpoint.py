import numpy as np
from micro_lm.domains.arc import args as arc_args

def test_rotate_k_basic():
    T=20; K=4
    env = np.zeros((T,K), dtype=np.float32)
    env[5:10,2] = 1.0  # channel 2 has biggest area in window
    k = arc_args.extract_rotate_k(env, (0,19))
    assert k == 2

def test_translate_extractor_centered_window():
    T=10
    com = np.stack([np.linspace(0,3,T), np.linspace(1,1,T)], axis=1)
    dx,dy = arc_args.extract_translate(com, (2,9))
    assert dx == 2 and dy == 0

def test_crop_bbox_halfmax():
    H=W=8
    E = np.zeros((H,W), dtype=np.float32)
    E[2:5,3:6] = 10.0
    x0,y0,x1,y1 = arc_args.extract_crop_bbox(E, (0,0), 0.5)
    assert (x0,y0,x1,y1) == (3,2,5,4)