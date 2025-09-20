# Generate a synthetic ARC mapper dataset with 8 primitives (1000 rows)
import json, random, numpy as np, pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

N_TOTAL = 1000
PRIMITIVES = [
    "flip_h",
    "flip_v",
    "rot90",
    "color_map",
    "translate",
    "crop_pad",
    "tile",
    "cc_mask_largest",
]
N_PER = N_TOTAL // len(PRIMITIVES)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def empty_grid(H=12, W=12, bg=0):
    return np.full((H, W), bg, dtype=int)

def draw_block(g, r, c, h, w, color):
    H, W = g.shape
    r2 = clamp(r+h, 0, H)
    c2 = clamp(c+w, 0, W)
    g[clamp(r,0,H):r2, clamp(c,0,W):c2] = color

def rand_color(exclude=None):
    exclude = set(exclude or [])
    choices = [c for c in range(10) if c not in exclude]
    return int(np.random.choice(choices))

def gen_flip_h():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # left/right different structures
    for _ in range(np.random.randint(2,4)):
        color = rand_color({0})
        r = np.random.randint(0, H-3)
        c = np.random.randint(0, W//2 - 3)
        draw_block(g, r, c, np.random.randint(2,4), np.random.randint(2,4), color)
    for _ in range(np.random.randint(2,4)):
        color = rand_color({0})
        r = np.random.randint(0, H-3)
        c = np.random.randint(W//2 + 1, W-3)
        draw_block(g, r, c, np.random.randint(2,4), np.random.randint(2,4), color)
    return g

def gen_flip_v():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # top/bottom different structures
    for _ in range(np.random.randint(2,4)):
        color = rand_color({0})
        r = np.random.randint(0, H//2 - 3)
        c = np.random.randint(0, W-3)
        draw_block(g, r, c, np.random.randint(2,4), np.random.randint(2,4), color)
    for _ in range(np.random.randint(2,4)):
        color = rand_color({0})
        r = np.random.randint(H//2 + 1, H-3)
        c = np.random.randint(0, W-3)
        draw_block(g, r, c, np.random.randint(2,4), np.random.randint(2,4), color)
    return g

def gen_rot90():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # build an L-shape or rectangle off-center to encourage rotation cue
    color = rand_color({0})
    r = np.random.randint(2, H-6)
    c = np.random.randint(2, W-6)
    draw_block(g, r, c, 2, 6, color)
    draw_block(g, r, c, 6, 2, color)
    # add a small distinct block
    color2 = rand_color({0, color})
    draw_block(g, r+5, c+5, 2, 2, color2)
    return g

def gen_color_map():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # scatter many small colored pixels/1x2 blocks (palette-heavy)
    for _ in range(np.random.randint(18, 28)):
        color = rand_color({0})
        r = np.random.randint(0, H)
        c = np.random.randint(0, W)
        g[r, c] = color
        # sometimes extend to a tiny 1x2 or 2x1
        if np.random.rand() < 0.4 and c+1 < W:
            g[r, c+1] = color
        if np.random.rand() < 0.4 and r+1 < H:
            g[r+1, c] = color
    return g

def gen_translate():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # place a single object near an edge
    color = rand_color({0})
    side = np.random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        r = np.random.randint(2, H-4); c = 1
    elif side == "right":
        r = np.random.randint(2, H-4); c = W-4
    elif side == "top":
        r = 1; c = np.random.randint(2, W-4)
    else:
        r = H-4; c = np.random.randint(2, W-4)
    draw_block(g, r, c, 3, 3, color)
    return g

def gen_crop_pad():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # draw a border (padding) and a small object near a corner
    border_color = rand_color({0})
    g[0, :] = border_color
    g[-1, :] = border_color
    g[:, 0] = border_color
    g[:, -1] = border_color
    color = rand_color({0, border_color})
    corner = np.random.choice(["tl","tr","bl","br"])
    if corner == "tl":
        draw_block(g, 1, 1, 3, 3, color)
    elif corner == "tr":
        draw_block(g, 1, W-4, 3, 3, color)
    elif corner == "bl":
        draw_block(g, H-4, 1, 3, 3, color)
    else:
        draw_block(g, H-4, W-4, 3, 3, color)
    return g

def gen_tile():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # create a 3x3 motif and tile it 3x3 times
    motif = np.zeros((3,3), dtype=int)
    motif[0,0] = rand_color({0})
    motif[1,1] = rand_color({0, motif[0,0]})
    motif[2,2] = rand_color({0, motif[0,0], motif[1,1]})
    for i in range(0, H, 3):
        for j in range(0, W, 3):
            g[i:i+3, j:j+3] = motif
    return g

def gen_cc_mask_largest():
    H, W = 12, 12
    g = empty_grid(H, W, 0)
    # multiple small components; one clearly largest
    # draw several small 2x2 blocks + one 4x4
    colors = [rand_color({0}) for _ in range(5)]
    # small comps
    for color in colors[:4]:
        r = np.random.randint(0, H-2); c = np.random.randint(0, W-2)
        draw_block(g, r, c, 2, 2, color)
    # largest comp
    bigc = colors[4]
    r = np.random.randint(0, H-5); c = np.random.randint(0, W-5)
    draw_block(g, r, c, 4, 4, bigc)
    return g

GENS = {
    "flip_h": gen_flip_h,
    "flip_v": gen_flip_v,
    "rot90": gen_rot90,
    "color_map": gen_color_map,
    "translate": gen_translate,
    "crop_pad": gen_crop_pad,
    "tile": gen_tile,
    "cc_mask_largest": gen_cc_mask_largest,
}

rows = []
for label in PRIMITIVES:
    gen = GENS[label]
    for _ in range(N_PER):
        g = gen()
        rows.append({
            "grid": json.dumps(g.tolist()),
            "label": label
        })

# If N_TOTAL is not divisible by len(PRIMITIVES), pad remaining with random classes
while len(rows) < N_TOTAL:
    label = np.random.choice(PRIMITIVES)
    g = GENS[label]()
    rows.append({"grid": json.dumps(g.tolist()), "label": label})

df = pd.DataFrame(rows)
out_path = Path("tests/fixtures/arc/arc_mapper_labeled.csv")
df.to_csv(out_path, index=False)

# Show a small sample
df_sample = df.sample(10, random_state=123)
import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("ARC Mapper Labeled (sample)", df_sample)

out_path.as_posix()
