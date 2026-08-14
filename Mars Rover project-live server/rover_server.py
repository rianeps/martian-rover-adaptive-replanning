"""
Martian Rover Real-Time Navigation Server
"""

import socket
import json
import threading
import time
import base64
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image
import cv2
import heapq
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR = '/Users/ria/Desktop/Dissertation-Mars Rover Project/Mars Rover Project'
PORT        = 9000


GRID_SIZE      = 100   
FINE_GRID_SIZE = 400
START          = (7, 7)
GOAL           = (90, 90)

# ── Terrain constants 
SAFE      = 0
ROCKY     = 1
HAZARDOUS = 2
BASE_COST = {SAFE: 1, ROCKY: 50, HAZARDOUS: 500}
# Extra cost added to an edge when mid-segment checking finds hazardous terrain
# between two waypoints that the coarse grid itself missed.
SEGMENT_HAZARD_PENALTY = 1000
CLEAN     = 'Clean'
DEGRADED  = 'Degraded'
CRITICAL  = 'Critical'
ROCKY_MULTIPLIERS = {CLEAN: 1.0, DEGRADED: 8.0, CRITICAL: 20.0}


#  Load CNN 
def load_cnn():
    print('Loading CNN terrain classifier...')
    checkpoint    = torch.load(
        os.path.join(PROJECT_DIR, 'terrain_classifier.pth'),
        map_location='cpu'
    )
    CLASS_NAMES   = checkpoint['class_names']
    NUM_CLASSES   = checkpoint['num_classes']
    IMAGENET_MEAN = checkpoint['imagenet_mean']
    IMAGENET_STD  = checkpoint['imagenet_std']
    IMG_SIZE      = checkpoint['img_size']

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print(f'CNN loaded. Classes: {CLASS_NAMES}')
    return model, transform, CLASS_NAMES


#  Load monitor config 
def load_monitor_config():
    with open(os.path.join(PROJECT_DIR, 'monitor_config.json')) as f:
        cfg = json.load(f)
    print('Monitor config loaded.')
    return (
        cfg['ssim_thresholds']['clean'],
        cfg['ssim_thresholds']['degraded'],
        cfg['confidence_thresholds']['clean'],
        cfg['confidence_thresholds']['degraded'],
    )


#  Load terrain 
def parse_obj(obj_path):
    print('Parsing OBJ terrain file...')
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    print(f'Vertices parsed: {len(vertices):,}')
    return np.array(vertices)


def build_elevation_grid(vertices, grid_size=30):
   
    x_coords = vertices[:, 0]
    y_coords = -vertices[:, 1]
    z_coords = vertices[:, 2]

    x_int = np.round(vertices[:, 0]).astype(int)
    y_int = np.round(vertices[:, 1]).astype(int)

    x_max_int = x_int.max()
    y_max_int = y_int.max()
    native_size = max(x_max_int, y_max_int) + 1  # e.g. 400 for range 0-399

    bounds = {
        'x_min': x_coords.min(), 'x_max': x_coords.max(),
        'y_min': y_coords.min(), 'y_max': y_coords.max(),
        'elev_min': z_coords.min(), 'elev_max': z_coords.max()
    }

    elevation_grid = np.full((native_size, native_size), -np.inf)
    for i in range(len(vertices)):
        col = x_int[i]
        row = y_max_int - y_int[i]  # matches the negated-Y orientation the binned version used
        if z_coords[i] > elevation_grid[row][col]:
            elevation_grid[row][col] = z_coords[i]

    empty = np.isneginf(elevation_grid)
    if empty.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
        elevation_grid[empty] = elevation_grid[tuple(idx[:, empty])]
        print(f'  (filled {empty.sum()} missing lattice points via nearest-neighbour)')

    return elevation_grid, bounds


def compute_slope_grid(elevation_grid):
    rows, cols = elevation_grid.shape
    slope_grid = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            elev   = elevation_grid[r][c]
            slopes = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    slopes.append(abs(elev - elevation_grid[nr][nc]))
            slope_grid[r][c] = max(slopes) if slopes else 0
    return slope_grid


def classify_terrain(slope_grid):
   
    safe_threshold   = 0.268   # ~15 degrees
    hazard_threshold = 0.364   # ~20 degrees
    terrain_grid = np.zeros(slope_grid.shape, dtype=int)
    terrain_grid[slope_grid > safe_threshold]   = ROCKY
    terrain_grid[slope_grid > hazard_threshold] = HAZARDOUS
    return terrain_grid


def downsample_terrain_worstcase(fine_grid, coarse_size):
    
    fine_size = fine_grid.shape[0]
    factor = fine_size / coarse_size
    coarse = np.zeros((coarse_size, coarse_size), dtype=int)
    for i in range(coarse_size):
        for j in range(coarse_size):
            r0, r1 = int(i * factor), max(int((i + 1) * factor), int(i * factor) + 1)
            c0, c1 = int(j * factor), max(int((j + 1) * factor), int(j * factor) + 1)
            block = fine_grid[r0:r1, c0:c1]
            coarse[i][j] = block.max() if block.size > 0 else SAFE
    return coarse


#  Navigation 
def build_cost_map(terrain_grid, visited=None, multiplier=1.0):
    cost_map = np.zeros(terrain_grid.shape)
    for r in range(terrain_grid.shape[0]):
        for c in range(terrain_grid.shape[1]):
            base = BASE_COST[terrain_grid[r][c]]
            if visited and (r,c) not in visited and terrain_grid[r][c] == ROCKY:
                cost_map[r][c] = base * multiplier
            else:
                cost_map[r][c] = base
    return cost_map


def simplify_path(path):
  
    if len(path) < 3:
        return path
    simplified = [path[0]]
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(1, len(path) - 1):
        cur_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if cur_dir != prev_dir:
            simplified.append(path[i])
            prev_dir = cur_dir
    simplified.append(path[-1])
    return simplified


def astar(terrain_grid, cost_map, start, goal, segment_penalty_fn=None):
    
    rows, cols = terrain_grid.shape
    open_set   = []
    heapq.heappush(open_set, (0, 0, start))
    g_score   = {start: 0}
    came_from = {}
    visited   = set()

    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_set:
        f, g, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                nb = (nx, ny)
                edge_penalty = segment_penalty_fn(cur, nb) if segment_penalty_fn else 0
                tg = g_score[cur] + cost_map[nx][ny] + edge_penalty
                if tg < g_score.get(nb, float('inf')):
                    came_from[nb] = cur
                    g_score[nb]   = tg
                    heapq.heappush(open_set, (tg+h(nb,goal), tg, nb))
    return None


def grid_to_unity(row, col, elevation_grid, bounds):
    elev_min   = bounds['elev_min']
    elev_max   = bounds['elev_max']
    elev_range = elev_max - elev_min
    SCALE_X  = 39900.0
    SCALE_Z  = 39900.0
    OFFSET_X = -39900.0
    OFFSET_Z = -39900.0
    HOVER    = -1595.0
    SCALE_Y  = 0.0
    unity_x = (col / GRID_SIZE) * SCALE_X + OFFSET_X
    unity_z = (row / GRID_SIZE) * SCALE_Z + OFFSET_Z
    # elevation_grid is native resolution (FINE_GRID_SIZE) but row/col here are
    # coarse planning-grid indices — scale before indexing. Currently harmless
    # since SCALE_Y=0 means elev never affects the output, but kept correct.
    native_scale = FINE_GRID_SIZE / GRID_SIZE
    native_row = min(int(row * native_scale), FINE_GRID_SIZE - 1)
    native_col = min(int(col * native_scale), FINE_GRID_SIZE - 1)
    elev    = elevation_grid[native_row][native_col]
    norm    = (elev - elev_min) / elev_range if elev_range > 0 else 0
    unity_y = norm * SCALE_Y + HOVER
    return round(unity_x, 2), round(unity_y, 2), round(unity_z, 2)


#  Degradation 
def apply_gaussian_noise(frame, severity=0.5, seed=None):
    rng = np.random.default_rng(seed)
    return np.clip(
        frame.astype(np.float32) + rng.normal(0, severity*80, frame.shape),
        0, 255).astype(np.uint8)


def apply_occlusion(frame, severity=0.3, seed=None):
    rng = np.random.default_rng(seed)
    d = frame.copy()
    h, w = frame.shape[:2]
    oh, ow = int(h*severity*0.9), int(w*severity*0.9)
    if oh > 0 and ow > 0:
        t = rng.integers(0, max(1, h-oh))
        l = rng.integers(0, max(1, w-ow))
        d[t:t+oh, l:l+ow] = 160
    return d


def apply_dropout(frame, severity=0.3, seed=None):
    rng = np.random.default_rng(seed)
    d = frame.copy()
    d[rng.random(frame.shape[:2]) < severity] = 0
    return d


DEGRADATION_FNS = {
    'gaussian':  apply_gaussian_noise,
    'occlusion': apply_occlusion,
    'dropout':   apply_dropout,
}


#  Server 
class RoverServer:
    def __init__(self):
        print('Initialising server...')
        self.cnn_model, self.transform, self.class_names = load_cnn()
        ssim_clean, ssim_deg, conf_clean, conf_deg       = load_monitor_config()
        self.SSIM_CLEAN    = ssim_clean
        self.SSIM_DEGRADED = ssim_deg
        
        self.CONF_CLEAN    = 0.18
        self.CONF_DEGRADED = 0.10
        print(f'Confidence thresholds recalibrated for synthetic renders: '
              f'clean={self.CONF_CLEAN} (was {conf_clean}), '
              f'degraded={self.CONF_DEGRADED} (was {conf_deg})')

        vertices = parse_obj(os.path.join(PROJECT_DIR, 'terrain', 'model.obj'))

        
        unique_verts = np.unique(vertices, axis=0)
        x_is_int = np.allclose(vertices[:, 0], np.round(vertices[:, 0]))
        y_is_int = np.allclose(vertices[:, 1], np.round(vertices[:, 1]))
        print(f'Vertex diagnostic — total: {len(vertices):,}, unique: {len(unique_verts):,}, '
              f'duplication factor: {len(vertices)/max(len(unique_verts),1):.2f}x')
        print(f'  X integer-lattice: {x_is_int}, Y integer-lattice: {y_is_int}')
        if x_is_int and y_is_int:
            print(f'  X range: {vertices[:,0].min():.0f} to {vertices[:,0].max():.0f} '
                  f'({int(vertices[:,0].max()-vertices[:,0].min())+1} unique values)')
            print(f'  Y range: {vertices[:,1].min():.0f} to {vertices[:,1].max():.0f} '
                  f'({int(vertices[:,1].max()-vertices[:,1].min())+1} unique values)')

        
        self.elevation_grid, self.bounds = build_elevation_grid(vertices, FINE_GRID_SIZE)
        fine_slope_grid = compute_slope_grid(self.elevation_grid)
        print(f'Native slope stats — min: {fine_slope_grid.min():.3f}, max: {fine_slope_grid.max():.3f}, '
              f'mean: {fine_slope_grid.mean():.3f}, median: {np.median(fine_slope_grid):.3f}')
        self.fine_terrain_grid = classify_terrain(fine_slope_grid)
        print(f'Fine (native) terrain: Safe={(self.fine_terrain_grid==SAFE).sum()} '
              f'Rocky={(self.fine_terrain_grid==ROCKY).sum()} '
              f'Hazardous={(self.fine_terrain_grid==HAZARDOUS).sum()}')

       
        print(f'Downsampling to {GRID_SIZE}x{GRID_SIZE} planning grid (worst-case aggregation)...')
        self.terrain_grid = downsample_terrain_worstcase(self.fine_terrain_grid, GRID_SIZE)

        print(f'START cell terrain: {["Safe","Rocky","Hazardous"][self.terrain_grid[START]]}')
        print(f'GOAL cell terrain:  {["Safe","Rocky","Hazardous"][self.terrain_grid[GOAL]]}')
     
        clear_radius = 2  # coarse-grid cells
        rows, cols = self.terrain_grid.shape
        for center in (START, GOAL):
            for dr in range(-clear_radius, clear_radius + 1):
                for dc in range(-clear_radius, clear_radius + 1):
                    r, c = center[0] + dr, center[1] + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        self.terrain_grid[r][c] = SAFE

        native_scale = FINE_GRID_SIZE / GRID_SIZE
        fine_rows, fine_cols = self.fine_terrain_grid.shape
        fine_clear_radius = int(clear_radius * native_scale)
        for center in (START, GOAL):
            fr_center = int(center[0] * native_scale)
            fc_center = int(center[1] * native_scale)
            for dr in range(-fine_clear_radius, fine_clear_radius + 1):
                for dc in range(-fine_clear_radius, fine_clear_radius + 1):
                    fr, fc = fr_center + dr, fc_center + dc
                    if 0 <= fr < fine_rows and 0 <= fc < fine_cols:
                        self.fine_terrain_grid[fr][fc] = SAFE

        print(f'Coarse (planning) terrain: Safe={(self.terrain_grid==SAFE).sum()} '
              f'Rocky={(self.terrain_grid==ROCKY).sum()} '
              f'Hazardous={(self.terrain_grid==HAZARDOUS).sum()}')

        self.sample_images = {}
        data_root = os.path.join(PROJECT_DIR, 'mars_terrain', 'Auburn_1')
        for cls in self.class_names:
            cls_path = os.path.join(data_root, cls)
            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))]
                if files:
                    self.sample_images[cls] = np.array(
                        Image.open(os.path.join(cls_path, files[0])).convert('RGB'))

        self.nav_state = {
            'path':         None,
            'current_idx':  0,
            'visited':      set(),
            'replans':      0,
            'deg_type':     'gaussian',
            'deg_severity': 0.5,
            'step_count':   0,
        }
        print('Server initialised successfully.')

    def segment_penalty(self, cell_a, cell_b, samples=16):
        """
        Samples points along the straight line between two COARSE grid cells and
        checks each against the FINE terrain grid. Returns a large penalty if any
        sample is Hazardous, so A* avoids this edge unless there's truly no
        alternative — catching terrain features the coarse grid alone would miss
        between two individually-Safe waypoints.
        """
        ra, ca = cell_a
        rb, cb = cell_b
        worst_penalty = 0
        for t in np.linspace(0.0, 1.0, samples):
            r = ra + (rb - ra) * t
            c = ca + (cb - ca) * t
            fine_r = min(int((r / GRID_SIZE) * FINE_GRID_SIZE), FINE_GRID_SIZE - 1)
            fine_c = min(int((c / GRID_SIZE) * FINE_GRID_SIZE), FINE_GRID_SIZE - 1)
            fine_r = max(fine_r, 0)
            fine_c = max(fine_c, 0)
            if self.fine_terrain_grid[fine_r][fine_c] == HAZARDOUS:
                worst_penalty = SEGMENT_HAZARD_PENALTY
        return worst_penalty

    def get_terrain_frame(self, terrain_type):
        mapping = {SAFE: 'other', ROCKY: 'dark dune', HAZARDOUS: 'crater'}
        cls = mapping.get(terrain_type, 'other')
        return self.sample_images.get(cls, list(self.sample_images.values())[0])

    @staticmethod
    def decode_live_frame(base64_jpeg):
        try:
            raw = base64.b64decode(base64_jpeg)
            img = Image.open(BytesIO(raw)).convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f'Failed to decode live frame from Unity: {e}')
            return None

    def compute_ssim(self, clean, degraded):
        cg = cv2.cvtColor(clean,    cv2.COLOR_RGB2GRAY) if clean.ndim    == 3 else clean
        dg = cv2.cvtColor(degraded, cv2.COLOR_RGB2GRAY) if degraded.ndim == 3 else degraded
        return ssim_metric(cg, dg, data_range=255)

    def get_cnn_confidence(self, frame):
        pil = Image.fromarray(frame).convert('RGB')
        t   = self.transform(pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.cnn_model(t), dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()

    def assess_severity(self, ssim_score, confidence):
        if ssim_score < self.SSIM_DEGRADED or confidence < self.CONF_DEGRADED:
            return CRITICAL
        if ssim_score < self.SSIM_CLEAN or confidence < self.CONF_CLEAN:
            return DEGRADED
        return CLEAN

    def plan(self, start, goal, terrain_grid, visited=None, multiplier=1.0):
        t0 = time.time()
        cost_map = build_cost_map(terrain_grid, visited, multiplier)
        
        path = astar(terrain_grid, cost_map, start, goal, segment_penalty_fn=self.segment_penalty)
        elapsed = time.time() - t0
        raw_len = len(path) if path else 0
        if path:
            path = simplify_path(path)
        print(f'  [plan] {start} -> {goal} took {elapsed:.3f}s '
              f'({"found" if path else "NO PATH"}, {raw_len} raw steps -> {len(path) if path else 0} after simplify)')
        return path

    def print_ascii_map(self, path=None, max_width=100):
        """
        Prints a downsampled ASCII view of the terrain grid with the planned path
        overlaid to visually confirm what the algorithm actually sees
        '.' Safe | 'r' Rocky | '#' Hazardous | '*' planned path | 'S'/'G' start/goal
        """
        size = self.terrain_grid.shape[0]
        chars = np.full((size, size), '.', dtype='<U1')
        chars[self.terrain_grid == ROCKY] = 'r'
        chars[self.terrain_grid == HAZARDOUS] = '#'
        if path:
            for (r, c) in path:
                chars[r][c] = '*'
        chars[START] = 'S'
        chars[GOAL] = 'G'

        step = max(1, size // max_width)
        priority = {'S': 0, 'G': 1, '*': 2, '#': 3, 'r': 4, '.': 5}
        print(f'\nASCII terrain map (each char = {step}x{step} cells | '
              f'"." Safe  "r" Rocky  "#" Hazardous  "*" path  "S"/"G" start/goal):')
        for r in range(0, size, step):
            row_str = ''
            for c in range(0, size, step):
                block = chars[r:r+step, c:c+step].flatten()
                best = min(block, key=lambda ch: priority.get(ch, 9))
                row_str += best
            print(row_str)
        print()

    def reset_navigation(self):
        state             = self.nav_state
        state['path']     = None
        state['current_idx'] = 0
        state['visited']  = set()
        state['replans']  = 0
        state['step_count'] = 0
        state['path']     = self.plan(START, GOAL, self.terrain_grid)
        print(f'Navigation reset. Path: {len(state["path"]) if state["path"] else 0} steps')

        if state['path']:
            names = {SAFE:'S', ROCKY:'R', HAZARDOUS:'H'}
            seq = ''.join(names[self.terrain_grid[cell]] for cell in state['path'])
            print(f'Path terrain sequence: {seq}')
            print(f'  Safe={seq.count("S")} Rocky={seq.count("R")} Hazardous={seq.count("H")}')
            self.print_ascii_map(state['path'])

            native_scale = FINE_GRID_SIZE / GRID_SIZE
            elevations = [self.elevation_grid[min(int(cell[0]*native_scale), FINE_GRID_SIZE-1)]
                                              [min(int(cell[1]*native_scale), FINE_GRID_SIZE-1)]
                          for cell in state['path']]
            total_ascent = sum(max(0, elevations[i+1] - elevations[i]) for i in range(len(elevations)-1))
            total_descent = sum(max(0, elevations[i] - elevations[i+1]) for i in range(len(elevations)-1))
            net_change = elevations[-1] - elevations[0]
            print(f'Elevation profile — start: {elevations[0]:.2f}, end: {elevations[-1]:.2f}, '
                  f'net change: {net_change:.2f}, total ascent: {total_ascent:.2f}, '
                  f'total descent: {total_descent:.2f}, min: {min(elevations):.2f}, max: {max(elevations):.2f}')

    def run_step(self, live_frame=None):
        state = self.nav_state

        if state['path'] is None:
            state['path'] = self.plan(START, GOAL, self.terrain_grid)
            if state['path'] is None:
                return {'type': 'error', 'message': 'No path found'}

        if state['current_idx'] >= len(state['path']):
            return {
                'type':    'complete',
                'message': 'Goal reached',
                'replans': state['replans'],
                'steps':   state['step_count'],
            }

        cell         = state['path'][state['current_idx']]
        row, col     = cell
        terrain_type = self.terrain_grid[row][col]
        terrain_name = {SAFE:'Safe', ROCKY:'Rocky', HAZARDOUS:'Hazardous'}.get(terrain_type, 'Unknown')

        if live_frame is not None:
            clean_frame = live_frame
        else:
            clean_frame = self.get_terrain_frame(terrain_type)

        deg_fn      = DEGRADATION_FNS.get(state['deg_type'], apply_gaussian_noise)
        deg_frame   = deg_fn(clean_frame, state['deg_severity'])
        ssim_score  = self.compute_ssim(clean_frame, deg_frame)
        _, conf     = self.get_cnn_confidence(deg_frame)
        severity    = self.assess_severity(ssim_score, conf)

        replanned = False
        if severity in [DEGRADED, CRITICAL]:
            multiplier = ROCKY_MULTIPLIERS[severity]
            new_path   = self.plan(cell, GOAL, self.terrain_grid, state['visited'], multiplier)
            if new_path and len(new_path) > 1:
                state['path']        = list(state['visited']) + new_path
                state['current_idx'] = len(state['visited'])
                state['replans']    += 1
                replanned            = True
                print(f'REPLAN step={state["step_count"]} | {severity} | '
                      f'SSIM={ssim_score:.3f} | Conf={conf:.3f}')

        ux, uy, uz = grid_to_unity(row, col, self.elevation_grid, self.bounds)

        state['visited'].add(cell)
        state['current_idx'] += 1
        state['step_count']  += 1

        return {
            'type':       'waypoint',
            'step':       state['step_count'],
            'unity_x':    ux,
            'unity_y':    uy,
            'unity_z':    uz,
            'grid_row':   int(row),
            'grid_col':   int(col),
            'terrain':    terrain_name,
            'severity':   severity,
            'ssim':       round(ssim_score, 4),
            'confidence': round(conf, 4),
            'is_replan':  replanned,
            'replans':    state['replans'],
        }

    def handle_client(self, conn, addr):
        print(f'Unity connected from {addr}')
        self.reset_navigation()
        buffer = ''

        try:
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue

                    if line == 'NEXT':
                        result = self.run_step()
                        conn.sendall((json.dumps(result) + '\n').encode('utf-8'))

                    elif line.startswith('NEXT:'):
                        base64_payload = line[len('NEXT:'):]
                        live_frame = self.decode_live_frame(base64_payload)
                        result = self.run_step(live_frame=live_frame)
                        conn.sendall((json.dumps(result) + '\n').encode('utf-8'))

                    elif line == 'RESET':
                        self.reset_navigation()
                        conn.sendall(b'{"type":"ack","message":"reset"}\n')

                    elif line.startswith('SET_DEG:'):
                        parts = line.split(':')
                        if len(parts) == 3:
                            self.nav_state['deg_type']     = parts[1]
                            self.nav_state['deg_severity'] = float(parts[2])
                            conn.sendall(b'{"type":"ack"}\n')
                            print(f'Degradation: {parts[1]} severity={parts[2]}')

                    elif line == 'STATUS':
                        status = {
                            'type':         'status',
                            'step':         self.nav_state['step_count'],
                            'replans':      self.nav_state['replans'],
                            'deg_type':     self.nav_state['deg_type'],
                            'deg_severity': self.nav_state['deg_severity'],
                        }
                        conn.sendall((json.dumps(status) + '\n').encode('utf-8'))

        except Exception as e:
            print(f'Client error: {e}')
        finally:
            conn.close()
            print('Unity disconnected.')

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', PORT))
        server.listen(5)
        print(f'\nServer ready. Connect Unity to:')
        print(f'  Host: 127.0.0.1')
        print(f'  Port: {PORT}')
        print(f'\nWaiting for Unity...\n')

        while True:
            try:
                conn, addr = server.accept()
                t = threading.Thread(target=self.handle_client, args=(conn, addr))
                t.daemon = True
                t.start()
            except KeyboardInterrupt:
                print('\nServer stopped.')
                break
            except Exception as e:
                print(f'Server error: {e}')


if __name__ == '__main__':
    server = RoverServer()
    server.start()