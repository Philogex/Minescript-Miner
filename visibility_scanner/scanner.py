from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict, Any, FrozenSet, Mapping, NamedTuple, Sequence
from functools import lru_cache
from numba import types
from numba.typed import Dict

import numpy as np
import numba as nb

# ------------------------------
# consts
# ------------------------------

EPS = 1e-12
INF = 1e300
BVH_THRESHOLD = 8192


# ------------------------------
# types
# ------------------------------

Vec3 = Tuple[float, float, float]
BlockPos = Tuple[int, int, int]
AABB = Tuple[float, float, float, float, float, float]

class TargetInfo(NamedTuple):
    world_pos: tuple[int, int, int]
    centroid_world_pos: tuple[float, float, float]
    target_angle: tuple[float, float]
    yaw_bounds: tuple[float, float]
    pitch_bounds: tuple[float, float]


# ------------------------------
# bvh builder/refit and bvh traversal
# ------------------------------

def build_bvh(prims_min: np.ndarray, prims_max: np.ndarray, prim_ids: np.ndarray, max_leaf_size: int = 8):
    Na = prims_min.shape[0]

    node_min_list = []
    node_max_list = []
    node_left = []
    node_right = []
    node_first = []
    node_count = []
    leaf_prim_indices = []
    postorder_nodes = []

    centroids = 0.5 * (prims_min + prims_max)

    prim_indices_init = np.arange(Na, dtype=np.int32)

    def _build_recursive(prim_idxs: np.ndarray):
        node_idx = len(node_min_list)
        node_min_list.append(np.empty(3, dtype=np.float64))
        node_max_list.append(np.empty(3, dtype=np.float64))
        node_left.append(-1)
        node_right.append(-1)
        node_first.append(-1)
        node_count.append(0)

        if prim_idxs.size == 0:
            node_min_list[node_idx][:] = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            node_max_list[node_idx][:] = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
            node_first[node_idx] = 0
            node_count[node_idx] = 0
            postorder_nodes.append(node_idx)
            return node_idx

        if prim_idxs.size <= max_leaf_size:
            mn = prims_min[prim_idxs[0]].copy()
            mx = prims_max[prim_idxs[0]].copy()
            for i in range(1, prim_idxs.size):
                pi = int(prim_idxs[i])
                mn = np.minimum(mn, prims_min[pi])
                mx = np.maximum(mx, prims_max[pi])
            node_min_list[node_idx][:] = mn
            node_max_list[node_idx][:] = mx
            node_first[node_idx] = len(leaf_prim_indices)
            for i in range(prim_idxs.size):
                leaf_prim_indices.append(int(prim_idxs[i]))
            node_count[node_idx] = prim_idxs.size
            postorder_nodes.append(node_idx)
            return node_idx

        mn = prims_min[prim_idxs[0]].copy()
        mx = prims_max[prim_idxs[0]].copy()
        for i in range(1, prim_idxs.size):
            pi = int(prim_idxs[i])
            mn = np.minimum(mn, prims_min[pi])
            mx = np.maximum(mx, prims_max[pi])
        node_min_list[node_idx][:] = mn
        node_max_list[node_idx][:] = mx

        ext = mx - mn
        axis = int(np.argmax(ext))

        cent = centroids[prim_idxs, axis]
        order = np.argsort(cent)
        mid = prim_idxs.size // 2
        left_idxs = prim_idxs[order[:mid]]
        right_idxs = prim_idxs[order[mid:]]

        left_idx = _build_recursive(left_idxs)
        right_idx = _build_recursive(right_idxs)
        node_left[node_idx] = left_idx
        node_right[node_idx] = right_idx

        postorder_nodes.append(node_idx)
        return node_idx

    _build_recursive(prim_indices_init)

    node_min = np.vstack(node_min_list).astype(np.float64)
    node_max = np.vstack(node_max_list).astype(np.float64)
    node_left_arr = np.array(node_left, dtype=np.int32)
    node_right_arr = np.array(node_right, dtype=np.int32)
    node_first_arr = np.array(node_first, dtype=np.int32)
    node_count_arr = np.array(node_count, dtype=np.int32)
    leaf_prim_indices_arr = np.array(leaf_prim_indices, dtype=np.int32)
    postorder_arr = np.array(postorder_nodes, dtype=np.int32)

    return (node_min, node_max, node_left_arr, node_right_arr,
            node_first_arr, node_count_arr, leaf_prim_indices_arr, postorder_arr)

@nb.njit(cache=True, fastmath=True)
def bvh_refit_numba(node_min, node_max,
                    node_left, node_right,
                    node_first, node_count, leaf_prim_indices,
                    prim_min, prim_max,
                    postorder_nodes):
    n_post = postorder_nodes.shape[0]
    for idx in range(n_post):
        node = postorder_nodes[idx]
        l = node_left[node]; r = node_right[node]
        if l == -1 and r == -1:
            first = node_first[node]; cnt = node_count[node]
            if cnt <= 0:
                node_min[node, 0] = INF
                node_min[node, 1] = INF
                node_min[node, 2] = INF
                node_max[node, 0] = -INF
                node_max[node, 1] = -INF
                node_max[node, 2] = -INF
            else:
                pi0 = leaf_prim_indices[first]
                node_min[node, 0] = prim_min[pi0, 0]
                node_min[node, 1] = prim_min[pi0, 1]
                node_min[node, 2] = prim_min[pi0, 2]
                node_max[node, 0] = prim_max[pi0, 0]
                node_max[node, 1] = prim_max[pi0, 1]
                node_max[node, 2] = prim_max[pi0, 2]
                for k in range(1, cnt):
                    pi = leaf_prim_indices[first + k]
                    if prim_min[pi, 0] < node_min[node, 0]:
                        node_min[node, 0] = prim_min[pi, 0]
                    if prim_min[pi, 1] < node_min[node, 1]:
                        node_min[node, 1] = prim_min[pi, 1]
                    if prim_min[pi, 2] < node_min[node, 2]:
                        node_min[node, 2] = prim_min[pi, 2]

                    if prim_max[pi, 0] > node_max[node, 0]:
                        node_max[node, 0] = prim_max[pi, 0]
                    if prim_max[pi, 1] > node_max[node, 1]:
                        node_max[node, 1] = prim_max[pi, 1]
                    if prim_max[pi, 2] > node_max[node, 2]:
                        node_max[node, 2] = prim_max[pi, 2]
        else:
            if l != -1:
                node_min[node, 0] = node_min[l, 0]
                node_min[node, 1] = node_min[l, 1]
                node_min[node, 2] = node_min[l, 2]
                node_max[node, 0] = node_max[l, 0]
                node_max[node, 1] = node_max[l, 1]
                node_max[node, 2] = node_max[l, 2]
            else:
                node_min[node, 0] = INF
                node_min[node, 1] = INF
                node_min[node, 2] = INF
                node_max[node, 0] = -INF
                node_max[node, 1] = -INF
                node_max[node, 2] = -INF

            if r != -1:
                if node_min[r, 0] < node_min[node, 0]:
                    node_min[node, 0] = node_min[r, 0]
                if node_min[r, 1] < node_min[node, 1]:
                    node_min[node, 1] = node_min[r, 1]
                if node_min[r, 2] < node_min[node, 2]:
                    node_min[node, 2] = node_min[r, 2]

                if node_max[r, 0] > node_max[node, 0]:
                    node_max[node, 0] = node_max[r, 0]
                if node_max[r, 1] > node_max[node, 1]:
                    node_max[node, 1] = node_max[r, 1]
                if node_max[r, 2] > node_max[node, 2]:
                    node_max[node, 2] = node_max[r, 2]

@nb.njit(cache=True, parallel=True, fastmath=True)
def rasterize_with_bvh_nb(
    px: float, py: float, pz: float,
    dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
    node_min: np.ndarray, node_max: np.ndarray,
    node_left: np.ndarray, node_right: np.ndarray,
    node_first: np.ndarray, node_count: np.ndarray,
    leaf_prim_indices: np.ndarray,
    prim_min: np.ndarray, prim_max: np.ndarray, prim_id: np.ndarray,
    depth: np.ndarray, top_idx: np.ndarray,
    max_depth: float
):
    nrays = dx.shape[0]
    n_nodes = node_min.shape[0]
    STACK_SIZE = 128
    for i in nb.prange(nrays):
        di = dx[i]; dj = dy[i]; dk = dz[i]
        best_t = depth[i]
        best_oid = top_idx[i]
        stack = np.empty(STACK_SIZE, dtype=np.int32)
        sp = 0
        if n_nodes == 0:
            depth[i] = best_t
            top_idx[i] = best_oid
            continue
        stack[sp] = 0
        sp += 1
        while sp > 0:
            sp -= 1
            node_idx = stack[sp]
            tmin, tmax = _ray_aabb_intersect_single(px, py, pz, di, dj, dk,
                                                    node_min[node_idx, 0], node_max[node_idx, 0],
                                                    node_min[node_idx, 1], node_max[node_idx, 1],
                                                    node_min[node_idx, 2], node_max[node_idx, 2])
            if math.isnan(tmin):
                continue
            t_entry = tmin if tmin >= 0.0 else 0.0
            if math.isnan(t_entry) or t_entry < 0.0 or t_entry > max_depth:
                continue
            if t_entry >= best_t:
                continue

            left = node_left[node_idx]
            right = node_right[node_idx]
            if left == -1 and right == -1:
                first = node_first[node_idx]
                cnt = node_count[node_idx]
                for p_i in range(first, first + cnt):
                    prim_idx = leaf_prim_indices[p_i]
                    tmin_p, tmax_p = _ray_aabb_intersect_single(px, py, pz, di, dj, dk,
                                                prim_min[prim_idx, 0], prim_max[prim_idx, 0],
                                                prim_min[prim_idx, 1], prim_max[prim_idx, 1],
                                                prim_min[prim_idx, 2], prim_max[prim_idx, 2])
                    if math.isnan(tmin_p):
                        continue
                    tt = tmin_p if tmin_p >= 0.0 else tmax_p
                    if math.isnan(tt) or tt < 0.0:
                        continue
                    if tt > max_depth:
                        continue
                    if tt < best_t:
                        best_t = tt
                        best_oid = prim_id[prim_idx]
                continue

            if right != -1:
                stack[sp] = right; sp += 1
            if left != -1:
                stack[sp] = left; sp += 1

        depth[i] = best_t
        top_idx[i] = best_oid


# ------------------------------
# angle and direction helpers
# ------------------------------

def distance_to_block(position, ref):
    px, py, pz = position
    rx, ry, rz = ref
    dx = px - rx
    dy = py - ry
    dz = pz - rz
    return math.sqrt(dx * dx + dy * dy + dz * dz)

@nb.njit(cache=True, fastmath=True)
def distance_to_block_nb(position, ref):
    px, py, pz = position
    rx, ry, rz = ref
    dx = px - rx
    dy = py - ry
    dz = pz - rz
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def distances_to_blocks(previous_target: Tuple[float, float, float],
                        positions: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    prev = np.asarray(previous_target, dtype=np.float64)
    pos_arr = np.asarray(positions, dtype=np.float64)
    diffs = pos_arr - prev[None, :]
    return np.sqrt(np.sum(diffs * diffs, axis=1))

@nb.njit(cache=True, fastmath=True)
def distances_to_blocks_nb(prev: np.ndarray, pos_arr: np.ndarray) -> np.ndarray:
    n = pos_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        dx = pos_arr[i, 0] - prev[0]
        dy = pos_arr[i, 1] - prev[1]
        dz = pos_arr[i, 2] - prev[2]
        out[i] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return out

@nb.njit(cache=True, fastmath=True)
def normalize_angle_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

@nb.njit(cache=True, inline='always')
def normalize_angle_rad_nb(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

@nb.njit(cache=True, fastmath=True)
def mc_angles_to_internal_radians(mc_yaw_deg: float, mc_pitch_deg: float) -> Tuple[float, float]:
    yaw_rad = math.radians(mc_yaw_deg + 90.0)
    pitch_rad = math.radians(mc_pitch_deg)
    return yaw_rad, pitch_rad

@nb.njit(cache=True, fastmath=True)
def to_minecraft_angles_degrees(yaw_rad: float, pitch_rad: float) -> Tuple[float, float]:
    yaw_deg = math.degrees(yaw_rad) - 90.0
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
    pitch_deg = math.degrees(pitch_rad)
    if pitch_deg > 90.0:
        pitch_deg = 90.0
    elif pitch_deg < -90.0:
        pitch_deg = -90.0
    return yaw_deg, pitch_deg

@nb.njit(cache=True, fastmath=True)
def yaw_pitch_to_dir_scalar(yaw, pitch):
    cx = math.cos(yaw)
    sx = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    dx = cx * cp
    dy = -sp
    dz = sx * cp
    return dx, dy, dz

@nb.njit(cache=True, fastmath=True, parallel=True)
def yaw_pitch_to_dir_vec(yaw_arr, pitch_arr):
    n = yaw_arr.shape[0]
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    for i in nb.prange(n):
        y = yaw_arr[i]
        p = pitch_arr[i]
        cy = math.cos(y)
        sy = math.sin(y)
        cp = math.cos(p)
        sp = math.sin(p)
        dx[i] = cy * cp
        dy[i] = -sp
        dz[i] = sy * cp
    return dx, dy, dz

@nb.njit(cache=True, fastmath=True)
def wrapped_interval_from_angles(angles: np.ndarray) -> Tuple[float, float]:
    n = angles.size
    if n == 0:
        return 0.0, 0.0

    a = ((angles + math.pi) % (2.0 * math.pi)) - math.pi
    a_sorted = np.sort(a)

    max_gap = -1.0
    k = 0
    for i in range(n - 1):
        gap = a_sorted[i + 1] - a_sorted[i]
        if gap > max_gap:
            max_gap = gap
            k = i
    wrap_gap = (a_sorted[0] + 2.0 * math.pi) - a_sorted[n - 1]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        k = n - 1

    amin = a_sorted[(k + 1) % n]
    amax = amin + (2.0 * math.pi - max_gap)

    return normalize_angle_rad(amin), amax

@nb.njit(cache=True, fastmath=True)
def _interval_center_span(a: float, b: float) -> Tuple[float, float]:
    a_n = normalize_angle_rad(a)
    span = b - a
    center = normalize_angle_rad(a_n + 0.5 * span)
    return center, span

@nb.njit(cache=True, fastmath=True)
def yaw_intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    c1, s1 = _interval_center_span(a0, a1)
    c2, s2 = _interval_center_span(b0, b1)
    d = abs(normalize_angle_rad(c1 - c2))
    return d <= 0.5 * (s1 + s2) + 1e-12


# ------------------------------
# aabb and uv helpers
# ------------------------------

def block_aabb(bx: int, by: int, bz: int) -> AABB:
    return (bx, bx + 1.0, by, by + 1.0, bz, bz + 1.0)

def make_aabb_from_block(pos: BlockPos) -> AABB:
    bx, by, bz = pos
    return block_aabb(bx, by, bz)

@nb.njit(cache=True, fastmath=True)
def aabb_corners_nb(aabb):
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    corners = np.empty((8, 3), dtype=np.float64)
    corners[0] = (xmin, ymin, zmin)
    corners[1] = (xmin, ymin, zmax)
    corners[2] = (xmin, ymax, zmin)
    corners[3] = (xmin, ymax, zmax)
    corners[4] = (xmax, ymin, zmin)
    corners[5] = (xmax, ymin, zmax)
    corners[6] = (xmax, ymax, zmin)
    corners[7] = (xmax, ymax, zmax)
    return corners

@nb.njit(cache=True, fastmath=True)
def angular_bounds_for_aabb_nb(aabb, position):
    px, py, pz = position
    corners = aabb_corners_nb(aabb)
    
    yaw_min = INF
    yaw_max = -INF
    pitch_min = INF
    pitch_max = -INF
    
    for i in range(8):
        cx, cy, cz = corners[i]
        vx = cx - px
        vy = cy - py
        vz = cz - pz
        yaw = np.arctan2(vz, vx)
        hyp = np.hypot(vx, vz)
        pitch = -np.arctan2(vy, hyp)
        
        yaw_min = min(yaw_min, yaw)
        yaw_max = max(yaw_max, yaw)
        pitch_min = min(pitch_min, pitch)
        pitch_max = max(pitch_max, pitch)
        
    return yaw_min, yaw_max, pitch_min, pitch_max

@nb.njit(cache=True, fastmath=True)
def face_and_uv_for_hitpoint_nb(aabb, hx, hy, hz):
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    dxmin = abs(hx - xmin)
    dxmax = abs(hx - xmax)
    dymin = abs(hy - ymin)
    dymax = abs(hy - ymax)
    dzmin = abs(hz - zmin)
    dzmax = abs(hz - zmax)
    dists = np.array([dxmin, dxmax, dymin, dymax, dzmin, dzmax], dtype=np.float64)
    fid = int(np.argmin(dists))
    if fid == 0:
        u = (hz - zmin) / (zmax - zmin)
        v = (hy - ymin) / (ymax - ymin)
    elif fid == 1:
        u = (zmax - hz) / (zmax - zmin)
        v = (hy - ymin) / (ymax - ymin)
    elif fid == 2:
        u = (hx - xmin) / (xmax - xmin)
        v = (zmax - hz) / (zmax - zmin)
    elif fid == 3:
        u = (hx - xmin) / (xmax - xmin)
        v = (hz - zmin) / (zmax - zmin)
    elif fid == 4:
        u = (xmax - hx) / (xmax - xmin)
        v = (hy - ymin) / (ymax - ymin)
    else:
        u = (hx - xmin) / (xmax - xmin)
        v = (hy - ymin) / (ymax - ymin)
    return fid, (max(0.0, min(1.0, u)), max(0.0, min(1.0, v)))


# ------------------------------
# ray helpers
# ------------------------------

@nb.njit(cache=True, fastmath=True)
def _dda_ray_voxels(px, py, pz, ex, ey, ez):
    x = int(math.floor(px))
    y = int(math.floor(py))
    z = int(math.floor(pz))

    tx = int(math.floor(ex))
    ty = int(math.floor(ey))
    tz = int(math.floor(ez))

    dx = ex - px
    dy = ey - py
    dz = ez - pz

    dist2 = dx*dx + dy*dy + dz*dz
    if dist2 <= 1e-18:
        out = np.empty((1, 3), dtype=np.int64)
        out[0, 0] = x
        out[0, 1] = y
        out[0, 2] = z
        return out

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    inv_dx = 1.0/dx if abs(dx) > EPS else INF
    inv_dy = 1.0/dy if abs(dy) > EPS else INF
    inv_dz = 1.0/dz if abs(dz) > EPS else INF

    if dx > 0:
        tmax_x = ((x + 1.0) - px) * inv_dx
    else:
        tmax_x = (px - (x * 1.0)) * (-inv_dx)

    if dy > 0:
        tmax_y = ((y + 1.0) - py) * inv_dy
    else:
        tmax_y = (py - (y * 1.0)) * (-inv_dy)

    if dz > 0:
        tmax_z = ((z + 1.0) - pz) * inv_dz
    else:
        tmax_z = (pz - (z * 1.0)) * (-inv_dz)

    tdelta_x = abs(inv_dx)
    tdelta_y = abs(inv_dy)
    tdelta_z = abs(inv_dz)

    max_est = int(math.ceil(math.sqrt(dist2) * 3.0)) + 16
    buf = np.empty((max_est, 3), dtype=np.int64)
    buf[0, 0] = x
    buf[0, 1] = y
    buf[0, 2] = z
    count = 1

    max_steps = int(math.ceil(math.sqrt(dist2) * 3.0)) + 10000
    steps = 0

    while True:
        if (x == tx) and (y == ty) and (z == tz):
            break

        if (tmax_x > 1.0) and (tmax_y > 1.0) and (tmax_z > 1.0):
            if not ((x == tx) and (y == ty) and (z == tz)):
                txi = tx
                tyi = ty
                tzi = tz
                if count >= buf.shape[0]:
                    new_buf = np.empty((buf.shape[0] * 2, 3), dtype=np.int64)
                    for ii in range(buf.shape[0]):
                        new_buf[ii, 0] = buf[ii, 0]
                        new_buf[ii, 1] = buf[ii, 1]
                        new_buf[ii, 2] = buf[ii, 2]
                    buf = new_buf
                buf[count, 0] = txi
                buf[count, 1] = tyi
                buf[count, 2] = tzi
                count += 1
            break

        if (tmax_x <= tmax_y) and (tmax_x <= tmax_z):
            x += step_x
            tmax_x += tdelta_x
        elif (tmax_y <= tmax_x) and (tmax_y <= tmax_z):
            y += step_y
            tmax_y += tdelta_y
        else:
            z += step_z
            tmax_z += tdelta_z

        if count >= buf.shape[0]:
            new_buf = np.empty((buf.shape[0] * 2, 3), dtype=np.int64)
            for ii in range(buf.shape[0]):
                new_buf[ii, 0] = buf[ii, 0]
                new_buf[ii, 1] = buf[ii, 1]
                new_buf[ii, 2] = buf[ii, 2]
            buf = new_buf

        buf[count, 0] = x
        buf[count, 1] = y
        buf[count, 2] = z
        count += 1

        steps += 1
        if steps > max_steps:
            if count >= buf.shape[0]:
                new_buf = np.empty((buf.shape[0] * 2, 3), dtype=np.int64)
                for ii in range(buf.shape[0]):
                    new_buf[ii, 0] = buf[ii, 0]
                    new_buf[ii, 1] = buf[ii, 1]
                    new_buf[ii, 2] = buf[ii, 2]
                buf = new_buf
            buf[count, 0] = tx
            buf[count, 1] = ty
            buf[count, 2] = tz
            count += 1
            break

    return buf[:count].copy()

@nb.njit(cache=True, fastmath=True)
def _expand_neighbors_into_dict_njit(voxels: np.ndarray, radius: int, d):
    n = voxels.shape[0]
    for i in range(n):
        x0 = voxels[i, 0]
        y0 = voxels[i, 1]
        z0 = voxels[i, 2]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    key = (x0 + dx, y0 + dy, z0 + dz)
                    if key not in d:
                        d[key] = 1


def _expand_neighbors(voxels: np.ndarray, radius: int = 1) -> np.ndarray:
    voxels = np.asarray(voxels, dtype=np.int64)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError("voxels must be an (n,3) int64 array")

    key_type = types.UniTuple(types.int64, 3)
    d = Dict.empty(key_type=key_type, value_type=types.int64)

    _expand_neighbors_into_dict_njit(voxels, int(radius), d)

    keys = list(d.keys())
    out = np.empty((len(keys), 3), dtype=np.int64)
    for i, k in enumerate(keys):
        out[i, 0] = k[0]
        out[i, 1] = k[1]
        out[i, 2] = k[2]

    if out.shape[0] > 1:
        order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
        out = out[order]

    return [(int(i[0]), int(i[1]), int(i[2])) for i in out]

@nb.njit(cache=True, parallel=True, fastmath=True)
def ray_aabb_intersection_vec(px, py, pz,
                                    dx, dy, dz,
                                    xmin, xmax, ymin, ymax, zmin, zmax):
    n = dx.shape[0]
    tmin_out = np.empty(n, dtype=np.float64)
    tmax_out = np.empty(n, dtype=np.float64)

    for i in nb.prange(n):
        di = dx[i]
        if di > EPS or di < -EPS:
            inv = 1.0 / di
            t1 = (xmin - px) * inv
            t2 = (xmax - px) * inv
            if t1 < t2:
                tx_min = t1; tx_max = t2
            else:
                tx_min = t2; tx_max = t1
            miss_x = False
        else:
            tx_min = -INF; tx_max = INF
            miss_x = (px < xmin) or (px > xmax)

        di = dy[i]
        if di > EPS or di < -EPS:
            inv = 1.0 / di
            t1 = (ymin - py) * inv
            t2 = (ymax - py) * inv
            if t1 < t2:
                ty_min = t1; ty_max = t2
            else:
                ty_min = t2; ty_max = t1
            miss_y = False
        else:
            ty_min = -INF; ty_max = INF
            miss_y = (py < ymin) or (py > ymax)

        di = dz[i]
        if di > EPS or di < -EPS:
            inv = 1.0 / di
            t1 = (zmin - pz) * inv
            t2 = (zmax - pz) * inv
            if t1 < t2:
                tz_min = t1; tz_max = t2
            else:
                tz_min = t2; tz_max = t1
            miss_z = False
        else:
            tz_min = -INF; tz_max = INF
            miss_z = (pz < zmin) or (pz > zmax)

        tmin_i = tx_min
        if ty_min > tmin_i:
            tmin_i = ty_min
        if tz_min > tmin_i:
            tmin_i = tz_min

        tmax_i = tx_max
        if ty_max < tmax_i:
            tmax_i = ty_max
        if tz_max < tmax_i:
            tmax_i = tz_max

        miss = (tmin_i > tmax_i) or miss_x or miss_y or miss_z or (tmax_i < 0.0)

        if miss:
            tmin_out[i] = np.nan
            tmax_out[i] = np.nan
        else:
            tmin_out[i] = tmin_i
            tmax_out[i] = tmax_i

    return tmin_out, tmax_out

@nb.njit(cache=True, inline='always', fastmath=True)
def _ray_aabb_intersect_single(px, py, pz, dx, dy, dz,
                               xmin, xmax, ymin, ymax, zmin, zmax) -> Tuple[float, float]:
    if abs(dx) > EPS:
        tx1 = (xmin - px) / dx
        tx2 = (xmax - px) / dx
        if tx1 <= tx2:
            tmin_x = tx1
            tmax_x = tx2
        else:
            tmin_x = tx2
            tmax_x = tx1
    else:
        if px < xmin - EPS or px > xmax + EPS:
            return (np.nan, np.nan)
        tmin_x = -INF
        tmax_x = INF

    if abs(dy) > EPS:
        ty1 = (ymin - py) / dy
        ty2 = (ymax - py) / dy
        if ty1 <= ty2:
            tmin_y = ty1
            tmax_y = ty2
        else:
            tmin_y = ty2
            tmax_y = ty1
    else:
        if py < ymin - EPS or py > ymax + EPS:
            return (np.nan, np.nan)
        tmin_y = -INF
        tmax_y = INF

    if abs(dz) > EPS:
        tz1 = (zmin - pz) / dz
        tz2 = (zmax - pz) / dz
        if tz1 <= tz2:
            tmin_z = tz1
            tmax_z = tz2
        else:
            tmin_z = tz2
            tmax_z = tz1
    else:
        if pz < zmin - EPS or pz > zmax + EPS:
            return (np.nan, np.nan)
        tmin_z = -INF
        tmax_z = INF

    tmin = tmin_x
    if tmin_y > tmin:
        tmin = tmin_y
    if tmin_z > tmin:
        tmin = tmin_z

    tmax = tmax_x
    if tmax_y < tmax:
        tmax = tmax_y
    if tmax_z < tmax:
        tmax = tmax_z

    if tmax < tmin:
        return (np.nan, np.nan)
    return (tmin, tmax)


# ------------------------------
# solid angle helpers
# ------------------------------

@nb.njit(cache=True, fastmath=True)
def tri_solid_angle(r0x, r0y, r0z,
                                 r1x, r1y, r1z,
                                 r2x, r2y, r2z):
    cx = r1y * r2z - r1z * r2y
    cy = r1z * r2x - r1x * r2z
    cz = r1x * r2y - r1y * r2x

    det = r0x * cx + r0y * cy + r0z * cz

    d01 = r0x * r1x + r0y * r1y + r0z * r1z
    d12 = r1x * r2x + r1y * r2y + r1z * r2z
    d20 = r2x * r0x + r2y * r0y + r2z * r0z

    denom = 1.0 + d01 + d12 + d20

    return 2.0 * math.atan2(det, denom)

@nb.njit(cache=True, fastmath=True)
def quad_solid_angle(r00x, r00y, r00z,
                                  r01x, r01y, r01z,
                                  r11x, r11y, r11z,
                                  r10x, r10y, r10z):
    a = tri_solid_angle(r00x, r00y, r00z,
                                     r01x, r01y, r01z,
                                     r11x, r11y, r11z)
    b = tri_solid_angle(r00x, r00y, r00z,
                                     r11x, r11y, r11z,
                                     r10x, r10y, r10z)
    return a + b

@nb.njit(cache=True, parallel=True, fastmath=True)
def compute_sample_solid_angles(yaw_min, yaw_step, pitch_min, pitch_step,
                                      yaw_bins, pitch_bins):
    total = yaw_bins * pitch_bins
    out = np.empty(total, dtype=np.float64)

    for idx in nb.prange(total):
        ip = idx // yaw_bins
        iy = idx % yaw_bins

        p0 = pitch_min + ip * pitch_step
        p1 = p0 + pitch_step
        y0 = yaw_min + iy * yaw_step
        y1 = y0 + yaw_step

        r00x, r00y, r00z = yaw_pitch_to_dir_scalar(y0, p0)
        r10x, r10y, r10z = yaw_pitch_to_dir_scalar(y1, p0)
        r11x, r11y, r11z = yaw_pitch_to_dir_scalar(y1, p1)
        r01x, r01y, r01z = yaw_pitch_to_dir_scalar(y0, p1)

        norm = math.sqrt(r00x * r00x + r00y * r00y + r00z * r00z)
        if norm > 0.0:
            r00x /= norm; r00y /= norm; r00z /= norm

        norm = math.sqrt(r10x * r10x + r10y * r10y + r10z * r10z)
        if norm > 0.0:
            r10x /= norm; r10y /= norm; r10z /= norm

        norm = math.sqrt(r11x * r11x + r11y * r11y + r11z * r11z)
        if norm > 0.0:
            r11x /= norm; r11y /= norm; r11z /= norm

        norm = math.sqrt(r01x * r01x + r01y * r01y + r01z * r01z)
        if norm > 0.0:
            r01x /= norm; r01y /= norm; r01z /= norm

        omega = quad_solid_angle(r00x, r00y, r00z,
                                              r10x, r10y, r10z,
                                              r11x, r11y, r11z,
                                              r01x, r01y, r01z)
        out[idx] = abs(omega)

    return out


# ------------------------------
# geometry cache
# ------------------------------

@nb.njit(cache=True, fastmath=True)
def polygon_sphere_bounds_numba(verts: np.ndarray, px: float, py: float, pz: float) -> Tuple[float, float, float, float, float]:
    n = verts.shape[0]
    yaws = np.empty(n, dtype=np.float64)
    pitches = np.empty(n, dtype=np.float64)

    dmin = INF
    for i in range(n):
        vx = verts[i, 0] - px
        vy = verts[i, 1] - py
        vz = verts[i, 2] - pz
        yaw = math.atan2(vz, vx)
        hyp = math.hypot(vx, vz)
        pitch = -math.atan2(vy, hyp)
        yaws[i] = normalize_angle_rad_nb(yaw)
        pitches[i] = pitch
        dist = math.sqrt(vx*vx + vy*vy + vz*vz)
        if dist < dmin:
            dmin = dist

    yaw_min = INF
    yaw_max = -INF
    sx = 0.0
    sy = 0.0
    for i in range(n):
        sx += math.cos(yaws[i])
        sy += math.sin(yaws[i])
    if n > 0:
        center = math.atan2(sy, sx)
    else:
        center = 0.0
    for i in range(n):
        a = normalize_angle_rad_nb(yaws[i] - center)
        if a < yaw_min:
            yaw_min = a
        if a > yaw_max:
            yaw_max = a
    yaw_min = normalize_angle_rad_nb(yaw_min + center)
    yaw_max = normalize_angle_rad_nb(yaw_max + center)
    pitch_min = INF
    pitch_max = -INF
    for i in range(n):
        if pitches[i] < pitch_min:
            pitch_min = pitches[i]
        if pitches[i] > pitch_max:
            pitch_max = pitches[i]
    return yaw_min, yaw_max, pitch_min, pitch_max, dmin

class BlockGeometryCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, FrozenSet[Tuple[str, Any]]], List[Dict[str, Any]]] = {}

    def _serialize_meta_val(self, v: Any):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), self._serialize_meta_val(vv)) for k, vv in v.items()))
        return v

    def _meta_key(self, block_type: str, meta: Optional[Mapping[str, Any]]) -> Tuple[str, FrozenSet[Tuple[str, Any]]]:
        if meta is None:
            meta = {}
        items = tuple(sorted((str(k), self._serialize_meta_val(v)) for k, v in meta.items()))
        return (block_type, frozenset(items))

    @staticmethod
    def _quad(v0, v1, v2, v3):
        return np.array([v0, v1, v2, v3], dtype=np.float64)

    @staticmethod
    def _box_faces(xmin, xmax, ymin, ymax, zmin, zmax) -> List[Dict[str, Any]]:
        q = BlockGeometryCache._quad
        return [
            {'verts': q((xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)), 'opaque': True},
            {'verts': q((xmin, ymin, zmax), (xmin, ymax, zmax), (xmin, ymax, zmin), (xmin, ymin, zmin)), 'opaque': True},
            {'verts': q((xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)), 'opaque': True},
            {'verts': q((xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymin, zmin), (xmin, ymin, zmin)), 'opaque': True},
            {'verts': q((xmin, ymin, zmax), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymin, zmax)), 'opaque': True},
            {'verts': q((xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin), (xmin, ymin, zmin)), 'opaque': True},
        ]

    def get_polygons_for_block(self, block_id: str, meta: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        key = self._meta_key(block_id, meta)
        if key in self._cache:
            return self._cache[key]

        base_low = block_id.lower().strip()

        if base_low in ("minecraft:air", "minecraft:water"):
            polys: List[Dict[str, Any]] = []
            self._cache[key] = polys
            return polys

        if base_low.endswith("_slab") or ":slab" in base_low:
            short_type = "slab"
        elif base_low.endswith("_stairs") or "stairs" in base_low or ":stair" in base_low:
            short_type = "stair"
        elif base_low.endswith("_pane") or "pane" in base_low:
            short_type = "pane"
        else:
            short_type = "full_block"

        if short_type == "pane":
            polys = self._build_pane(meta or {})
        elif short_type == "slab":
            polys = self._build_slab(meta or {})
        elif short_type == "stair":
            polys = self._build_stair(meta or {})
        else:
            polys = self._build_full_block()

        self._cache[key] = polys
        return polys

    def _build_full_block(self) -> List[Dict[str, Any]]:
        return self._box_faces(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    def _build_slab(self, meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
        half = str(meta.get("half", "bottom")).lower()
        if half == "top":
            ymin, ymax = 0.5, 1.0
        else:
            ymin, ymax = 0.0, 0.5
        return self._box_faces(0.0, 1.0, ymin, ymax, 0.0, 1.0)

    def _build_pane(self, meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
        conns = set(meta.get("connections", ()))
        t = float(meta.get("thickness", 1.0 / 16.0))
        half = 0.5 * t
        ymin, ymax = 0.0, 1.0
        xmin = 0.5 - 0.125
        xmax = 0.5 + 0.125
        zmin = 0.5 - 0.125
        zmax = 0.5 + 0.125
        if "north" in conns:
            zmin = 0.0
        if "south" in conns:
            zmax = 1.0
        if "west" in conns:
            xmin = 0.0
        if "east" in conns:
            xmax = 1.0

        faces: List[Dict[str, Any]] = []
        dx = xmax - xmin
        dz = zmax - zmin
        if dx >= dz:
            z0 = 0.5 - half
            z1 = 0.5 + half
            v0 = (xmin, ymin, z0)
            v1 = (xmax, ymin, z0)
            v2 = (xmax, ymax, z0)
            v3 = (xmin, ymax, z0)
            v4 = (xmin, ymin, z1)
            v5 = (xmax, ymin, z1)
            v6 = (xmax, ymax, z1)
            v7 = (xmin, ymax, z1)
        else:
            x0 = 0.5 - half
            x1 = 0.5 + half
            v0 = (x0, ymin, zmin)
            v1 = (x1, ymin, zmin)
            v2 = (x1, ymax, zmin)
            v3 = (x0, ymax, zmin)
            v4 = (x0, ymin, zmax)
            v5 = (x1, ymin, zmax)
            v6 = (x1, ymax, zmax)
            v7 = (x0, ymax, zmax)

        q = self._quad
        faces.extend([
            {'verts': q(v0, v1, v2, v3), 'opaque': True},
            {'verts': q(v4, v5, v6, v7), 'opaque': True},
            {'verts': q(v0, v4, v7, v3), 'opaque': True},
            {'verts': q(v1, v5, v6, v2), 'opaque': True},
        ])
        return faces

    def _build_stair(self, meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
        facing = str(meta.get('facing', 'north')).lower()
        half = str(meta.get('half', 'bottom')).lower()
        shape = str(meta.get('shape', 'straight')).lower()

        X0, X1, X2 = 0.0, 0.5, 1.0
        Z0, Z1, Z2 = 0.0, 0.5, 1.0

        def add_box(b, out):
            (xmin, xmax, ymin, ymax, zmin, zmax) = b
            out.extend(self._box_faces(xmin, xmax, ymin, ymax, zmin, zmax))

        def step_footprint_boxes(facing: str, shape: str) -> List[Tuple[float, float, float, float]]:
            north = (X0, X2, Z0, Z1)
            south = (X0, X2, Z1, Z2)
            west = (X0, X1, Z0, Z2)
            east = (X1, X2, Z0, Z2)

            if shape == 'straight':
                return {'north': [north], 'south': [south], 'west': [west], 'east': [east]}[facing]

            if shape.startswith('inner_'):
                side = 'left' if shape.endswith('left') else 'right'
                if facing == 'north':
                    return [north, (west if side == 'left' else east)]
                if facing == 'south':
                    return [south, (east if side == 'left' else west)]
                if facing == 'west':
                    return [west, (south if side == 'left' else north)]
                return [east, (north if side == 'left' else south)]

            if shape.startswith('outer_'):
                side = 'left' if shape.endswith('left') else 'right'
                if facing == 'north':
                    return [(X0, X1, Z0, Z1)] if side == 'right' else [(X1, X2, Z0, Z1)]
                if facing == 'south':
                    return [(X1, X2, Z1, Z2)] if side == 'left' else [(X0, X1, Z1, Z2)]
                if facing == 'west':
                    return [(X0, X1, Z1, Z2)] if side == 'left' else [(X0, X1, Z0, Z1)]
                return [(X1, X2, Z0, Z1)] if side == 'left' else [(X1, X2, Z1, Z2)]

            return {'north': [north], 'south': [south], 'west': [west], 'east': [east]}[facing]

        polys: List[Dict[str, Any]] = []
        if half == 'bottom':
            add_box((0.0, 1.0, 0.0, 0.5, 0.0, 1.0), polys)
            for (xmin, xmax, zmin, zmax) in step_footprint_boxes(facing, shape):
                add_box((xmin, xmax, 0.5, 1.0, zmin, zmax), polys)
        else:
            add_box((0.0, 1.0, 0.5, 1.0, 0.0, 1.0), polys)
            for (xmin, xmax, zmin, zmax) in step_footprint_boxes(facing, shape):
                add_box((xmin, xmax, 0.0, 0.5, zmin, zmax), polys)
        return polys

    def world_polygons(self, block_pos: BlockPos, block_type: str, meta: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        bx, by, bz = block_pos
        base = self.get_polygons_for_block(block_type, meta)
        out = []
        base_off = np.array([bx, by, bz], dtype=np.float64)
        for p in base:
            out.append({'verts': p['verts'] + base_off, 'opaque': p.get('opaque', True),
                        'block': (block_pos, block_type, meta)})
        return out

    def polygon_sphere_bounds(self, verts_world, position):
        verts = np.ascontiguousarray(np.asarray(verts_world, dtype=np.float64), dtype=np.float64)
        px, py, pz = map(float, position)
        return polygon_sphere_bounds_numba(verts, px, py, pz)
    
    def analytic_refine_depth_in_target_cone(self,
        adb: HighResADB,
        position: Vec3,
        target_aabb: AABB,
        blocks: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]],
        max_depth: float = 200.0,
        yaw_margin_deg: float = 0.4,
        pitch_margin_deg: float = 0.4,
        pos_to_occluder_id: Optional[Dict[BlockPos, int]] = None,
    ):
        target_aabb = np.array([target_aabb[0], target_aabb[1], target_aabb[2], target_aabb[3], target_aabb[4], target_aabb[5]], dtype=np.float64)
        position = np.array([position[0], position[1], position[2]], dtype=np.float64)
        tymin, tymax, tpmin, tpmax = angular_bounds_for_aabb_nb(target_aabb, position)
        yaw_margin = math.radians(yaw_margin_deg)
        pitch_margin = math.radians(pitch_margin_deg)
        tymin_m = normalize_angle_rad(tymin - yaw_margin)
        tymax_m = tymax + yaw_margin
        tpmin_m = tpmin - pitch_margin
        tpmax_m = tpmax + pitch_margin

        cand_tri_arrays = []
        for (pos, base, short_type, meta) in blocks:
            polys = self.world_polygons(pos, base, meta)
            for p in polys:
                verts = p['verts']
                yaw_min, yaw_max, pitch_min, pitch_max, dmin = self.polygon_sphere_bounds(verts, position)
                if dmin > max_depth:
                    continue
                if (pitch_max < tpmin_m) or (pitch_min > tpmax_m):
                    continue
                if not yaw_intervals_overlap(yaw_min, yaw_max, tymin_m, tymax_m):
                    continue
                tri_list = _triangulate_convex_polygon(verts)
                if not tri_list:
                    continue
                tris = np.stack(tri_list, axis=0).astype(np.float64)
                cand_tri_arrays.append((tris, p.get('block', (None,))[0], p))

        if not cand_tri_arrays:
            return

        iy_min_f = (normalize_angle_rad(tymin_m) - adb.yaw_min) / (adb.yaw_max - adb.yaw_min) * adb.yaw_bins
        iy_max_f = (normalize_angle_rad(tymax_m) - adb.yaw_min) / (adb.yaw_max - adb.yaw_min) * adb.yaw_bins
        if (tymax_m - tymin_m) > math.pi * 1.5:
            iy_ranges = [(0, adb.yaw_bins - 1)]
        else:
            a = iy_min_f
            b = iy_max_f
            if b >= a:
                iy_ranges = [(max(0, int(math.floor(a))), min(adb.yaw_bins - 1, int(math.ceil(b))))]
            else:
                iy_ranges = [
                    (0, min(adb.yaw_bins - 1, int(math.ceil(b)))),
                    (max(0, int(math.floor(a))), adb.yaw_bins - 1),
                ]

        ip_min_f = (tpmin_m - adb.pitch_min) / (adb.pitch_max - adb.pitch_min) * adb.pitch_bins
        ip_max_f = (tpmax_m - adb.pitch_min) / (adb.pitch_max - adb.pitch_min) * adb.pitch_bins
        ip_min = max(0, int(math.floor(ip_min_f)))
        ip_max = min(adb.pitch_bins - 1, int(math.ceil(ip_max_f)))

        idx_list: List[np.ndarray] = []
        for (iy0, iy1) in iy_ranges:
            iy_range = np.arange(iy0, iy1 + 1, dtype=np.int32)
            ip_range = np.arange(ip_min, ip_max + 1, dtype=np.int32)
            IY, IP = np.meshgrid(iy_range, ip_range, indexing='xy')
            idx_list.append((IP * adb.yaw_bins + IY).ravel())
        if not idx_list:
            return
        idxs = np.unique(np.concatenate(idx_list, axis=0))
        if idxs.size == 0:
            return

        dx = adb.dx[idxs].astype(np.float64)
        dy = adb.dy[idxs].astype(np.float64)
        dz = adb.dz[idxs].astype(np.float64)
        px, py, pz = map(float, position)

        for tris, block_pos, p in cand_tri_arrays:
            tmin_poly = ray_triangles_min_t_nb_parallel(px, py, pz, dx, dy, dz, tris)
            mask_hit = (tmin_poly < np.inf) & (tmin_poly >= 0.0) & (tmin_poly <= max_depth)
            if not np.any(mask_hit):
                continue
            better = mask_hit & (tmin_poly < adb.depth[idxs])
            if not np.any(better):
                continue
            adb.depth[idxs[better]] = tmin_poly[better]
            if pos_to_occluder_id is not None and block_pos is not None:
                oid = pos_to_occluder_id.get(block_pos, -1)
                if oid >= 0:
                    adb.top_occluder_idx[idxs[better]] = int(oid)


# ------------------------------
# raster grid (ADB)
# ------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _rasterize_occluders_nb(
    px: float, py: float, pz: float,
    dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
    aabbs: np.ndarray,
    occluder_ids: np.ndarray,
    depth: np.ndarray,
    top_idx: np.ndarray,
    max_depth: float
) -> None:
    nrays = dx.shape[0]
    na = aabbs.shape[0]

    for i in nb.prange(nrays):
        di = dx[i]; dj = dy[i]; dk = dz[i]
        best_t = depth[i]
        best_oid = top_idx[i]
        for j in range(na):
            xmin = aabbs[j, 0]; xmax = aabbs[j, 1]
            ymin = aabbs[j, 2]; ymax = aabbs[j, 3]
            zmin = aabbs[j, 4]; zmax = aabbs[j, 5]

            tmin, tmax = _ray_aabb_intersect_single(px, py, pz, di, dj, dk,
                                                    xmin, xmax, ymin, ymax, zmin, zmax)
            if math.isnan(tmin):
                continue
            t = tmin if tmin >= 0.0 else tmax
            if math.isnan(t) or t < 0.0:
                continue
            if t > max_depth:
                continue
            if t < best_t:
                best_t = t
                best_oid = occluder_ids[j]

        depth[i] = best_t
        top_idx[i] = best_oid

class HighResADB:
    def __init__(self, yaw_bins: int = 512, pitch_bins: int = 256,
                 yaw_center: float = 0.0, yaw_span: float = 2.0 * math.pi,
                 pitch_min: float = -math.pi / 2, pitch_max: float = math.pi / 2):
        self.yaw_bins = yaw_bins
        self.pitch_bins = pitch_bins
        self.yaw_min = yaw_center - yaw_span / 2.0
        self.yaw_max = yaw_center + yaw_span / 2.0
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

        self.yaw_step = (self.yaw_max - self.yaw_min) / yaw_bins
        self.pitch_step = (self.pitch_max - self.pitch_min) / pitch_bins

        self.yaw_samples = self.yaw_min + (np.arange(yaw_bins) + 0.5) * self.yaw_step
        self.pitch_samples = self.pitch_min + (np.arange(pitch_bins) + 0.5) * self.pitch_step

        YA, PA = np.meshgrid(self.yaw_samples, self.pitch_samples, indexing='xy')
        YAf = YA.ravel().astype(np.float64)
        PAf = PA.ravel().astype(np.float64)
        self.dx, self.dy, self.dz = yaw_pitch_to_dir_vec(YAf, PAf)
        self.dx = np.ascontiguousarray(self.dx, dtype=np.float64)
        self.dy = np.ascontiguousarray(self.dy, dtype=np.float64)
        self.dz = np.ascontiguousarray(self.dz, dtype=np.float64)
        self.N = self.dx.size

        self.depth = np.full(self.N, np.inf, dtype=np.float64)
        self.top_occluder_idx = np.full(self.N, -1, dtype=np.int32)

        self.sample_solid_angle = compute_sample_solid_angles(
            self.yaw_min, 
            self.yaw_step, 
            self.pitch_min, 
            self.pitch_step,
            int(self.yaw_bins), 
            int(self.pitch_bins))

    def reset_depth(self) -> None:
        self.depth.fill(np.inf)
        self.top_occluder_idx.fill(-1)

    def idx_from_iy_ip(self, iy: int, ip: int) -> int:
        return ip * self.yaw_bins + iy

    def iy_ip_from_idx(self, idx: int) -> Tuple[int, int]:
        ip = idx // self.yaw_bins
        iy = idx % self.yaw_bins
        return iy, ip

    def rasterize_occluders(self, occluder_aabbs: List[AABB], position: Vec3,
                            occluder_ids: Optional[List[int]] = None,
                            max_depth: float = 200.0) -> None:
        px, py, pz = map(float, position)
        dx = np.ascontiguousarray(self.dx, dtype=np.float64)
        dy = np.ascontiguousarray(self.dy, dtype=np.float64)
        dz = np.ascontiguousarray(self.dz, dtype=np.float64)

        Na = len(occluder_aabbs)
        if Na == 0:
            return


        if len(occluder_aabbs) < BVH_THRESHOLD:
            aabbs_arr = np.empty((Na, 6), dtype=np.float64)
            for i, aabb in enumerate(occluder_aabbs):
                xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in aabb)
                aabbs_arr[i, 0] = xmin
                aabbs_arr[i, 1] = xmax
                aabbs_arr[i, 2] = ymin
                aabbs_arr[i, 3] = ymax
                aabbs_arr[i, 4] = zmin
                aabbs_arr[i, 5] = zmax

            if occluder_ids is None:
                oc_ids = np.full((Na,), -1, dtype=np.int32)
            else:
                if len(occluder_ids) != Na:
                    raise ValueError("occluder_ids must be same length as occluder_aabbs")
                oc_ids = np.array(occluder_ids, dtype=np.int32)

            depth_arr = np.ascontiguousarray(self.depth, dtype=np.float64)
            top_idx_arr = np.ascontiguousarray(self.top_occluder_idx, dtype=np.int32)

            _rasterize_occluders_nb(px, py, pz, dx, dy, dz, aabbs_arr, oc_ids, depth_arr, top_idx_arr, float(max_depth))

            self.depth = depth_arr
            self.top_occluder_idx = top_idx_arr
        
        else:
            prim_min = np.empty((Na, 3), dtype=np.float64)
            prim_max = np.empty((Na, 3), dtype=np.float64)
            prim_id = np.empty((Na,), dtype=np.int32)
            for i, aabb in enumerate(occluder_aabbs):
                xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in aabb)
                prim_min[i, 0] = xmin
                prim_min[i, 1] = ymin
                prim_min[i, 2] = zmin
                prim_max[i, 0] = xmax
                prim_max[i, 1] = ymax
                prim_max[i, 2] = zmax
                prim_id[i] = -1 if occluder_ids is None else int(occluder_ids[i])

            rebuild = True
            if hasattr(self, "_bvh_prim_count") and self._bvh_prim_count == Na:
                if hasattr(self, "_bvh_prim_min") and np.array_equal(self._bvh_prim_min, prim_min) == False:
                    try:
                        pm = np.ascontiguousarray(prim_min, dtype=np.float64)
                        pM = np.ascontiguousarray(prim_max, dtype=np.float64)
                        bvh_refit_numba(self.bvh_node_min, self.bvh_node_max,
                                        self.bvh_left, self.bvh_right,
                                        self.bvh_first, self.bvh_count, self.bvh_leaf_indices,
                                        pm, pM, self.bvh_postorder)
                        rebuild = False
                    except Exception:
                        rebuild = True
                else:
                    rebuild = False

            if rebuild:
                (node_min, node_max, node_left, node_right,
                node_first, node_count, leaf_prim_indices, postorder) = build_bvh(prim_min, prim_max, prim_id, max_leaf_size=4)

                self.bvh_node_min = node_min
                self.bvh_node_max = node_max
                self.bvh_left = node_left
                self.bvh_right = node_right
                self.bvh_first = node_first
                self.bvh_count = node_count
                self.bvh_leaf_indices = leaf_prim_indices
                self.bvh_postorder = postorder
                self._bvh_prim_count = Na

            self._bvh_prim_min = prim_min.copy()
            self._bvh_prim_max = prim_max.copy()

            depth_arr = np.ascontiguousarray(self.depth, dtype=np.float64)
            top_idx_arr = np.ascontiguousarray(self.top_occluder_idx, dtype=np.int32)

            rasterize_with_bvh_nb(px, py, pz,
                                dx, dy, dz,
                                self.bvh_node_min, self.bvh_node_max,
                                self.bvh_left, self.bvh_right,
                                self.bvh_first, self.bvh_count,
                                self.bvh_leaf_indices,
                                prim_min, prim_max, prim_id,
                                depth_arr, top_idx_arr,
                                float(max_depth))

            self.depth = depth_arr
            self.top_occluder_idx = top_idx_arr

    def visible_samples_for_aabb(self, target_aabb: AABB, position: Vec3) -> Dict[str, Any]:
        px, py, pz = map(float, position)
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in target_aabb)
        tmin_targ, tmax_targ = ray_aabb_intersection_vec(
            px, py, pz,
            self.dx, self.dy, self.dz,
            xmin, xmax, ymin, ymax, zmin, zmax
        )
        ttarget = np.where(~np.isnan(tmin_targ), np.where(tmin_targ >= 0.0, tmin_targ, tmax_targ), np.nan)
        visible_mask = (~np.isnan(ttarget)) & (ttarget <= self.depth - EPS)
        if not np.any(visible_mask):
            return {
                'visible_mask': visible_mask,
                'ttarget': ttarget,
                'hit_points': np.empty((0, 3), dtype=np.float64),
                'face_ids': [],
                'uvs': np.empty((0, 2), dtype=np.float64),
                'yaw_bounds': None,
                'pitch_bounds': None,
                'solid_angle': 0.0,
            }

        idxs = np.nonzero(visible_mask)[0]
        tvis = ttarget[idxs]
        dxv = self.dx[idxs]
        dyv = self.dy[idxs]
        dzv = self.dz[idxs]
        hits_x = px + dxv * tvis
        hits_y = py + dyv * tvis
        hits_z = pz + dzv * tvis
        hit_points = np.stack((hits_x, hits_y, hits_z), axis=1)

        face_ids: List[int] = []
        uvs_list: List[Tuple[float, float]] = []
        for (hx, hy, hz) in hit_points:
            fid, (u, v) = face_and_uv_for_hitpoint_nb(target_aabb, float(hx), float(hy), float(hz))
            face_ids.append(fid)
            uvs_list.append((u, v))

        iy = idxs % self.yaw_bins
        ip = idxs // self.yaw_bins
        yaw_centers = self.yaw_min + (iy + 0.5) * self.yaw_step
        pitch_centers = self.pitch_min + (ip + 0.5) * self.pitch_step
        yaw_min, yaw_max = wrapped_interval_from_angles(yaw_centers)
        pitch_min = float(np.min(pitch_centers))
        pitch_max = float(np.max(pitch_centers))

        solid_angle = float(np.sum(self.sample_solid_angle[idxs]))

        return {
            'visible_mask': visible_mask,
            'ttarget': ttarget,
            'ttarget_visible': tvis,
            'hit_points': hit_points,
            'face_ids': np.asarray(face_ids, dtype=np.int32),
            'uvs': np.asarray(uvs_list, dtype=np.float64),
            'yaw_bounds': (yaw_min, yaw_max),
            'pitch_bounds': (pitch_min, pitch_max),
            'solid_angle': solid_angle,
        }


# ------------------------------
# polygon helpers
# ------------------------------

def _triangulate_convex_polygon(verts: np.ndarray) -> List[np.ndarray]:
    tris: List[np.ndarray] = []
    if verts.shape[0] < 3:
        return tris
    for i in range(1, verts.shape[0] - 1):
        tris.append(np.stack([verts[0], verts[i], verts[i + 1]], axis=0))
    return tris

@nb.njit(cache=True, fastmath=True)
def _ray_triangle_t_single(px, py, pz,
                           dx, dy, dz,
                           v0, v1, v2):
    e1x = v1[0] - v0[0]
    e1y = v1[1] - v0[1]
    e1z = v1[2] - v0[2]
    e2x = v2[0] - v0[0]
    e2y = v2[1] - v0[1]
    e2z = v2[2] - v0[2]

    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) <= EPS:
        return math.inf
    inv_det = 1.0 / det

    tx = px - v0[0]
    ty = py - v0[1]
    tz = pz - v0[2]

    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < -EPS or u > 1.0 + EPS:
        return math.inf

    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x

    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < -EPS or (u + v) > 1.0 + EPS:
        return math.inf

    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    if t >= 0.0:
        return t
    return math.inf

@nb.njit(cache=True, parallel=False, fastmath=True)
def ray_triangles_min_t_nb(px, py, pz,
                           dx_arr, dy_arr, dz_arr,
                           tris):
    n_rays = dx_arr.shape[0]
    n_tris = tris.shape[0]
    tmin = np.full(n_rays, np.inf, dtype=np.float64)

    for i in range(n_rays):
        dx = dx_arr[i]; dy = dy_arr[i]; dz = dz_arr[i]
        best = math.inf
        for j in range(n_tris):
            v0 = tris[j, 0]
            v1 = tris[j, 1]
            v2 = tris[j, 2]
            t = _ray_triangle_t_single(px, py, pz, dx, dy, dz, v0, v1, v2)
            if t < best:
                best = t
        tmin[i] = best
    return tmin

@nb.njit(cache=True, parallel=True, fastmath=True)
def ray_triangles_min_t_nb_parallel(px, py, pz,
                                    dx_arr, dy_arr, dz_arr,
                                    tris):
    n_rays = dx_arr.shape[0]
    tmin = np.full(n_rays, np.inf, dtype=np.float64)
    for i in nb.prange(n_rays):
        dx = dx_arr[i]; dy = dy_arr[i]; dz = dz_arr[i]
        best = math.inf
        for j in range(tris.shape[0]):
            v0 = tris[j, 0]
            v1 = tris[j, 1]
            v2 = tris[j, 2]
            t = _ray_triangle_t_single(px, py, pz, dx, dy, dz, v0, v1, v2)
            if t < best:
                best = t
        tmin[i] = best
    return tmin


# ------------------------------
# visibility clustering and aim
# ------------------------------

def _find_connected_components(visible_mask_2d: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = visible_mask_2d.shape
    labels = np.zeros_like(visible_mask_2d, dtype=np.int32)
    current_label = 0
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(H):
        for c in range(W):
            if not visible_mask_2d[r, c] or labels[r, c] != 0:
                continue
            current_label += 1
            stack = [(r, c)]
            labels[r, c] = current_label
            while stack:
                rr, cc = stack.pop()
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= H:
                        continue
                    nc_wrapped = (nc + W) % W
                    if not visible_mask_2d[nr, nc_wrapped] or labels[nr, nc_wrapped] != 0:
                        continue
                    labels[nr, nc_wrapped] = current_label
                    stack.append((nr, nc_wrapped))
    return labels, current_label

def compute_visible_face_centroid_aim_with_clustering(
    adb: HighResADB, target_aabb: AABB, position: Vec3
) -> Optional[Dict[str, Any]]:
    res = adb.visible_samples_for_aabb(target_aabb, position)
    visible_mask_flat = res['visible_mask']
    if not np.any(visible_mask_flat):
        return None

    idxs = np.nonzero(visible_mask_flat)[0]
    hit_points = res['hit_points']
    face_ids = res['face_ids']
    uvs = res['uvs']
    weights = adb.sample_solid_angle[idxs]

    H = adb.pitch_bins
    W = adb.yaw_bins
    full_mask_2d = visible_mask_flat.reshape((H, W)).copy()
    labels_2d, n_components = _find_connected_components(full_mask_2d)
    if n_components == 0:
        return None

    ip = (idxs // adb.yaw_bins).astype(int)
    iy = (idxs % adb.yaw_bins).astype(int)
    component_labels = labels_2d[ip, iy]

    unique_faces = np.unique(face_ids)
    best: Optional[Dict[str, Any]] = None
    best_total_w = -1.0

    for face in unique_faces:
        mask_face = (face_ids == face)
        if not np.any(mask_face):
            continue
        labels_face = component_labels[mask_face]
        weights_face = weights[mask_face]
        hit_pts_face = hit_points[mask_face]
        uvs_face = uvs[mask_face]

        comp_sum: Dict[int, float] = {}
        comp_indices: Dict[int, List[int]] = {}
        for local_idx, lab in enumerate(labels_face):
            if lab == 0:
                continue
            w = float(weights_face[local_idx])
            comp_sum[lab] = comp_sum.get(lab, 0.0) + w
            comp_indices.setdefault(lab, []).append(local_idx)
        if not comp_sum:
            continue

        best_lab = max(comp_sum.keys(), key=lambda L: comp_sum[L])
        total_w = comp_sum[best_lab]
        idxs_local = comp_indices[best_lab]

        pts_sel = hit_pts_face[idxs_local]
        uvs_sel = uvs_face[idxs_local]
        w_sel = weights_face[idxs_local]

        centroid_world = tuple(np.sum(pts_sel * w_sel[:, None], axis=0) / float(np.sum(w_sel)))
        centroid_uv = tuple(np.sum(uvs_sel * w_sel[:, None], axis=0) / float(np.sum(w_sel)))
        sample_count = len(idxs_local)

        if total_w > best_total_w:
            best_total_w = total_w
            best = {
                'face_id': int(face),
                'centroid_world': centroid_world,
                'centroid_uv': centroid_uv,
                'face_solid_angle': float(total_w),
                'sample_count': sample_count,
                'cluster_label': int(best_lab),
            }

    if best is None:
        return None

    px, py, pz = position
    cx, cy, cz = best['centroid_world']
    vx = cx - px
    vy = cy - py
    vz = cz - pz
    hyp = math.hypot(vx, vz)
    yaw_rad = math.atan2(vz, vx)
    pitch_rad = -math.atan2(vy, hyp)
    yaw_deg, pitch_deg = to_minecraft_angles_degrees(yaw_rad, pitch_rad)

    best.update({'yaw_deg': yaw_deg, 'pitch_deg': pitch_deg})
    return best


# ------------------------------
# block scanning and parsing
# ------------------------------

def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _parse_block_string(bs: str) -> Tuple[str, str, Dict[str, Any]]:
    s = bs.strip()
    meta: Dict[str, Any] = {}
    if '[' in s and s.endswith(']'):
        head, rest = s.split('[', 1)
        rest = rest[:-1]
        kvs = rest.split(',')
        for kv in kvs:
            if '=' not in kv:
                continue
            k, v = kv.split('=', 1)
            k = k.strip()
            v = v.strip()
            if v in ('true', 'false'):
                meta[k] = (v == 'true')
            else:
                try:
                    meta[k] = int(v)
                except Exception:
                    meta[k] = v
        base = head
    else:
        base = s

    base = base.strip()
    base_lower = base.lower()

    if base_lower.endswith(':water') or base_lower == 'minecraft:water':
        simple = 'transparent'
    elif base_lower.endswith(':air') or base_lower == 'minecraft:air':
        simple = 'transparent'
    elif base_lower.endswith('_slab') or ':slab' in base_lower:
        simple = 'slab'
        if 'type' in meta:
            meta['half'] = meta.get('type')
            meta.pop('type', None)
    elif base_lower.endswith('_stairs') or 'stairs' in base_lower or ':stair' in base_lower:
        simple = 'stair'
    elif base_lower.endswith('_pane') or 'pane' in base_lower:
        simple = 'pane'
        conns = []
        for d in ('east', 'west', 'north', 'south'):
            if meta.get(d, False):
                conns.append(d)
        meta['connections'] = conns
    else:
        short = base_lower.split(':')[-1]
        simple = short
    return base, simple, meta


# ------------------------------
# candidate gathering in cone
# ------------------------------

def gather_candidate_polygons_in_cone(
    cache: BlockGeometryCache,
    position: Vec3,
    target_yaw_min: float,
    target_yaw_max: float,
    target_pitch_min: float,
    target_pitch_max: float,
    nearby_blocks: List[Tuple[BlockPos, str, str, Optional[Mapping[str, Any]]]],
    depth_limit: float = 200.0,
) -> List[Dict[str, Any]]:
    cand: List[Dict[str, Any]] = []
    for (pos, btype, short_type, meta) in nearby_blocks:
        polys = cache.world_polygons(pos, btype, meta)
        for p in polys:
            yaw_min, yaw_max, pitch_min, pitch_max, dmin = cache.polygon_sphere_bounds(p['verts'], position)
            if dmin > depth_limit:
                continue
            if (pitch_max < target_pitch_min) or (pitch_min > target_pitch_max):
                continue
            c1, s1 = _interval_center_span(yaw_min, yaw_max)
            c2, s2 = _interval_center_span(target_yaw_min, target_yaw_max)
            d = abs(normalize_angle_rad(c1 - c2))
            if d <= 0.5 * (s1 + s2) + 1e-12:
                p_copy = p.copy()
                p_copy.update({'sph_bounds': (yaw_min, yaw_max, pitch_min, pitch_max, dmin)})
                cand.append(p_copy)
    return cand

# ------------------------------
# library internal objects
# ------------------------------

@lru_cache(maxsize=1)
def get_adb(yaw_bins, pitch_bins) -> HighResADB:
    return HighResADB(yaw_bins, pitch_bins)

@lru_cache(maxsize=1)
def get_blockcache() -> BlockGeometryCache:
    return BlockGeometryCache()


# ------------------------------
# library api
# ------------------------------

def scan_target(
    position: Tuple[float, float, float],
    target: Tuple[int, int, int],
    occluders: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]],
    adb_granularity: Tuple[int, int] = (256, 124),
) -> Optional[TargetInfo]:

    if not occluders:
        return None

    pos_to_occluder_id: Dict[BlockPos, int] = {tuple(entry[0]): i for i, entry in enumerate(occluders)}

    tpos = target
    if tuple(tpos) not in pos_to_occluder_id:
        return None

    tid = pos_to_occluder_id[tuple(tpos)]
    tentry = next((entry for entry in occluders if entry[0] == tpos), None)
    if tentry is None:
        return None
    tbase, tshort, tmeta = tentry[1], tentry[2], tentry[3]

    block_geom_cache = get_blockcache()

    adb = get_adb(adb_granularity[0], adb_granularity[1])
    adb.reset_depth()

    n_occluders = sum(1 for pos, base, _, _ in occluders if base not in ('minecraft:air', 'minecraft:water'))
    coarse_aabbs = [None] * n_occluders
    coarse_ids = np.empty(n_occluders, dtype=np.int32)
    i = 0
    for pos, base, _, _ in occluders:
        if base in ('minecraft:air', 'minecraft:water'):
            continue
        coarse_aabbs[i] = make_aabb_from_block(pos)
        coarse_ids[i] = pos_to_occluder_id[tuple(pos)]
        i += 1

    adb.rasterize_occluders(coarse_aabbs, position, occluder_ids=coarse_ids, max_depth=float('inf'))

    depth_baseline = adb.depth.copy()
    idx_baseline = adb.top_occluder_idx.copy()

    def restore_baseline():
        adb.depth[:] = depth_baseline
        adb.top_occluder_idx[:] = idx_baseline

    restore_baseline()
    target_aabb = make_aabb_from_block(tpos)

    block_geom_cache.analytic_refine_depth_in_target_cone(
        adb=adb,
        position=position,
        target_aabb=target_aabb,
        blocks=occluders,
        max_depth=float('inf'),
        yaw_margin_deg=0.5,
        pitch_margin_deg=0.5,
        pos_to_occluder_id=pos_to_occluder_id,
    )

    mask_top = (adb.top_occluder_idx == int(tid))
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in target_aabb)
    tmin_targ, tmax_targ = ray_aabb_intersection_vec(
        *position,
        adb.dx, adb.dy, adb.dz,
        xmin, xmax, ymin, ymax, zmin, zmax
    )
    ttarget = np.where(~np.isnan(tmin_targ), np.where(tmin_targ >= 0.0, tmin_targ, tmax_targ), np.nan)
    mask_tvalid = ~np.isnan(ttarget)
    mask_visible = mask_top & mask_tvalid
    if not np.any(mask_visible):
        return None

    idxs = np.nonzero(mask_visible)[0]
    hits_n = len(idxs)
    tvis = ttarget[idxs]
    dxv, dyv, dzv = adb.dx[idxs], adb.dy[idxs], adb.dz[idxs]
    hits_x = position[0] + dxv * tvis
    hits_y = position[1] + dyv * tvis
    hits_z = position[2] + dzv * tvis
    hit_points = np.stack((hits_x, hits_y, hits_z), axis=1)

    face_ids = np.empty(hits_n, dtype=np.int32)
    uvs_arr = np.empty((hits_n, 2), dtype=np.float64)
    for j in range(hits_n):
        hx, hy, hz = hit_points[j]
        fid, (u, v) = face_and_uv_for_hitpoint_nb(target_aabb, hx, hy, hz)
        face_ids[j] = fid
        uvs_arr[j, 0] = u
        uvs_arr[j, 1] = v

    yaw_all = np.arctan2(adb.dz[idxs], adb.dx[idxs])
    pitch_all = -np.arctan2(adb.dy[idxs], np.hypot(adb.dx[idxs], adb.dz[idxs]))
    yaw_min, yaw_max = wrapped_interval_from_angles(yaw_all)
    pitch_min, pitch_max = float(np.min(pitch_all)), float(np.max(pitch_all))

    weights = adb.sample_solid_angle[idxs]
    wsum = float(np.sum(weights))
    #centroid_world = compute_visible_face_centroid_aim_with_clustering(adb, target_aabb, position)
    centroid_world = tuple(np.sum(hit_points * weights[:, None], axis=0) / wsum)
    centroid_uv = tuple(np.sum(uvs_arr * weights[:, None], axis=0) / wsum)
    solid_angle = float(np.sum(weights))

    best_candidate = {
        'target_pos': tpos,
        'target_base': tbase,
        'idxs': idxs,
        'sample_count': hits_n,
        'solid_angle': solid_angle,
        'centroid_world': centroid_world,
        'centroid_uv': centroid_uv,
        'face_ids': face_ids,
        'yaw_bounds': (yaw_min, yaw_max),
        'pitch_bounds': (pitch_min, pitch_max),
    }

    cx, cy, cz = centroid_world
    vx, vy, vz = cx - position[0], cy - position[1], cz - position[2]
    hyp = math.hypot(vx, vz)
    yaw_rad = math.atan2(vz, vx)
    pitch_rad = -math.atan2(vy, hyp)
    yaw_deg, pitch_deg = to_minecraft_angles_degrees(yaw_rad, pitch_rad)

    return TargetInfo(
        best_candidate['target_pos'],
        best_candidate['centroid_world'],
        (yaw_deg, pitch_deg),
        best_candidate['yaw_bounds'],
        best_candidate['pitch_bounds']
    )

def scan_targets(
    position: Tuple[float, float, float],
    target_ids: List[str],
    occluders: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]],
    adb_granularity: Tuple[int, int] = (256, 124),
    previous_target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[TargetInfo]:

    if not occluders:
        return None

    pos_to_occluder_id: Dict[BlockPos, int] = {tuple(entry[0]): i for i, entry in enumerate(occluders)}

    target_set = set(target_ids)
    entries = []
    pos_list = []
    for (pos, base, short_type, meta) in occluders:
        if base in target_set:
            entries.append((pos, base, short_type, meta, pos_to_occluder_id[tuple(pos)]))
            pos_list.append(pos)

    previous_target = np.asarray(previous_target, dtype=np.float64)
    pos_list = np.asarray(pos_list, dtype=np.float64)
    dists = distances_to_blocks_nb(previous_target, pos_list)

    targets: List[Tuple[BlockPos, str, str, Dict[str, Any], int, float]] = [
        (entries[i][0], entries[i][1], entries[i][2], entries[i][3], entries[i][4], float(dists[i]))
        for i in range(len(entries))
    ]

    if not targets:
        return None

    block_geom_cache = get_blockcache()

    adb = get_adb(adb_granularity[0], adb_granularity[1])
    adb.reset_depth()

    n_occluders = sum(1 for pos, base, _, _ in occluders if base not in ('minecraft:air', 'minecraft:water'))
    coarse_aabbs = [None] * n_occluders
    coarse_ids = np.empty(n_occluders, dtype=np.int32)
    i = 0
    for pos, base, _, _ in occluders:
        if base in ('minecraft:air', 'minecraft:water'):
            continue
        coarse_aabbs[i] = make_aabb_from_block(pos)
        coarse_ids[i] = pos_to_occluder_id[tuple(pos)]
        i += 1

    adb.rasterize_occluders(coarse_aabbs, position, occluder_ids=coarse_ids, max_depth=float('inf'))

    depth_baseline = adb.depth.copy()
    idx_baseline = adb.top_occluder_idx.copy()

    def restore_baseline():
        adb.depth[:] = depth_baseline
        adb.top_occluder_idx[:] = idx_baseline

    best_candidate: Optional[Dict[str, Any]] = None
    targets.sort(key=lambda t: t[-1])

    for tpos, tbase, tshort, tmeta, tid, dist in targets:
        restore_baseline()
        target_aabb = make_aabb_from_block(tpos)

        block_geom_cache.analytic_refine_depth_in_target_cone(
            adb=adb,
            position=position,
            target_aabb=target_aabb,
            blocks=occluders,
            max_depth=float('inf'),
            yaw_margin_deg=0.5,
            pitch_margin_deg=0.5,
            pos_to_occluder_id=pos_to_occluder_id,
        )

        mask_top = (adb.top_occluder_idx == int(tid))
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in target_aabb)
        tmin_targ, tmax_targ = ray_aabb_intersection_vec(
            *position,
            adb.dx, adb.dy, adb.dz,
            xmin, xmax, ymin, ymax, zmin, zmax
        )
        ttarget = np.where(~np.isnan(tmin_targ), np.where(tmin_targ >= 0.0, tmin_targ, tmax_targ), np.nan)
        mask_tvalid = ~np.isnan(ttarget)
        mask_visible = mask_top & mask_tvalid
        if not np.any(mask_visible):
            continue

        idxs = np.nonzero(mask_visible)[0]
        hits_n = len(idxs)
        tvis = ttarget[idxs]
        dxv, dyv, dzv = adb.dx[idxs], adb.dy[idxs], adb.dz[idxs]
        hits_x = position[0] + dxv * tvis
        hits_y = position[1] + dyv * tvis
        hits_z = position[2] + dzv * tvis
        hit_points = np.stack((hits_x, hits_y, hits_z), axis=1)

        face_ids = np.empty(hits_n, dtype=np.int32)
        uvs_arr = np.empty((hits_n, 2), dtype=np.float64)
        for j in range(hits_n):
            hx, hy, hz = hit_points[j]
            fid, (u, v) = face_and_uv_for_hitpoint_nb(target_aabb, hx, hy, hz)
            face_ids[j] = fid
            uvs_arr[j, 0] = u
            uvs_arr[j, 1] = v

        yaw_all = np.arctan2(adb.dz[idxs], adb.dx[idxs])
        pitch_all = -np.arctan2(adb.dy[idxs], np.hypot(adb.dx[idxs], adb.dz[idxs]))
        yaw_min, yaw_max = wrapped_interval_from_angles(yaw_all)
        pitch_min, pitch_max = float(np.min(pitch_all)), float(np.max(pitch_all))

        weights = adb.sample_solid_angle[idxs]
        wsum = float(np.sum(weights))
        centroid_world = tuple(np.sum(hit_points * weights[:, None], axis=0) / wsum)
        centroid_uv = tuple(np.sum(uvs_arr * weights[:, None], axis=0) / wsum)
        solid_angle = float(np.sum(weights))

        candidate = {
            'target_pos': tpos,
            'target_base': tbase,
            'idxs': idxs,
            'sample_count': hits_n,
            'solid_angle': solid_angle,
            'centroid_world': centroid_world,
            'centroid_uv': centroid_uv,
            'face_ids': face_ids,
            'yaw_bounds': (yaw_min, yaw_max),
            'pitch_bounds': (pitch_min, pitch_max),
        }

        if best_candidate is None or candidate['solid_angle'] > best_candidate['solid_angle']:
            best_candidate = candidate

        restore_baseline()

        if best_candidate is None:
            return None

        cx, cy, cz = best_candidate['centroid_world']
        vx, vy, vz = cx - position[0], cy - position[1], cz - position[2]
        hyp = math.hypot(vx, vz)
        yaw_rad = math.atan2(vz, vx)
        pitch_rad = -math.atan2(vy, hyp)
        yaw_deg, pitch_deg = to_minecraft_angles_degrees(yaw_rad, pitch_rad)

        return TargetInfo(
            best_candidate['target_pos'],
            best_candidate['centroid_world'],
            (yaw_deg, pitch_deg),
            best_candidate['yaw_bounds'],
            best_candidate['pitch_bounds']
        )