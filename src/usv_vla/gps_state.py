"""Versioned GPS state helpers; latitude/longitude are degrees in WGS84.

Coordinates are computed in double precision in a local tangent approximation,
then cast to float32 only after conversion to metres.  The approximation is
intended for the 150 m simulation tasks, not long-distance geodesy.
"""
import math

import numpy as np

STATE_DIM = 39
MAX_GOAL_M = 150.0
LOCAL_POSITION_SCALE_M = 256.0
NEAR_SCALE_M = 5.0


def _latlon(value):
    if value is None or len(value) != 2:
        raise ValueError('Expected latitude and longitude in degrees')
    lat, lon = map(float, value)
    if not math.isfinite(lat) or not math.isfinite(lon) or abs(lat) > 90 or abs(lon) > 180:
        raise ValueError('Invalid GPS latitude or longitude')
    return lat, lon


def gps_e7(position):
    """Store latitude/longitude as signed integers at 1e-7 degree resolution."""
    lat, lon = _latlon(position)
    return np.array([round(lat * 10_000_000), round(lon * 10_000_000)], dtype=np.int32)


def gps_position_state(state, gps):
    """35-value state with absolute WGS84 degrees; keep float64 until tokenized."""
    base = np.asarray(state, dtype=np.float64)
    if base.shape != (33,) or not np.isfinite(base).all():
        raise ValueError('Expected finite 33-dimensional legacy state')
    lat, lon = _latlon(gps)
    return np.concatenate((base, np.array([lat, lon], dtype=np.float64)))


def local_east_north(origin, position):
    lat0, lon0 = _latlon(origin)
    lat, lon = _latlon(position)
    # WGS84 meridional and prime-vertical radii of curvature.
    a, e2 = 6378137.0, 6.69437999014e-3
    phi = math.radians((lat0 + lat) / 2)
    denom = math.sqrt(1 - e2 * math.sin(phi) ** 2)
    n = a / denom
    m = a * (1 - e2) / denom ** 3
    delta_lon = (lon - lon0 + 180) % 360 - 180
    return n * math.cos(phi) * math.radians(delta_lon), m * math.radians(lat - lat0)


def extend_state(state, *, gps=None, origin=None, goal=None, yaw=None):
    base = np.asarray(state, dtype=np.float32)
    if base.shape != (33,) or not np.isfinite(base).all():
        raise ValueError('Expected finite 33-dimensional legacy state')
    extra = np.zeros(6, dtype=np.float32)
    if gps is None:
        if goal is not None:
            raise ValueError('A GPS target requires a valid GPS fix')
    else:
        if origin is None:
            raise ValueError('GPS position requires a fixed episode origin')
        east, north = local_east_north(origin, gps)
        if max(abs(east), abs(north)) > LOCAL_POSITION_SCALE_M:
            raise ValueError('GPS position exceeds local origin range')
        extra[:3] = east, north, 1
        if goal is not None:
            if yaw is None or not math.isfinite(yaw):
                raise ValueError('GPS target requires a valid ENU heading')
            de, dn = local_east_north(gps, goal)
            if math.hypot(de, dn) > MAX_GOAL_M + 1e-6:
                raise ValueError('GPS target exceeds 150 m')
            c, s = math.cos(yaw), math.sin(yaw)
            extra[3:] = c*de + s*dn, -s*de + c*dn, 1
    return np.concatenate((base, extra))


def normalize_extra(extra):
    extra = np.asarray(extra, dtype=np.float32).copy()
    if extra.shape != (6,) or not np.isfinite(extra).all():
        raise ValueError('Expected finite six-dimensional GPS state')
    if extra[2] not in (0, 1) or extra[5] not in (0, 1):
        raise ValueError('GPS validity flags must be 0 or 1')
    if not extra[2] and (np.any(extra[:2] != 0) or np.any(extra[3:] != 0)):
        raise ValueError('Missing GPS must have zero position and no target')
    if not extra[5] and np.any(extra[3:5] != 0):
        raise ValueError('Missing GPS goal must have zero relative coordinates')
    if np.max(np.abs(extra[:2])) > LOCAL_POSITION_SCALE_M:
        raise ValueError('GPS position outside local range')
    if np.linalg.norm(extra[3:5].astype(np.float64)) > MAX_GOAL_M + 1e-4:
        raise ValueError('GPS goal outside 150 m range')
    extra[:2] /= LOCAL_POSITION_SCALE_M
    # More bins near zero: ~0.14 m at the origin, coarser far from target.
    extra[3:5] = np.sign(extra[3:5]) * np.log1p(np.abs(extra[3:5])/NEAR_SCALE_M) / math.log1p(MAX_GOAL_M/NEAR_SCALE_M)
    return extra
