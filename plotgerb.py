import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from shapely.geometry import LineString, box, Point as ShapelyPoint
from scipy.interpolate import interp1d
import matplotlib.patches as patches
import ezdxf
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm as rl_mm 
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from matplotlib.path import Path
import math
from shapely.ops import unary_union
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from shapely.geometry import Point, Polygon
from matplotlib.patches import Rectangle, Circle, PathPatch, Ellipse


# --- Gerber parsing functions  ---

def read_gerber_file(path):
    with open(path, 'r') as f:
        return f.readlines()
def parse_aperture_definitions(lines, units):
    aperture_defs = {}
    pattern = re.compile(r'%ADD(\d+)([A-Za-z]*),([^*]+)\*%')

    # Define conversion factor
    conversion_factor = 1.0
    if units == 'inches':
        conversion_factor = 25.4 # Convert inches to mm

    for line in lines:
        m = pattern.match(line)
        if m:
            num = int(m.group(1))
            shape = m.group(2)
            params_str = m.group(3) # Original string parameters

            # Convert parameters string to a list of floats, then apply conversion factor
            converted_params = []
            if 'X' in params_str:
                raw_sizes = map(float, params_str.split('X'))
            else:
                raw_sizes = map(float, params_str.split())

            for size in raw_sizes:
                converted_params.append(size * conversion_factor)

            aperture_defs[num] = {
                'shape': shape,
                'params': converted_params # Store as converted list of floats
            }
    return aperture_defs

def detect_units(lines):
    for line in lines:
        if '%MOIN*' in line:
            return 'inches'
        elif '%MOMM*' in line:
            return 'mm'
        elif 'G70*' in line or 'G70 ' in line:
            return 'inches'
        elif 'G71*' in line or 'G71 ' in line:
            return 'mm'
    return 'mm'

def extract_coordinates(gerber_lines, units, aperture_defs):
    paths = []
    flashes = []

    current_path = []
    current_pos = (0, 0)
    pen_down = False
    in_region = False
    current_aperture = None

    coord_pattern = re.compile(r'^X(-?\d+)Y(-?\d+)D(\d+)\*$')
    aperture_pattern = re.compile(r'^D(\d+)\*$')
    coord_format_pattern = re.compile(r'%FSLAX(\d)(\d)Y(\d)(\d)\*%')
    divisor = 1000000
    
    for line in gerber_lines:
        line = line.strip()
        if not line:
            continue

        if 'G36' in line:
            if current_path:
                paths.append(np.array(current_path))
                current_path = []
            in_region = True
            pen_down = True
            continue
        elif 'G37' in line:
            if current_path:
                if not np.allclose(current_path[0], current_path[-1]):
                    current_path.append(current_path[0])
                paths.append(np.array(current_path))
                current_path = []
            in_region = False
            pen_down = False
            continue
        elif 'G01' in line:
            continue
        elif 'G02' in line or 'G03' in line:
            pass

        m_fmt = coord_format_pattern.match(line)
        if m_fmt:
            dec_x = int(m_fmt.group(2))
            dec_y = int(m_fmt.group(4))
            max_dec = max(dec_x, dec_y)
            divisor = 10 ** max_dec
            continue

        if line.startswith('%'):
            continue

        m_ap = aperture_pattern.match(line)
        if m_ap:
            current_aperture = int(m_ap.group(1))
            continue

        m = coord_pattern.match(line)
        if m:
            x_raw = int(m.group(1))
            y_raw = int(m.group(2))
            d_code = int(m.group(3))

            if units == 'inches':
                x_raw = x_raw * 25.4
                y_raw = y_raw * 25.4

            x = x_raw / divisor
            y = y_raw / divisor

            if d_code == 1:
                if not pen_down:
                    current_path = [current_pos]
                current_path.append((x, y))
                current_pos = (x, y)
                pen_down = True

            elif d_code == 2:
                if current_path and not in_region:
                    paths.append(np.array(current_path))
                    current_path = []
                current_pos = (x, y)
                pen_down = False

            elif d_code == 3:
                if current_path and not in_region:
                    paths.append(np.array(current_path))
                    current_path = []
                    pen_down = False

                flash_cmd = {
                    'command': 'D03',
                    'x': x,
                    'y': y,
                    'aperture_def': aperture_defs.get(current_aperture, None)
                }
                flashes.append(flash_cmd)
                current_pos = (x, y)

    if current_path:
        if in_region and not np.allclose(current_path[0], current_path[-1]):
             current_path.append(current_path[0])
        paths.append(np.array(current_path))

    return paths, flashes

def calculate_centroids(paths):
    centroids = []
    for path in paths:
        area = 0
        cx = 0
        cy = 0
        for i in range(len(path) - 1):  
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            common_factor = (x1 * y2) - (x2 * y1)
            area += common_factor
            cx += (x1 + x2) * common_factor
            cy += (y1 + y2) * common_factor
        area *= 0.5

        if abs(area) < 1e-6: 
            cx = sum(x for x, y in path) / len(path)
            cy = sum(y for x, y in path) / len(path)
        else:
            cx /= (6 * area)
            cy /= (6 * area)

        centroids.append((cx, cy))
    return centroids

def calculate_flash_centroids(flashes):
    return [(flash['x'], flash['y']) for flash in flashes]

# --- PCA functions ---

def quarter_arc(cx, cy, radius_x, radius_y, start_angle, n_points):
    angles = np.linspace(start_angle, start_angle + np.pi / 2, n_points, endpoint=False)
    return np.array([(cx + radius_x * np.cos(a), cy + radius_y * np.sin(a)) for a in angles])



def generate_aperture_points(shape, params, center, axes, num_points=64):
    """
    Generates a numpy array of points for a given aperture shape, centered
    at 'center' and aligned with 'axes'. Now handles KiCad's composite ROUNDRECT.
    """
    u, v = np.array(axes[0]), np.array(axes[1]) # Unit vectors for local axes
    center_vec = np.array(center)
    shape = shape.upper()

    if isinstance(params, str):
        if 'X' in params:
            sizes = list(map(float, params.split('X')))
        else:
            sizes = list(map(float, params.split()))
    else:
        sizes = list(params)

    if shape in ['C', 'O']:
        if len(sizes) == 1:
            width, height = sizes[0], sizes[0]
        elif len(sizes) >= 2:
            width, height = sizes[0], sizes[1]
        else:
            return np.array([center_vec])

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points_relative_to_uv_basis = np.array([
            (width / 2) * np.cos(angles),
            (height / 2) * np.sin(angles)
        ]).T
        
        # Apply rotation and translation
        transformed_points = np.dot(points_relative_to_uv_basis, np.array([u,v]).T) + center_vec
        return transformed_points

    elif shape == 'R':
        if len(sizes) >= 2:
            width, height = sizes[0], sizes[1]
            # Corners relative to (0,0) in rectangle's local basis
            corners_uv_basis = np.array([
                [-(width/2), -(height/2)],
                [ (width/2), -(height/2)],
                [ (width/2),  (height/2)],
                [-(width/2),  (height/2)]
            ])
            
            # Apply rotation and translation
            transformed_corners = np.dot(corners_uv_basis, np.array([u,v]).T) + center_vec
            
            num_segments_per_side = max(2, num_points // 4)
            all_points = []
            for i in range(4):
                start_pt = transformed_corners[i]
                end_pt = transformed_corners[(i + 1) % 4]
                segment_points = np.linspace(start_pt, end_pt, num_segments_per_side, endpoint=False)
                all_points.extend(segment_points.tolist())
            
            if all_points and not np.allclose(all_points[0], all_points[-1]):
                all_points.append(all_points[0]) # Close polygon
            
            return np.array(all_points)
        else:
            return np.array([center_vec])

    elif shape == 'ROUNDRECT' or shape == 'RR':
        if len(sizes) >= 9:
            r = sizes[0] # Rounding radius
            
            # Macro corners (P1-P4) relative to aperture's local origin (0,0)
            macro_corners_local_offsets = np.array([
                [sizes[1], sizes[2]], # P1
                [sizes[3], sizes[4]], # P2
                [sizes[5], sizes[6]], # P3
                [sizes[7], sizes[8]]  # P4
            ])

            shapely_geometries = []
            
            # Add four circular pads (buffers around macro corners)
            for mc_offset in macro_corners_local_offsets:
                shapely_geometries.append(Point(mc_offset[0], mc_offset[1]).buffer(r))

            # Add four connecting rectangular pads
            for i in range(4):
                p_current_local = macro_corners_local_offsets[i]
                p_next_local = macro_corners_local_offsets[(i + 1) % 4]

                connector_vec_local = p_next_local - p_current_local
                connector_length = np.linalg.norm(connector_vec_local)

                if connector_length < 1e-6: # Handle degenerate connector
                    continue 

                connector_unit_vec_local = connector_vec_local / connector_length
                perp_unit_vec_local = np.array([-connector_unit_vec_local[1], connector_unit_vec_local[0]])

                # Calculate corners of the connecting rectangle in local offset space
                mid_point_local = (p_current_local + p_next_local) / 2.0
                
                half_len = connector_length / 2.0
                half_width = r 

                rect_corners_relative_to_midpoint = np.array([
                    mid_point_local - half_len * connector_unit_vec_local - half_width * perp_unit_vec_local,
                    mid_point_local + half_len * connector_unit_vec_local - half_width * perp_unit_vec_local,
                    mid_point_local + half_len * connector_unit_vec_local + half_width * perp_unit_vec_local,
                    mid_point_local - half_len * connector_unit_vec_local + half_width * perp_unit_vec_local 
                ])
                
                shapely_geometries.append(Polygon(rect_corners_relative_to_midpoint))

            if not shapely_geometries:
                return np.array([center_vec]) 

            # Perform union to get a single, clean outer boundary
            union_geometry = unary_union(shapely_geometries)

            # Extract exterior points
            if union_geometry.geom_type == 'Polygon':
                points_relative_to_uv_basis = np.array(list(union_geometry.exterior.coords))
            elif union_geometry.geom_type == 'MultiPolygon':
                # For MultiPolygon, take the largest exterior (common for well-formed ROUNDRECT)
                largest_polygon = max(union_geometry.geoms, key=lambda geom: geom.area)
                points_relative_to_uv_basis = np.array(list(largest_polygon.exterior.coords))
            else:
                return np.array([center_vec]) # Fallback for unexpected geometry types

            # Apply rotation and translation to clean perimeter points
            transformed_points = np.dot(points_relative_to_uv_basis, np.array([u,v]).T) + center_vec
            
            # Ensure polygon is explicitly closed
            if transformed_points.shape[0] > 1 and not np.allclose(transformed_points[0], transformed_points[-1]):
                transformed_points = np.append(transformed_points, [transformed_points[0]], axis=0)

            return transformed_points
        else:
            return np.array([center_vec]) # Insufficient parameters
    
    return np.array([center_vec]) # Fallback for unknown shapes

def apply_pca(points):
    points = np.array(points)
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order] # Get sorted eigenvalues
    return eigvecs.T, eigvals # Return both

# --- Symmetry line calculation ---

from scipy.spatial import cKDTree

def bound_line_to_geometry(line, shape):
    # shape: np.array of points [[x1,y1], [x2,y2], ...]
    min_x, min_y = shape[:, 0].min(), shape[:, 1].min()
    max_x, max_y = shape[:, 0].max(), shape[:, 1].max()
    shape_bbox = box(min_x, min_y, max_x, max_y)
    bounded_line = line.intersection(shape_bbox)

    if bounded_line.is_empty:
        return None
    if bounded_line.geom_type == "MultiLineString":
        return max(bounded_line, key=lambda seg: seg.length)
    return bounded_line

def axis_line_params(axis):
    p1, p2 = np.array(axis.coords[0]), np.array(axis.coords[1])
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Axis length zero")
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    return p1, direction, normal

def reflect_points(points, axis):
    p1, _, normal = axis_line_params(axis)
    vecs = points - p1
    dist_normal = vecs @ normal
    reflected_vecs = vecs - 2 * np.outer(dist_normal, normal)
    return p1 + reflected_vecs

def points_similarity(points_a, points_b, tol):
    if len(points_a) == 0 or len(points_b) == 0:
        return False
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    dists_ab, _ = tree_b.query(points_a, k=1)
    dists_ba, _ = tree_a.query(points_b, k=1)
    return np.all(dists_ab < tol) and np.all(dists_ba < tol)

def check_symmetry(path, axis, tol=1e-2 * 5):
    points = np.array(path)
    reflected = reflect_points(points, axis)
    return points_similarity(points, reflected, tol), points, reflected

def generate_candidate_axes(path, step_deg=5):
    min_x, min_y = min(x for x, y in path), min(y for x, y in path)
    max_x, max_y = max(x for x, y in path), max(y for x, y in path)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    length = max(max_x - min_x, max_y - min_y) * 2

    candidates = []
    for angle in range(0, 180, step_deg):
        rad = np.radians(angle)
        dx = np.cos(rad) * length / 2
        dy = np.sin(rad) * length / 2
        candidates.append(LineString([
            (center_x - dx, center_y - dy),
            (center_x + dx, center_y + dy)
        ]))
    return candidates

def angle_of_axis(axis):
    dx = axis.coords[1][0] - axis.coords[0][0]
    dy = axis.coords[1][1] - axis.coords[0][1]
    return np.degrees(np.arctan2(dy, dx)) % 180

def is_cardinal(angle, tol=10):
    return min(abs(angle - 0), abs(angle - 90), abs(angle - 180)) <= tol


def calculate_symmetry_axes(path, tolerance_deg=10, tol_compare=0.01):
    min_x, min_y = min(x for x, y in path), min(y for x, y in path)
    max_x, max_y = max(x for x, y in path), max(y for x, y in path)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    length = max(max_x - min_x, max_y - min_y) * 2

    def perpendicular_axis(center_coords, line_length, angle_degrees):
        rad = np.radians(angle_degrees + 90)
        dx, dy = np.cos(rad) * line_length / 2, np.sin(rad) * line_length / 2
        return LineString([(center_coords[0] - dx, center_coords[1] - dy), (center_coords[0] + dx, center_coords[1] + dy)])

    fixed_angles = [0, 45, 90, 135]
    fixed_axes = []

    for angle in fixed_angles:
        rad = np.radians(angle)
        dx = np.cos(rad) * length / 2
        dy = np.sin(rad) * length / 2
        axis = LineString([
            (center_x - dx, center_y - dy),
            (center_x + dx, center_y + dy)
        ])
        symmetric, _, _ = check_symmetry(path, axis, tol=tol_compare)
        if symmetric:
            fixed_axes.append(axis)

    if fixed_axes:
        best = fixed_axes[0]

        center = ((best.coords[0][0] + best.coords[1][0]) / 2,
                  (best.coords[0][1] + best.coords[1][1]) / 2)
        perp = perpendicular_axis(center, length, angle_of_axis(best))
        best = bound_line_to_geometry(best, np.array(path))
        perp = bound_line_to_geometry(perp, np.array(path))
        if best.length >= perp.length:
            return [best, perp]
        else:
            return [perp, best]

    candidate_axes = generate_candidate_axes(path)
    sym_axes = []

    for axis in candidate_axes:
        symmetric, _, _ = check_symmetry(path, axis, tol=tol_compare)
        if symmetric:
            sym_axes.append(axis)

    if sym_axes:
        cardinal_axes = [a for a in sym_axes if is_cardinal(angle_of_axis(a), tol=tolerance_deg)]

        if cardinal_axes:
            best = min(cardinal_axes,
                       key=lambda a: min(abs(angle_of_axis(a) - 0),
                                         abs(angle_of_axis(a) - 90),
                                         abs(angle_of_axis(a) - 180)))
        else:
            best = sym_axes[0]

        angle = angle_of_axis(best)
        center = ((best.coords[0][0] + best.coords[1][0]) / 2,
                  (best.coords[0][1] + best.coords[1][1]) / 2)
        length = best.length

        perp = perpendicular_axis(center, length, angle)
        best = bound_line_to_geometry(best, np.array(path))
        perp = bound_line_to_geometry(perp, np.array(path))

        if best.length >= perp.length:
            return [best, perp]
        else:
            return [perp, best]

    # Fallback PCA axes
    shape = np.array(path)
    center = np.mean(shape, axis=0)
    pca = PCA(n_components=2) 
    pca.fit(shape)
    axes = pca.components_
    length = max(np.ptp(shape[:, 0]), np.ptp(shape[:, 1])) * 2

    main_axis = axes[0]
    dx = main_axis[0] * length / 2
    dy = main_axis[1] * length / 2

    best = LineString([
        (center[0] - dx, center[1] - dy),
        (center[0] + dx, center[1] + dy)
    ])

    perp_dx = -main_axis[1] * length / 2
    perp_dy = main_axis[0] * length / 2

    perp = LineString([
        (center[0] - perp_dx, center[1] - perp_dy),
        (center[0] + perp_dx, center[1] + perp_dy)
    ])

    best = bound_line_to_geometry(best, np.array(path))
    perp = bound_line_to_geometry(perp, np.array(path))

    if best.length >= perp.length:
        return [best, perp]
    else:
        return [perp, best]

    # Fallback PCA axes
    shape = np.array(path)
    center = np.mean(shape, axis=0)
    pca = PCA(n_components=2)
    pca.fit(shape)
    axes = pca.components_
    length = max(np.ptp(shape[:, 0]), np.ptp(shape[:, 1])) * 2

    main_axis = axes[0]
    dx = main_axis[0] * length / 2
    dy = main_axis[1] * length / 2

    best = LineString([
        (center[0] - dx, center[1] - dy),
        (center[0] + dx, center[1] + dy)
    ])

    perp_dx = -main_axis[1] * length / 2
    perp_dy = main_axis[0] * length / 2

    perp = LineString([
        (center[0] - perp_dx, center[1] - perp_dy),
        (center[0] + perp_dx, center[1] + perp_dy)
    ])

    best = bound_line_to_geometry(best, np.array(path))
    perp = bound_line_to_geometry(perp, np.array(path))

    if best.length >= perp.length:
        return [best, perp]
    else:
        return [perp, best]
    

def scale_shapes(path, scale_factors, axes, flash_shape=None, flash_params=None, flash_center=None):
    if flash_shape is not None and flash_params is not None and flash_center is not None:
        points = generate_aperture_points(flash_shape, flash_params, flash_center, axes, num_points=64)
    else:
        points = np.array(path)
    # Calculate centroid from shape points
    centroid = points.mean(axis=0)
    # Axis directions
    _, dir_x, normal_x = axis_line_params(axes[0])
    _, dir_y, normal_y = axis_line_params(axes[1])
    # Project points on axis directions relative to centroid
    vecs = points - centroid
    x_proj = vecs @ dir_x
    y_proj = vecs @ dir_y
    # Scale projections
    x_proj_scaled = x_proj * scale_factors[0]
    y_proj_scaled = y_proj * scale_factors[1]
    # Rebuild points
    scaled_points = centroid + np.outer(x_proj_scaled, dir_x) + np.outer(y_proj_scaled, dir_y)
    return scaled_points

def plot_pca_axes(ax, center, axes, scale=1, color='m', label=None):
    for i, axis in enumerate(axes):
        axis = axis / np.linalg.norm(axis)
        p1 = center + axis * scale
        p2 = center - axis * scale
        if label and i == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, label=label)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2)

def get_scaled_aperture_data(aperture_def, x_pos, y_pos, scale_sym, scale_perp):
    shape = aperture_def['shape']
    params = aperture_def['params']

    # Step A: Generate Original Aperture Points (Axis-Aligned, Centered at 0,0)
    original_aperture_points = generate_aperture_points(
        shape=shape,
        params=params,
        center=(0, 0),
        axes=(np.array([1, 0]), np.array([0, 1]))
    )
    
    # Handle degenerate or empty point sets.
    if original_aperture_points.shape[0] == 0 or \
       (original_aperture_points.shape[0] == 1 and np.allclose(original_aperture_points[0], [0,0])):
        scaled_radius = 0.01 * min(scale_sym, scale_perp) if min(scale_sym, scale_perp) > 0 else 0.01
        return {
            'shape_type': 'Point',
            'center_x': x_pos,
            'center_y': y_pos,
            'scaled_dims': {'radius': scaled_radius}
        }

    # Step B: Calculate Aperture's Intrinsic Symmetry Axes
    aperture_sym_axes = calculate_symmetry_axes(original_aperture_points)

    if len(aperture_sym_axes) < 2:
        # Fallback if symmetry axes cannot be found
        scaled_radius = 0.01 * min(scale_sym, scale_perp) if min(scale_sym, scale_perp) > 0 else 0.01
        return {
            'shape_type': 'Point',
            'center_x': x_pos,
            'center_y': y_pos,
            'scaled_dims': {'radius': scaled_radius}
        }
    
    # Step C: Scale Aperture Points using its own symmetry axes
    scaled_aperture_points_centered = scale_shapes(
        original_aperture_points,
        (scale_sym, scale_perp),
        aperture_sym_axes
    )

    # Step D: Translate Scaled Points to the actual flash location
    scaled_aperture_points_translated = scaled_aperture_points_centered + np.array([x_pos, y_pos])

    # Re-parameterize scaled points for Matplotlib patches.
    # For robust plotting of scaled shapes, especially if rotation occurs during scaling,
    # we default to 'Polygon' for drawing.
    
    scaled_data = {
        'original_shape': shape,
        'center_x': x_pos,
        'center_y': y_pos,
        'scaled_dims': {}
    }

    scaled_data['shape_type'] = 'Polygon'
    scaled_data['scaled_dims'] = {'points': scaled_aperture_points_translated.tolist()}
    
    # Handle degenerate cases (very small or zero dimensions after scaling)
    min_x_final, min_y_final = np.min(scaled_aperture_points_translated, axis=0)
    max_x_final, max_y_final = np.max(scaled_aperture_points_translated, axis=0)
    final_width = max_x_final - min_x_final
    final_height = max_y_final - min_y_final


    if final_width <= 1e-6 and final_height <= 1e-6:
        scaled_data['shape_type'] = 'Point'
        scaled_data['scaled_dims'] = {'radius': 0.01 * min(scale_sym, scale_perp) if min(scale_sym, scale_perp) > 0 else 0.01}
    elif final_width <= 1e-6:
        scaled_data['shape_type'] = 'Ellipse'
        scaled_data['scaled_dims'] = {'width': 0.01 * scale_sym, 'height': final_height}
    elif final_height <= 1e-6:
        scaled_data['shape_type'] = 'Ellipse'
        scaled_data['scaled_dims'] = {'width': final_width, 'height': 0.01 * scale_perp}

    return scaled_data


# --- Plotting ---

def plot_paths_and_symmetry(ax, paths, centroids, flash_centroids, flash_commands, scale_sym=1, scale_perp=1):
    """
    Plots original and scaled paths, symmetry axes, centroids, and original and scaled flashes/apertures
      onto a given Matplotlib Axes object (ax).
    """
    ax.clear()
    all_plotted_points = []

    # Plot Paths
    for path in paths:
        current_path_points_np = np.array(path)
        if current_path_points_np.size > 0:
            all_plotted_points.append(current_path_points_np)

        xs, ys = zip(*path)
        ax.plot(xs, ys, 'b-', label='Original Path') # Consolidated plotting

        sym_axes = calculate_symmetry_axes(path)
        
        if len(sym_axes) < 2:
            continue

        scaled_path = scale_shapes(path, (scale_sym, scale_perp), sym_axes)
        xs_s, ys_s = scaled_path[:, 0], scaled_path[:, 1]
        ax.plot(xs_s, ys_s, color='orange', linestyle='--', label='Scaled Path')
        if scaled_path.size > 0:
            all_plotted_points.append(scaled_path)

        # Plot Symmetry Axes
        for axis in sym_axes:
            x_axis, y_axis = axis.xy
            ax.plot(x_axis, y_axis, 'm--', linewidth=2, label='Symmetry Axis')
            all_plotted_points.append(np.array(list(zip(x_axis, y_axis))))

    shape_colors = {'C': 'red', 'O': 'orange', 'R': 'green', 'ROUNDRECT': 'purple', 'RR': 'purple'}
    default_color = 'gray'
    plotted_labels = set()

    # Plot Flashes (Apertures)
    for cmd in flash_commands:
        if cmd['command'] == 'D03' and cmd['aperture_def']:
            shape = cmd['aperture_def']['shape'].upper()
            params = cmd['aperture_def']['params']
            x, y = cmd['x'], cmd['y']

            key_color = shape if shape not in ['ROUNDRECT', 'RR'] else 'ROUNDRECT'
            color = shape_colors.get(key_color, default_color)
            label_original = f"Original Aperture: {shape}"
            label_scaled = f"Scaled Aperture: {shape}"
            add_label_original = label_original if label_original not in plotted_labels else None
            add_label_scaled = label_scaled if label_scaled not in plotted_labels else None

            # Plot original aperture shape
            original_aperture_points_translated = generate_aperture_points(
                shape=shape,
                params=params,
                center=(x,y),
                axes=(np.array([1,0]), np.array([0,1]))
            )
            if original_aperture_points_translated.shape[0] > 1:
                original_path_obj = Path(original_aperture_points_translated)
                original_patch = patches.PathPatch(
                    original_path_obj, 
                    facecolor=color, 
                    edgecolor=color, 
                    alpha=0.5, 
                    label=add_label_original
                )
                ax.add_patch(original_patch)
                if original_aperture_points_translated.size > 0:
                    all_plotted_points.append(original_aperture_points_translated)
            elif original_aperture_points_translated.shape[0] == 1:
                original_patch = patches.Circle((x,y), 0.01, facecolor=color, edgecolor=color, alpha=0.5, label=add_label_original)
                ax.add_patch(original_patch)
                all_plotted_points.append(np.array([[x,y]]))


            # Plot scaled aperture shape overlay
            scaled_aperture_data = get_scaled_aperture_data(
                aperture_def=cmd['aperture_def'],
                x_pos=x,
                y_pos=y,
                scale_sym=scale_sym,
                scale_perp=scale_perp
            )
            
            scaled_type = scaled_aperture_data['shape_type']
            scaled_dims = scaled_aperture_data['scaled_dims']
            center_x_scaled_data, center_y_scaled_data = scaled_aperture_data['center_x'], scaled_aperture_data['center_y']

            patch = None

            if scaled_type == 'Ellipse':
                patch = patches.Ellipse(
                    (center_x_scaled_data, center_y_scaled_data),
                    scaled_dims['width'],
                    scaled_dims['height'],
                    edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5, alpha=0.7, label=add_label_scaled
                )
                half_width = scaled_dims['width'] / 2.0
                half_height = scaled_dims['height'] / 2.0
                all_plotted_points.append(np.array([
                    [center_x_scaled_data - half_width, center_y_scaled_data - half_height],
                    [center_x_scaled_data + half_width, center_y_scaled_data - half_height],
                    [center_x_scaled_data + half_width, center_y_scaled_data + half_height],
                    [center_x_scaled_data - half_width, center_y_scaled_data + half_height]
                ]))
            elif scaled_type == 'Rectangle':
                scaled_points_rect = np.array(scaled_dims['points'])
                min_x_rect, min_y_rect = np.min(scaled_points_rect, axis=0)
                max_x_rect, max_y_rect = np.max(scaled_points_rect, axis=0)
                width_rect = max_x_rect - min_x_rect
                height_rect = max_y_rect - min_y_rect

                patch = patches.FancyBboxPatch(
                    (min_x_rect, min_y_rect),
                    width_rect, height_rect,
                    boxstyle="round,pad=0,rounding_size=0",
                    edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5, alpha=0.7, label=add_label_scaled
                )
                if scaled_points_rect.size > 0:
                    all_plotted_points.append(scaled_points_rect)
            elif scaled_type == 'Polygon':
                scaled_points = np.array(scaled_dims['points'])
                if scaled_points.shape[0] > 1:
                    path = Path(scaled_points)
                    patch = patches.PathPatch(
                        path, 
                        facecolor='none',
                        edgecolor=color, 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.7, 
                        label=add_label_scaled
                    )
                    if scaled_points.size > 0:
                        all_plotted_points.append(scaled_points)
            elif scaled_type == 'Point':
                patch = patches.Circle(
                    (center_x_scaled_data, center_y_scaled_data),
                    scaled_dims['radius'],
                    edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5, alpha=0.7, label=add_label_scaled
                )
                radius = scaled_dims['radius']
                all_plotted_points.append(np.array([
                    [center_x_scaled_data - radius, center_y_scaled_data - radius],
                    [center_x_scaled_data + radius, center_y_scaled_data - radius],
                    [center_x_scaled_data + radius, center_y_scaled_data + radius],
                    [center_x_scaled_data - radius, center_y_scaled_data + radius]
                ]))
            
            if patch:
                ax.add_patch(patch)

            if add_label_original:
                plotted_labels.add(label_original)
            if add_label_scaled:
                plotted_labels.add(label_scaled)

    # Plot Centroids (originally commented out, now removed)
    if centroids:
        cx, cy = zip(*centroids)
    if flash_centroids:
        fcx, fcy = zip(*flash_centroids)

    # Final Plot Settings
    ax.set_title(f"Scaled Geometry (scale_sym: {scale_sym}, scale_perp: {scale_perp})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True)

    # --- Robust Plot Limits and Aspect Ratio ---
    if all_plotted_points:
        combined_points = np.vstack([p for p in all_plotted_points if p.size > 0])
        
        finite_points = combined_points[np.all(np.isfinite(combined_points), axis=1)]

        if finite_points.shape[0] > 0:
            min_x, min_y = np.min(finite_points, axis=0)
            max_x, max_y = np.max(finite_points, axis=0)
            
            range_x = max_x - min_x
            range_y = max_y - min_y
            
            if range_x < 1e-6: range_x = 1.0
            if range_y < 1e-6: range_y = 1.0

            padding_factor = 0.1
            min_absolute_padding = 0.5

            padding_x = max(range_x * padding_factor, min_absolute_padding)
            padding_y = max(range_y * padding_factor, min_absolute_padding)
            
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    ax.set_aspect('equal', adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))

    return ax.figure


def export_scaled_geometry_to_dxf(paths, flash_commands, filename="output.dxf", scale_sym=1, scale_perp=1):
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Process and export scaled paths
    for path in paths:
        current_path_points = np.array(path)
        scaled_path_points = current_path_points # Default to unscaled

        if current_path_points.size > 0:
            try:
                sym_axes = calculate_symmetry_axes(current_path_points)
                if len(sym_axes) >= 2:
                    scaled_path_points = scale_shapes(current_path_points, (scale_sym, scale_perp), sym_axes)
            except Exception as e:
                scaled_path_points = current_path_points # Fallback if scaling fails

        if scaled_path_points.shape[0] > 1:
            # Export paths as LWPOLYLINE entities
            is_closed = np.allclose(scaled_path_points[0], scaled_path_points[-1])
            msp.add_lwpolyline(scaled_path_points.tolist(), close=is_closed)

    # Process and export scaled flashes (apertures)
    for cmd in flash_commands:
        if cmd['command'] == 'D03' and cmd['aperture_def']:
            try:
                scaled_aperture_data = get_scaled_aperture_data(
                    aperture_def=cmd['aperture_def'],
                    x_pos=cmd['x'],
                    y_pos=cmd['y'],
                    scale_sym=scale_sym,
                    scale_perp=scale_perp
                )

                shape_type = scaled_aperture_data['shape_type']
                center_x = scaled_aperture_data['center_x']
                center_y = scaled_aperture_data['center_y']
                scaled_dims = scaled_aperture_data['scaled_dims']

                if shape_type == 'Ellipse':
                    width = scaled_dims['width']
                    height = scaled_dims['height']
                    
                    # Handle ellipse major/minor axis for DXF compatibility
                    if width >= height:
                        major_axis_vec = (width / 2, 0)
                        ratio = height / width
                    else:
                        major_axis_vec = (0, height / 2)
                        ratio = width / height

                    msp.add_ellipse(
                        center=(center_x, center_y),
                        major_axis=major_axis_vec,
                        ratio=ratio
                    )

                elif shape_type == 'Point':
                    radius = scaled_dims['radius']
                    msp.add_circle(center=(center_x, center_y), radius=radius)

                elif shape_type == 'Polygon':
                    # Export polygons as LWPOLYLINE entities
                    points = np.array(scaled_dims['points'])
                    if points.shape[0] > 1:
                        is_closed = np.allclose(points[0], points[-1])
                        msp.add_lwpolyline(points.tolist(), close=is_closed)

            except Exception as e:
                continue 

    # Save to DXF file
    try:
        doc.saveas(filename)
        print(f"DXF file '{filename}' exported successfully.")
    except Exception as e:
        print(f"Error exporting DXF file: {e}")


def export_scaled_geometry_to_pdf(
    paths,
    flash_commands,
    scale_sym=1,
    scale_perp=1,
    translate_x_offset_mm=0,
    translate_y_offset_mm=0,
    fiducial_offset_percent=80,
    filename="output.pdf"
):
    pdf_canvas_obj = canvas.Canvas(filename, pagesize=landscape(A4))

    # Call the shared rendering function with all parameters
    # The render function will now handle all transformations internally based on 1:1 logic
    render_geometry_to_canvas_reportlab(
        pdf_canvas_obj,
        paths,
        flash_commands,
        scale_sym,
        scale_perp,
        translate_x_offset_mm,
        translate_y_offset_mm,
        fiducial_offset_percent
    )

    try:
        pdf_canvas_obj.showPage()
        pdf_canvas_obj.save()
    except Exception as e:
        print(f"Error exporting PDF file: {e}")


def render_geometry_to_canvas_reportlab(
    pdf_canvas_obj,
    paths,
    flash_commands,
    scale_sym,
    scale_perp,
    translate_x_offset_mm=0,
    translate_y_offset_mm=0,
    fiducial_offset_percent=0
):
    pdf_canvas_obj.saveState()

    # Define plot margin
    plot_margin_mm = 10.0

    all_scaled_points_for_bbox = []
    processed_paths = []
    
    # Process and collect scaled path points
    for path in paths:
        current_path_points = np.array(path)
        scaled_path_points = current_path_points

        if current_path_points.size > 0:
            try:
                sym_axes = calculate_symmetry_axes(current_path_points)
                if len(sym_axes) >= 2:
                    scaled_path_points = scale_shapes(current_path_points, (scale_sym, scale_perp), sym_axes)
                else:
                    scaled_path_points = current_path_points * scale_sym
            except Exception:
                scaled_path_points = current_path_points * scale_sym
        
        if scaled_path_points.size > 0:
            all_scaled_points_for_bbox.append(scaled_path_points)
            processed_paths.append(scaled_path_points)

    processed_apertures_data = []

    # Process and collect scaled aperture data
    for cmd in flash_commands:
        if cmd['command'] == 'D03' and cmd['aperture_def']:
            try:
                scaled_data = get_scaled_aperture_data(
                    aperture_def=cmd['aperture_def'],
                    x_pos=cmd['x'],
                    y_pos=cmd['y'],
                    scale_sym=scale_sym,
                    scale_perp=scale_perp
                )
                processed_apertures_data.append(scaled_data)

                if scaled_data['shape_type'] == 'Polygon':
                    if np.array(scaled_data['scaled_dims']['points']).size > 0:
                        all_scaled_points_for_bbox.append(np.array(scaled_data['scaled_dims']['points']))
                elif scaled_data['shape_type'] in ['Ellipse', 'Point']:
                    center_x = scaled_data['center_x']
                    center_y = scaled_data['center_y']
                    
                    if scaled_data['shape_type'] == 'Ellipse':
                        temp_width = scaled_data['scaled_dims']['width']
                        temp_height = scaled_data['scaled_dims']['height']
                    else:
                        temp_width = scaled_data['scaled_dims']['radius'] * 2
                        temp_height = scaled_data['scaled_dims']['radius'] * 2

                    temp_corners = np.array([
                        [center_x - temp_width/2, center_y - temp_height/2],
                        [center_x + temp_width/2, center_y - temp_height/2],
                        [center_x + temp_width/2, center_y + temp_height/2],
                        [center_x - temp_width/2, center_y + temp_height/2]
                    ])
                    all_scaled_points_for_bbox.append(temp_corners)
            except Exception:
                continue

    if not all_scaled_points_for_bbox:
        pdf_canvas_obj.restoreState()
        return

    combined_points = np.vstack(all_scaled_points_for_bbox)
    min_x_geom, min_y_geom = np.min(combined_points, axis=0)
    max_x_geom, max_y_geom = np.max(combined_points, axis=0)

    geom_width_mm = max_x_geom - min_x_geom
    geom_height_mm = max_y_geom - min_y_geom

    # Ensure minimum positive dimensions for geometry
    if geom_width_mm <= 0:
        geom_width_mm = 0.1
    if geom_height_mm <= 0:
        geom_height_mm = 0.1

    # Calculate canvas translation to align geometry to margin and offset
    total_translate_x_mm = plot_margin_mm - min_x_geom + translate_x_offset_mm
    total_translate_y_mm = plot_margin_mm - min_y_geom + translate_y_offset_mm

    # Apply translation to ReportLab canvas
    pdf_canvas_obj.translate(total_translate_x_mm * rl_mm, total_translate_y_mm * rl_mm)

    # Set canvas scale to 1 drawing unit = 1 millimeter
    pdf_canvas_obj.scale(rl_mm, rl_mm)

    padding_for_background_mm = 10.0
    
    # Draw black background rectangle
    bg_rect_x = min_x_geom - padding_for_background_mm
    bg_rect_y = min_y_geom - padding_for_background_mm
    bg_rect_width = geom_width_mm + (2 * padding_for_background_mm)
    bg_rect_height = geom_height_mm + (2 * padding_for_background_mm)

    pdf_canvas_obj.setFillColor(colors.black)
    pdf_canvas_obj.rect(
        bg_rect_x,
        bg_rect_y,
        bg_rect_width,
        bg_rect_height,
        fill=1
    )

    pdf_canvas_obj.setStrokeColor(colors.white)
    pdf_canvas_obj.setLineWidth(0.00025)
    pdf_canvas_obj.setFillColor(colors.white)

    # Draw scaled paths
    for scaled_path_points in processed_paths:
        if scaled_path_points.shape[0] > 1:
            path_obj = pdf_canvas_obj.beginPath()
            path_obj.moveTo(scaled_path_points[0, 0], scaled_path_points[0, 1])
            for i in range(1, scaled_path_points.shape[0]):
                path_obj.lineTo(scaled_path_points[i, 0], scaled_path_points[i, 1])
            
            do_fill = False
            if np.allclose(scaled_path_points[0], scaled_path_points[-1]) and scaled_path_points.shape[0] > 2:
                path_obj.close()
                do_fill = True

            pdf_canvas_obj.drawPath(path_obj, stroke=1, fill=do_fill)
        elif scaled_path_points.shape[0] == 1:
            point_x, point_y = scaled_path_points[0]
            radius = 0.1
            pdf_canvas_obj.circle(point_x, point_y, radius, fill=1, stroke=0)

    # Draw scaled apertures
    for scaled_aperture_data in processed_apertures_data:
        shape_type = scaled_aperture_data['shape_type']
        
        center_x = scaled_aperture_data['center_x']
        center_y = scaled_aperture_data['center_y']
        
        if shape_type == 'Ellipse':
            width = scaled_aperture_data['scaled_dims']['width']
            height = scaled_aperture_data['scaled_dims']['height']
            
            pdf_canvas_obj.ellipse(
                (center_x - width/2), (center_y - height/2),
                (center_x + width/2), (center_y + height/2),
                fill=1, stroke=0
            )
        elif shape_type == 'Point':
            radius = scaled_aperture_data['scaled_dims']['radius']
            pdf_canvas_obj.circle(center_x, center_y, radius, fill=1, stroke=0)
        elif shape_type == 'Polygon':
            points = np.array(scaled_aperture_data['scaled_dims']['points'])
            
            if points.shape[0] > 1:
                path_obj = pdf_canvas_obj.beginPath()
                path_obj.moveTo(points[0][0], points[0][1])
                for i in range(1, points.shape[0]):
                    path_obj.lineTo(points[i][0], points[i][1])
                path_obj.close()
                pdf_canvas_obj.drawPath(path_obj, fill=1, stroke=0)
    
    # Draw fiducial marks
    fiducial_radius_mm = 1.0
    
    center_x_geom = min_x_geom + (geom_width_mm / 2.0)
    center_y_geom = min_y_geom + (geom_height_mm / 2.0)

    offset_x = (geom_width_mm / 2.0) * (fiducial_offset_percent / 100.0)
    offset_y = (geom_height_mm / 2.0) * (fiducial_offset_percent / 100.0)

    fiducial_positions = [
        (center_x_geom - offset_x, center_y_geom - offset_y),
        (center_x_geom + offset_x, center_y_geom - offset_y),
        (center_x_geom + offset_x, center_y_geom + offset_y),
        (center_x_geom - offset_x, center_y_geom + offset_y)
    ]

    pdf_canvas_obj.setFillColor(colors.white)
    for fx, fy in fiducial_positions:
        pdf_canvas_obj.circle(fx, fy, fiducial_radius_mm, fill=1, stroke=0)

    pdf_canvas_obj.restoreState()

# --- Main ---

def main(file_path, scale_sym, scale_perp, plot_func, export_dxf_func=None, export_pdf_func=None):
    """
    Parses Gerber data, applies scaling, calls a plotting function, and conditionally calls export functions.
    """
    gerber_lines = read_gerber_file(file_path)
    units = detect_units(gerber_lines)
    aperture_defs = parse_aperture_definitions(gerber_lines, units)
    paths, flashes = extract_coordinates(gerber_lines, units, aperture_defs)
    centroids = calculate_centroids(paths) if paths else []
    flash_centroids = calculate_flash_centroids(flashes) if flashes else []

    plot_func(paths, centroids, flash_centroids, flashes, scale_sym=scale_sym, scale_perp=scale_perp)

    if export_dxf_func:
        export_dxf_func(paths, flashes, scale_sym=scale_sym, scale_perp=scale_perp)
    if export_pdf_func:
        export_pdf_func(paths, flashes, scale_sym=scale_sym, scale_perp=scale_perp)


class GeometryApp:
    def __init__(self, master):
        self.master = master
        master.title("Gerber Geometry Scaler")
        master.geometry("1000x800")
        master.config(bg="#f0f0f0")

        self.file_path = tk.StringVar(value="")
        self.scale_sym = tk.DoubleVar(value=1.0)
        self.scale_perp = tk.DoubleVar(value=1.0)

        self._create_widgets()

    def _create_widgets(self):
        control_frame = tk.Frame(self.master, bd=2, relief="groove", padx=10, pady=10, bg="#e0e0e0")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Gerber File:", bg="#e0e0e0", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=2)
        self.file_label = tk.Label(control_frame, textvariable=self.file_path, bg="#ffffff", relief="sunken", width=40, anchor="w", padx=5)
        self.file_label.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        tk.Button(control_frame, text="Browse", command=self._select_file).grid(row=0, column=2, padx=5, pady=2)

        tk.Label(control_frame, text="Scale Symmetric (X):", bg="#e0e0e0").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(control_frame, textvariable=self.scale_sym, width=10).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        tk.Label(control_frame, text="Scale Perpendicular (Y):", bg="#e0e0e0").grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(control_frame, textvariable=self.scale_perp, width=10).grid(row=2, column=1, padx=5, pady=2, sticky="w")

        tk.Button(control_frame, text="Run Visualization", command=self._run_visualization,
                  font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")

        export_button_frame = tk.Frame(control_frame, bg="#e0e0e0")
        export_button_frame.grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.export_dxf_btn = tk.Button(export_button_frame, text="Export DXF", command=self._export_dxf,
                                        state=tk.DISABLED, bg="#2196F3", fg="white", font=("Arial", 10))
        self.export_dxf_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.export_pdf_btn = tk.Button(export_button_frame, text="Export PDF", command=self._export_pdf,
                                        state=tk.DISABLED, bg="#F44336", fg="white", font=("Arial", 10))
        self.export_pdf_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        control_frame.grid_columnconfigure(1, weight=1)

        self.plot_frame = tk.Frame(self.master, bd=2, relief="sunken", bg="white")
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("No Plot Yet")
        self.ax.grid(True)
        self.canvas.draw()


    def _select_file(self):
        fpath = filedialog.askopenfilename(
            title="Select Gerber File",
            filetypes=(("Gerber Files", "*.gbr"), ("All Files", "*.*"))
        )
        if fpath:
            self.file_path.set(fpath)
            self.export_dxf_btn.config(state=tk.DISABLED)
            self.export_pdf_btn.config(state=tk.DISABLED)

    def _run_visualization(self):
        gerber_file = self.file_path.get()
        if not gerber_file:
            messagebox.showerror("Error", "Please select a Gerber file first.")
            return

        current_scale_sym = self.scale_sym.get()
        current_scale_perp = self.scale_perp.get()

        try:
            self.ax.clear()
            # Parse Gerber data and store the raw (unscaled) results as instance attributes
            gerber_lines = read_gerber_file(gerber_file)
            units = detect_units(gerber_lines)
            aperture_defs = parse_aperture_definitions(gerber_lines, units)
            # Store raw paths and flashes for later use by export functions
            self.raw_paths, self.raw_flashes = extract_coordinates(gerber_lines, units, aperture_defs)
            
            # For visualization, apply scaling and calculate centroids
            # The plot_paths_and_symmetry function will handle its own scaling for display
            centroids = calculate_centroids(self.raw_paths) if self.raw_paths else []
            flash_centroids = calculate_flash_centroids(self.raw_flashes) if self.raw_flashes else []

            # Call the plot function (which internally scales for display)
            plot_paths_and_symmetry(
                self.ax,
                self.raw_paths, # Pass raw data
                centroids,
                flash_centroids,
                self.raw_flashes, # Pass raw data
                current_scale_sym,
                current_scale_perp
            )
            self.canvas.draw()

            self.export_dxf_btn.config(state=tk.NORMAL)
            self.export_pdf_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.export_dxf_btn.config(state=tk.DISABLED)
            self.export_pdf_btn.config(state=tk.DISABLED)
            # Clear stored data on error to prevent inconsistent state
            self.raw_paths = []
            self.raw_flashes = []


    def _export_dxf(self):
        gerber_file = self.file_path.get()
        if not gerber_file:
            messagebox.showwarning("Warning", "Please run visualization first.")
            return

        output_filename = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=(("DXF Files", "*.dxf"), ("All Files", "*.*")),
            title="Save DXF File"
        )
        if output_filename:
            try:
                main(
                    file_path=gerber_file,
                    scale_sym=self.scale_sym.get(),
                    scale_perp=self.scale_perp.get(),
                    plot_func=lambda *args, **kwargs: None,
                    export_dxf_func=lambda paths, flashes, scale_sym, scale_perp:
                                    export_scaled_geometry_to_dxf(paths, flashes, filename=output_filename, scale_sym=scale_sym, scale_perp=scale_perp),
                    export_pdf_func=None
                )
                messagebox.showinfo("Success", f"DXF file exported to: {output_filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting DXF: {e}")


    def _export_pdf(self):
        # Check if visualization has been run and raw data is available
        if (not hasattr(self, 'raw_paths') or not self.raw_paths) and \
            (not hasattr(self, 'raw_flashes') or not self.raw_flashes):
            messagebox.showwarning("Warning", "Please run visualization first to load geometry data.")
            return
        PdfExportPreview(
            self.master,
            self.raw_paths,
            self.raw_flashes,
            self.scale_sym.get(),
            self.scale_perp.get()
        )

class PdfExportPreview(tk.Toplevel):
    def __init__(self, master, raw_paths, raw_flashes, scale_sym_main, scale_perp_main):
        super().__init__(master)
        self.title("PDF Export Preview")
        self.geometry("1200x900") # Adjust size as needed
        self.config(bg="#f0f0f0")

        # Store the raw geometry data and main scaling factors
        self.raw_paths = raw_paths
        self.raw_flashes = raw_flashes
        self.scale_sym_main = scale_sym_main
        self.scale_perp_main = scale_perp_main

        # Tkinter variables for adjustable PDF parameters
        self.translate_x_offset_mm = tk.DoubleVar(value=0.0)
        self.translate_y_offset_mm = tk.DoubleVar(value=0.0)
        self.fiducial_offset_percent = tk.DoubleVar(value=80.0) # Default to 80% as in original export

        # Matplotlib figure and canvas for the preview plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()

        self._create_widgets()
        self._initial_preview() # Display initial plot upon window open

    def _create_widgets(self):
        # Frame for controls at the bottom
        control_frame = tk.Frame(self, bg="#f0f0f0", bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Translation X Offset
        tk.Label(control_frame, text="Translate X (mm):", bg="#f0f0f0").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(control_frame, textvariable=self.translate_x_offset_mm, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        self.translate_x_offset_mm.trace_add("write", lambda *args: self._update_preview())

        # Translation Y Offset
        tk.Label(control_frame, text="Translate Y (mm):", bg="#f0f0f0").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(control_frame, textvariable=self.translate_y_offset_mm, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        self.translate_y_offset_mm.trace_add("write", lambda *args: self._update_preview())

        # Fiducial Offset Percentage
        tk.Label(control_frame, text="Fiducial Offset (%):", bg="#f0f0f0").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(control_frame, textvariable=self.fiducial_offset_percent, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        self.fiducial_offset_percent.trace_add("write", lambda *args: self._update_preview())

        # Export PDF Button
        tk.Button(control_frame, text="Export Final PDF", command=self._export_final_pdf).pack(side=tk.RIGHT, padx=5, pady=5)

        # Pack the Matplotlib canvas widget
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add Matplotlib Toolbar for navigation
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas_widget.pack() # Re-pack to ensure toolbar is visible properly

    def _initial_preview(self):
        self._update_preview() 

    def _update_preview(self):
        self.ax.clear()
        self.ax.set_aspect('equal', adjustable='box')

        page_width_pts, page_height_pts = landscape(A4)
        page_width_mm = page_width_pts / rl_mm
        page_height_mm = page_height_pts / rl_mm

        plot_margin_mm = 10.0
        padding_mm = 10.0 

        drawable_width_mm = page_width_mm - (2 * plot_margin_mm)
        drawable_height_mm = page_height_mm - (2 * plot_margin_mm)

        a4_outline = Rectangle(
            (0, 0), page_width_mm, page_height_mm,
            fill=False, edgecolor='blue', linewidth=1
        )
        
        self.ax.add_patch(a4_outline)

        all_scaled_points_for_bbox = []
        processed_paths = []

        if self.raw_paths:
            for path in self.raw_paths:
                current_path_points = np.array(path)
                scaled_path_points = current_path_points

                if current_path_points.size > 0:
                    try:
                        sym_axes = calculate_symmetry_axes(current_path_points)
                        if len(sym_axes) >= 2:
                            scaled_path_points = scale_shapes(current_path_points, (self.scale_sym_main, self.scale_perp_main), sym_axes)
                    except Exception as e:
                        print(f"Error scaling path: {e}")

                if scaled_path_points.size > 0:
                    all_scaled_points_for_bbox.append(scaled_path_points)
                    processed_paths.append(scaled_path_points)

        processed_apertures_data = []
        if self.raw_flashes:
            for cmd in self.raw_flashes:
                if cmd['command'] == 'D03' and cmd['aperture_def']:
                    try:
                        scaled_data = get_scaled_aperture_data(
                            aperture_def=cmd['aperture_def'],
                            x_pos=cmd['x'],
                            y_pos=cmd['y'],
                            scale_sym=self.scale_sym_main,
                            scale_perp=self.scale_perp_main
                        )
                        processed_apertures_data.append(scaled_data)

                        # Collect bounding box points for apertures
                        if scaled_data['shape_type'] == 'Polygon':
                            if np.array(scaled_data['scaled_dims']['points']).size > 0:
                                all_scaled_points_for_bbox.append(np.array(scaled_data['scaled_dims']['points']))
                        elif scaled_data['shape_type'] in ['Ellipse', 'Point']:
                            center_x = scaled_data['center_x']
                            center_y = scaled_data['center_y']
                            
                            if scaled_data['shape_type'] == 'Ellipse':
                                temp_width = scaled_data['scaled_dims']['width']
                                temp_height = scaled_data['scaled_dims']['height']
                            else: # Point
                                temp_width = scaled_data['scaled_dims']['radius'] * 2
                                temp_height = scaled_data['scaled_dims']['radius'] * 2
                            
                            temp_corners = np.array([
                                [center_x - temp_width/2, center_y - temp_height/2],
                                [center_x + temp_width/2, center_y - temp_height/2],
                                [center_x + temp_width/2, center_y + temp_height/2],
                                [center_x - temp_width/2, center_y + temp_height/2]
                            ])
                            all_scaled_points_for_bbox.append(temp_corners)
                    except Exception as e:
                        print(f"Error processing aperture: {e}") 
                        continue

        geom_width_mm = 0.0
        geom_height_mm = 0.0
        min_x_geom, max_x_geom, min_y_geom, max_y_geom = 0, 0, 0, 0

        if all_scaled_points_for_bbox:
            combined_points = np.vstack(all_scaled_points_for_bbox)
            min_x_geom, min_y_geom = np.min(combined_points, axis=0)
            max_x_geom, max_y_geom = np.max(combined_points, axis=0)

            geom_width_mm = max_x_geom - min_x_geom
            geom_height_mm = max_y_geom - min_y_geom

        
            if geom_width_mm <= 0:
                geom_width_mm = 0.1
                if min_x_geom == max_x_geom: # If it's a true zero width line/point
                    min_x_geom -= 0.05
                    max_x_geom += 0.05
            if geom_height_mm <= 0:
                geom_height_mm = 0.1
                if min_y_geom == max_y_geom: # If it's a true zero height line/point
                    min_y_geom -= 0.05
                    max_y_geom += 0.05
        else:
            self.ax.set_xlim(0, page_width_mm)
            self.ax.set_ylim(0, page_height_mm)
            self.canvas.draw_idle()
            return

        
        pdf_plot_scale = 1.0

        if geom_width_mm > drawable_width_mm or geom_height_mm > drawable_height_mm:
            scale_factor_x = drawable_width_mm / geom_width_mm
            scale_factor_y = drawable_height_mm / geom_height_mm
            pdf_plot_scale = min(scale_factor_x, scale_factor_y)

        
        base_tx_mm = plot_margin_mm - (min_x_geom * pdf_plot_scale)
        base_ty_mm = plot_margin_mm - (min_y_geom * pdf_plot_scale)

        tx_final_mm = base_tx_mm + self.translate_x_offset_mm.get()
        ty_final_mm = base_ty_mm + self.translate_y_offset_mm.get()

        # Background rectangle (visual aid for geometry's overall bounding box)
        
        box_ll_x = (min_x_geom * pdf_plot_scale) + tx_final_mm - padding_mm
        box_ll_y = (min_y_geom * pdf_plot_scale) + ty_final_mm - padding_mm
        box_width = (geom_width_mm * pdf_plot_scale) + (2 * padding_mm)
        box_height = (geom_height_mm * pdf_plot_scale) + (2 * padding_mm)

        bg_rect = Rectangle((box_ll_x, box_ll_y), box_width, box_height,
                            facecolor='black', edgecolor='none', zorder=1)
        self.ax.add_patch(bg_rect)

        line_color = 'white'
        fill_color = 'white'
        line_width = 0.00025 

       
        for scaled_path_points in processed_paths:
            if scaled_path_points.shape[0] > 1:
                transformed_path_points = (scaled_path_points * pdf_plot_scale) + [tx_final_mm, ty_final_mm]
                
                path_obj = Path(transformed_path_points)
                
                do_fill = False
                if np.allclose(transformed_path_points[0], transformed_path_points[-1]) and transformed_path_points.shape[0] > 2:
                    do_fill = True

                patch = PathPatch(path_obj, facecolor=fill_color if do_fill else 'none',
                                    edgecolor=line_color, linewidth=line_width, zorder=2)
                self.ax.add_patch(patch)
            elif scaled_path_points.shape[0] == 1: # Handle single points in paths (if any)
                point_x, point_y = (scaled_path_points[0] * pdf_plot_scale) + [tx_final_mm, ty_final_mm]
                radius = line_width * 2 # Make single points slightly visible
                circle = Circle((point_x, point_y), radius, facecolor=fill_color, edgecolor=line_color, linewidth=line_width, zorder=2)
                self.ax.add_patch(circle)

        # Draw scaled flash commands (apertures)
        for scaled_aperture_data in processed_apertures_data:
            shape_type = scaled_aperture_data['shape_type']
            
            center_x_orig = scaled_aperture_data['center_x']
            center_y_orig = scaled_aperture_data['center_y']
            center_x_transformed = (center_x_orig * pdf_plot_scale) + tx_final_mm
            center_y_transformed = (center_y_orig * pdf_plot_scale) + ty_final_mm

            if shape_type == 'Ellipse':
                width_orig = scaled_aperture_data['scaled_dims']['width']
                height_orig = scaled_aperture_data['scaled_dims']['height']
                
                width_transformed = width_orig * pdf_plot_scale
                height_transformed = height_orig * pdf_plot_scale

                patch = Ellipse((center_x_transformed, center_y_transformed), width_transformed, height_transformed,
                                facecolor=fill_color, edgecolor='none', zorder=2)
                self.ax.add_patch(patch)
            elif shape_type == 'Point':
                radius_orig = scaled_aperture_data['scaled_dims']['radius']
                radius_transformed = radius_orig * pdf_plot_scale
                patch = Circle((center_x_transformed, center_y_transformed), radius_transformed,
                               facecolor=fill_color, edgecolor='none', zorder=2)
                self.ax.add_patch(patch)
            elif shape_type == 'Polygon':
                points_orig = np.array(scaled_aperture_data['scaled_dims']['points'])
                transformed_points = (points_orig * pdf_plot_scale) + [tx_final_mm, ty_final_mm]
                
                path_obj = Path(transformed_points, closed=True)
                patch = PathPatch(path_obj, facecolor=fill_color, edgecolor='none', zorder=2)
                self.ax.add_patch(patch)
        
        # Draw Fiducial Marks
        fiducial_radius_mm = 1.0 
        fiducial_offset_percent = self.fiducial_offset_percent.get()

        center_x_geom = min_x_geom + (geom_width_mm / 2.0)
        center_y_geom = min_y_geom + (geom_height_mm / 2.0)

        # Calculate offset from the geometry's *original* center, then apply overall transformation
        offset_x_geom_local = (geom_width_mm / 2.0) * (fiducial_offset_percent / 100.0)
        offset_y_geom_local = (geom_height_mm / 2.0) * (fiducial_offset_percent / 100.0)

        fiducial_positions_orig_geom = [
            (center_x_geom - offset_x_geom_local, center_y_geom - offset_y_geom_local), # Bottom-left
            (center_x_geom + offset_x_geom_local, center_y_geom - offset_y_geom_local), # Bottom-right
            (center_x_geom + offset_x_geom_local, center_y_geom + offset_y_geom_local), # Top-right
            (center_x_geom - offset_x_geom_local, center_y_geom + offset_y_geom_local)  # Top-left
        ]

        for fx_orig, fy_orig in fiducial_positions_orig_geom:
            fx_transformed = (fx_orig * pdf_plot_scale) + tx_final_mm
            fy_transformed = (fy_orig * pdf_plot_scale) + ty_final_mm

            radius_transformed = fiducial_radius_mm * pdf_plot_scale
            circle = Circle((fx_transformed, fy_transformed), radius_transformed,
                            facecolor='white', edgecolor='none', zorder=3)
            self.ax.add_patch(circle)

        # Set plot limits to the A4 page size
        self.ax.set_xlim(0, page_width_mm)
        self.ax.set_ylim(0, page_height_mm)
        self.ax.set_title("Gerber Geometry Preview") 
        self.ax.grid(False)

        self.canvas.draw_idle() 


    def _export_final_pdf(self):
        filename = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".pdf",
            filetypes=(("PDF Files", "*.pdf"), ("All Files", "*.*")),
            title="Save PDF File"
        )
        if filename:
            try:
                # Call the external PDF export function with all necessary parameters
                export_scaled_geometry_to_pdf(
                    paths=self.raw_paths, # Pass raw paths
                    flash_commands=self.raw_flashes, # Pass raw flashes
                    filename=filename,
                    scale_sym=self.scale_sym_main, # Main symmetric scale from GeometryApp
                    scale_perp=self.scale_perp_main, # Main perpendicular scale from GeometryApp
                    translate_x_offset_mm=self.translate_x_offset_mm.get(), # User-defined X translation
                    translate_y_offset_mm=self.translate_y_offset_mm.get(), # User-defined Y translation
                    fiducial_offset_percent=self.fiducial_offset_percent.get() # User-defined fiducial offset
                )
                messagebox.showinfo("Success", f"PDF file exported to: {filename}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting PDF: {e}", parent=self)


if __name__ == "__main__":
    root = tk.Tk()
    app = GeometryApp(root)
    root.mainloop()



