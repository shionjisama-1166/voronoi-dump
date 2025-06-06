import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
import cmath
import collections
from matplotlib.colors import Normalize
import shapely.geometry as geom
import matplotlib.cm as cm # For ScalarMappable


def replicate_x_periodic(atoms_real, box_bounds):
    xlo, xhi = box_bounds[0]; ylo, yhi = box_bounds[1]
    Lx = xhi - xlo; Ly = yhi - ylo
    left = atoms_real.copy(); left[:, 0] -= Lx
    right = atoms_real.copy(); right[:, 0] += Lx
    tiled = np.vstack([left, atoms_real, right])
    return tiled, (Lx, Ly)

def compute_voronoi_phi6_binned_top_surface(
    atoms_real,
    box_bounds,
    bin_width_x,
    atom_radius,
    figure_size=(8, 6), # New parameter
    figure_dpi=300,     # New parameter
    clip_x_min=0, clip_x_max=800, y_range=None,
    phi6_vmin=0.0, phi6_vmax=0.8,
    axis_label_fontsize=14, tick_label_fontsize=12,
    plot_top_ghost_cells=False, # New switch parameter
    verbose=True
    ):

    matplotlib.rc('axes',  labelsize=axis_label_fontsize)
    matplotlib.rc('xtick', labelsize=tick_label_fontsize)
    matplotlib.rc('ytick', labelsize=tick_label_fontsize)

    # --- 1. Initial Setup & Y-Shift ---

    atoms_real_shifted = atoms_real.copy()
    if atoms_real_shifted.shape[0] == 0:
        if verbose: print("INFO (Voronoi binned top): atoms_real is empty.")
        return
    y_shift = box_bounds[1][0]; atoms_real_shifted[:, 1] -= y_shift
    xlo_orig, xhi_orig = box_bounds[0]; Lx_orig = xhi_orig - xlo_orig
    Ly_box = (box_bounds[1][1] - y_shift)
    shifted_box_bounds_for_tiling = [(xlo_orig, xhi_orig), (0, Ly_box)]

    # --- 2. Lateral Tiling ---
    tiled_atoms, (Lx_calc, _) = replicate_x_periodic(atoms_real_shifted, shifted_box_bounds_for_tiling)
    if not np.isclose(Lx_calc, Lx_orig) and verbose:
        print(f"WARNING: Calculated Lx ({Lx_calc}) differs from original Lx ({Lx_orig}). Using Lx_orig for PBC.")
    num_real_tiled_atoms = len(tiled_atoms)
    if num_real_tiled_atoms == 0:
        if verbose: print("INFO (Voronoi binned top): No tiled atoms."); return

    # --- 3. Bottom Mirror ---
    mirrored_atoms_bottom_y = tiled_atoms.copy(); mirrored_atoms_bottom_y[:, 1] = -mirrored_atoms_bottom_y[:, 1]
    num_bottom_mirror_atoms = len(mirrored_atoms_bottom_y)

    # --- 4. Top Boundary Definition & Ghosts ---
    x_coords_tiled = tiled_atoms[:, 0]
    x_min_for_binning = np.min(x_coords_tiled) - 1e-6
    x_max_for_binning = np.max(x_coords_tiled) + bin_width_x
    bin_edges = np.arange(x_min_for_binning, x_max_for_binning, bin_width_x)
    num_bins_x = len(bin_edges) - 1
    temp_top_profile_atom_coords = []
    if num_bins_x > 0:
        max_y_in_bin = np.full(num_bins_x, -np.inf)
        top_atom_idx_in_bin = np.full(num_bins_x, -1, dtype=int)
        for atom_idx in range(num_real_tiled_atoms):
            x_coord, y_coord = tiled_atoms[atom_idx]
            if x_coord >= x_min_for_binning and x_coord < x_max_for_binning - bin_width_x + 1e-5:
                 bin_idx = np.floor((x_coord - x_min_for_binning) / bin_width_x).astype(int)
                 if 0 <= bin_idx < num_bins_x:
                    if y_coord > max_y_in_bin[bin_idx]:
                        max_y_in_bin[bin_idx] = y_coord
                        top_atom_idx_in_bin[bin_idx] = atom_idx
        unique_top_atom_indices = np.unique(top_atom_idx_in_bin[top_atom_idx_in_bin != -1])
        if unique_top_atom_indices.size > 0: temp_top_profile_atom_coords = tiled_atoms[unique_top_atom_indices]
    top_profile_atoms_from_tiled_set = np.array(temp_top_profile_atom_coords)
    top_ghost_atoms_tiled_list = []
    if top_profile_atoms_from_tiled_set.size > 0:
        for xc, yc in top_profile_atoms_from_tiled_set:
            yg = yc + 2 * atom_radius
            top_ghost_atoms_tiled_list.append([xc, yg])
    top_ghost_atoms_tiled = np.array(top_ghost_atoms_tiled_list)
    if verbose: print(f"INFO (Voronoi binned top): Identified {len(top_profile_atoms_from_tiled_set)} unique top profile atoms, created {len(top_ghost_atoms_tiled)} top ghost atoms.")
    num_top_ghost_atoms = len(top_ghost_atoms_tiled)

    # --- 5. Combine All Points for Voronoi Diagram ---
    points_list_for_voronoi = [tiled_atoms, mirrored_atoms_bottom_y]
    if top_ghost_atoms_tiled.size > 0: points_list_for_voronoi.append(top_ghost_atoms_tiled)
    if not points_list_for_voronoi:
        if verbose: print("ERROR (Voronoi binned top): No points to send to Voronoi."); return
    points_for_voronoi = np.vstack(points_list_for_voronoi)

    # --- 6. Compute Voronoi Diagram & phi6 ---
    try: vor = Voronoi(points_for_voronoi)
    except Exception as e: print(f"ERROR (Voronoi binned top): Voronoi computation failed: {e}"); return

    neighbor_map = collections.defaultdict(set)
    for p1, p2 in vor.ridge_points:
        if p1 < len(points_for_voronoi) and p2 < len(points_for_voronoi):
            neighbor_map[p1].add(p2); neighbor_map[p2].add(p1)

    phi6_values = np.zeros(num_real_tiled_atoms) # Phi6 only for real tiled atoms

    for i in range(num_real_tiled_atoms):
        neighs_indices = neighbor_map[i]
        if not neighs_indices: phi6_values[i] = 0.0; continue
        sum_angle = 0j; xi, yi = points_for_voronoi[i]; actual_neighbors = 0
        for j_idx in neighs_indices:
            xj, yj = points_for_voronoi[j_idx]; dx, dy = xj - xi, yj - yi
            if dx < -0.5 * Lx_orig: dx += Lx_orig
            elif dx > 0.5 * Lx_orig: dx -= Lx_orig
            angle = np.arctan2(dy, dx); sum_angle += cmath.exp(1j * 6 * angle); actual_neighbors +=1
        phi6_values[i] = abs(sum_angle / actual_neighbors) if actual_neighbors > 0 else 0.0

    # --- 7. Polygon Processing & Plotting ---
    polygons_phi6_colored = []
    colors_phi6_colored = []
    polygons_top_ghosts_to_plot = []

    display_clip_box_shapely = geom.Polygon([
        (clip_x_min, 0), (clip_x_max, 0),
        (clip_x_max, Ly_box), (clip_x_min, Ly_box)
    ])

    start_idx_real_tiled = 0
    end_idx_real_tiled = num_real_tiled_atoms
    start_idx_top_ghosts = end_idx_real_tiled + num_bottom_mirror_atoms # num_bottom_mirror_atoms == num_real_tiled_atoms


    for i_vor_point in range(len(points_for_voronoi)):
        process_this_cell = False
        is_this_a_top_ghost_cell = False

        if start_idx_real_tiled <= i_vor_point < end_idx_real_tiled: # Real tiled atom
            process_this_cell = True
        elif plot_top_ghost_cells and \
             start_idx_top_ghosts <= i_vor_point < (start_idx_top_ghosts + num_top_ghost_atoms) : # Top ghost atom
            process_this_cell = True
            is_this_a_top_ghost_cell = True

        if not process_this_cell:
            continue

        region_idx = vor.point_region[i_vor_point]
        if region_idx == -1: continue
        region_vertex_indices = vor.regions[region_idx]
        if not region_vertex_indices: continue

        finite_vertices = []
        try: finite_vertices = [vor.vertices[v_idx] for v_idx in region_vertex_indices if v_idx != -1]
        except IndexError: continue
        if len(finite_vertices) < 3: shapely_poly_initial = geom.Polygon()
        else:
            try: shapely_poly_initial = geom.Polygon(finite_vertices)
            except Exception: shapely_poly_initial = geom.Polygon()
        if not shapely_poly_initial.is_valid: shapely_poly_initial = shapely_poly_initial.buffer(0)
        if shapely_poly_initial.is_empty: continue
        try: clipped_poly = shapely_poly_initial.intersection(display_clip_box_shapely)
        except Exception: continue

        current_cell_polygons_coords_parts = []
        if not clipped_poly.is_empty:
            if clipped_poly.geom_type == "Polygon": current_cell_polygons_coords_parts.append(list(clipped_poly.exterior.coords))
            elif clipped_poly.geom_type == "MultiPolygon":
                for subpoly in clipped_poly.geoms:
                    if subpoly.geom_type == "Polygon" and not subpoly.is_empty:
                        current_cell_polygons_coords_parts.append(list(subpoly.exterior.coords))

        if not current_cell_polygons_coords_parts:
            continue

        if is_this_a_top_ghost_cell:
            polygons_top_ghosts_to_plot.extend(current_cell_polygons_coords_parts)
        else:
            polygons_phi6_colored.extend(current_cell_polygons_coords_parts)
            for _ in range(len(current_cell_polygons_coords_parts)):
                colors_phi6_colored.append(phi6_values[i_vor_point])

    fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
    ax.set_facecolor('white')

    if verbose:
        print(f"INFO (Voronoi binned top): Plotting {len(polygons_phi6_colored)} physical cell parts.")
        if plot_top_ghost_cells:
            print(f"INFO (Voronoi binned top): Plotting {len(polygons_top_ghosts_to_plot)} top ghost cell parts.")

    if plot_top_ghost_cells and polygons_top_ghosts_to_plot:
        ghost_collection = PolyCollection(
            polygons_top_ghosts_to_plot,
            facecolor='none',         # Makes cells hollow
            edgecolors='black',
            linewidths=0.2,
            alpha=1.0,                # transparency
            linestyle='-',            # lines for ghosts
            zorder=0                  # Plot them behind physical cells
        )
        ax.add_collection(ghost_collection)

    # Plot phi6 colored cells (real physical cells)
    if polygons_phi6_colored:
        phi6_array_plot = np.array(colors_phi6_colored)

        if phi6_array_plot.size > 0:
            norm = Normalize(vmin=phi6_vmin, vmax=phi6_vmax)
            collection = PolyCollection(
                polygons_phi6_colored,
                array=phi6_array_plot,
                cmap='viridis',
                norm=norm,
                edgecolors='black',
                linewidths=0.2,
                zorder=1          # Above ghost cells
            )
            ax.add_collection(collection)
            cbar = fig.colorbar(collection, ax=ax, label=r'$\phi_6$ value')
            cbar.ax.tick_params(labelsize=tick_label_fontsize)
            cbar.set_label(r'$\phi_6$ value', size=axis_label_fontsize)

    ax.set_xlim(clip_x_min, clip_x_max)
    plot_y_min_final, plot_y_max_final = (0, Ly_box)
    if y_range: plot_y_min_final, plot_y_max_final = y_range
    ax.set_ylim(plot_y_min_final, plot_y_max_final)
    ax.set_aspect('equal'); ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    plt.tight_layout(); plt.show()


def parse_all_timesteps(file_path):
    with open(file_path, 'r') as file: lines = file.readlines()
    data = {}; current_timestep = None; box_bounds_from_file = None; atoms_list = []
    in_atoms_section = False; headers = []
    for i, line_content in enumerate(lines):
        line_stripped = line_content.strip()
        if line_stripped.startswith("ITEM: TIMESTEP"):
            in_atoms_section = False
            if current_timestep is not None and box_bounds_from_file is not None and atoms_list:
                data[current_timestep] = {"box_bounds": np.array(box_bounds_from_file), "atoms": np.array(atoms_list)}
            atoms_list = []; box_bounds_from_file = None
            if i + 1 < len(lines):
                try: current_timestep = int(lines[i + 1].strip())
                except ValueError: current_timestep = None; continue
            else: break
        elif line_stripped.startswith("ITEM: BOX BOUNDS"):
            current_box_bounds_list = []
            for j in range(3):
                if i + j + 1 < len(lines):
                    try: current_box_bounds_list.append(list(map(float, lines[i + j + 1].split())))
                    except ValueError: current_box_bounds_list = None; break
                else: current_box_bounds_list = None; break
            if current_box_bounds_list and len(current_box_bounds_list) == 3: box_bounds_from_file = np.array(current_box_bounds_list)
            else: box_bounds_from_file = None
        elif line_stripped.startswith("ITEM: ATOMS"):
            in_atoms_section = True; headers = line_stripped.split()[2:]
            if "xs" not in headers or "ys" not in headers: in_atoms_section = False; headers = []
        elif in_atoms_section and headers:
            try:
                atom_line_values = list(map(float, line_stripped.split()))
                if len(atom_line_values) == len(headers):
                    atom_dict = {headers[k]: atom_line_values[k] for k in range(len(headers))}
                    if "xs" in atom_dict and "ys" in atom_dict: atoms_list.append([atom_dict["xs"], atom_dict["ys"]])
            except ValueError: pass
    if current_timestep is not None and box_bounds_from_file is not None and atoms_list:
        data[current_timestep] = {"box_bounds": np.array(box_bounds_from_file), "atoms": np.array(atoms_list)}
    if not data: print("WARNING (parser): No timesteps were successfully parsed from the file.")
    return data

def convert_to_real_coords(atoms_fractional, simulation_box_bounds):
    if not isinstance(atoms_fractional, np.ndarray) or atoms_fractional.ndim != 2 or atoms_fractional.shape[1] != 2: return np.array([])
    if not isinstance(simulation_box_bounds, (list, np.ndarray)) or len(simulation_box_bounds) < 2: return np.array([])
    x_bounds, y_bounds = simulation_box_bounds[0], simulation_box_bounds[1]
    if not (hasattr(x_bounds, '__len__') and len(x_bounds) == 2 and hasattr(y_bounds, '__len__') and len(y_bounds) == 2): return np.array([])
    try:
        x_span = float(x_bounds[1]) - float(x_bounds[0]); y_span = float(y_bounds[1]) - float(y_bounds[0])
        x_lo = float(x_bounds[0]); y_lo = float(y_bounds[0])
    except ValueError: return np.array([])
    real_coords = np.zeros_like(atoms_fractional)
    real_coords[:, 0] = atoms_fractional[:, 0] * x_span + x_lo
    real_coords[:, 1] = atoms_fractional[:, 1] * y_span + y_lo
    return real_coords


if __name__ == '__main__':
    file_path = 'dump_LNP_10M.data'


    timestep_index_to_try = -39     # -1 for last


    bin_width_for_top_surface = 40.0
    particle_radius_for_ghosts = 10.0
    y_range_view = (0, 200)
    clip_x_min_view = None
    clip_x_max_view = None

    phi6_heatmap_min = 0.0
    phi6_heatmap_max = 0.8

    show_top_ghost_cells = True # Set to True to visualize ghost cells, False to hide
    verbose_output = True

    # Font sizes for the plot
    plot_axis_label_fontsize = 14
    plot_tick_label_fontsize = 14

    all_data = parse_all_timesteps(file_path)
    timesteps = []
    if all_data:
        timesteps = sorted(all_data.keys())

    if not timesteps:
        print("ERROR (main): No timesteps available to process from the file.")
    else:
        print(f"INFO (main): Available timesteps count: {len(timesteps)}. First: {timesteps[0]}, Last: {timesteps[-1]}")

        chosen_timestep = None
        if timestep_index_to_try < 0: # Handle negative indices
            actual_index = len(timesteps) + timestep_index_to_try
        else: # Handle positive indices
            actual_index = timestep_index_to_try

        if 0 <= actual_index < len(timesteps):
            chosen_timestep = timesteps[actual_index]
        else:
            print(f"WARNING (main): Timestep index {timestep_index_to_try} (resolved to {actual_index}) is out of range for {len(timesteps)} timesteps. Using last timestep.")
            chosen_timestep = timesteps[-79]

        print(f"INFO (main): Selected timestep for processing: {chosen_timestep}")

        if chosen_timestep in all_data:
            current_timestep_data = all_data[chosen_timestep]
            atoms_fractional = np.array(current_timestep_data.get("atoms"), copy=True)
            simulation_box = np.array(current_timestep_data.get("box_bounds"), copy=True)

            if not isinstance(atoms_fractional, np.ndarray) or atoms_fractional.size == 0:
                 print(f"ERROR (main): 'atoms' data for timestep {chosen_timestep} is empty or not found.")
            elif not isinstance(simulation_box, np.ndarray) or simulation_box.size == 0:
                 print(f"ERROR (main): 'box_bounds' data for timestep {chosen_timestep} is empty or not found.")
            else:
                real_coords = convert_to_real_coords(atoms_fractional, simulation_box)

                if real_coords.size > 0:
                    if verbose_output:
                        print(f"INFO (main): Successfully converted coordinates for timestep: {chosen_timestep}. Number of atoms: {len(real_coords)}")
                        print(f"DEBUG (main): Simulation Box for selected timestep ({chosen_timestep}): Lo={simulation_box[:,0].tolist()}, Hi={simulation_box[:,1].tolist()}")

                    current_clip_x_min = clip_x_min_view if clip_x_min_view is not None else simulation_box[0,0]
                    current_clip_x_max = clip_x_max_view if clip_x_max_view is not None else simulation_box[0,1]
                    current_y_range_view = y_range_view
                    compute_voronoi_phi6_binned_top_surface(
                        atoms_real=real_coords,
                        box_bounds=simulation_box, # Pass the original box_bounds
                        bin_width_x=bin_width_for_top_surface,
                        atom_radius=particle_radius_for_ghosts,
                        clip_x_min=current_clip_x_min,
                        clip_x_max=current_clip_x_max,
                        y_range=current_y_range_view,
                        phi6_vmin=phi6_heatmap_min,
                        phi6_vmax=phi6_heatmap_max,
                        axis_label_fontsize=plot_axis_label_fontsize,
                        tick_label_fontsize=plot_tick_label_fontsize,
                        plot_top_ghost_cells=show_top_ghost_cells,
                        verbose=verbose_output
                    )
                else:
                    print(f"ERROR (main): No real coordinates to process for timestep {chosen_timestep} after conversion, or conversion failed.")
        else:
            print(f"ERROR (main): Timestep {chosen_timestep} not found in parsed data.")
