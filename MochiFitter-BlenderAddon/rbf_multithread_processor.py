#!/usr/bin/env python3
"""
External Multi-Process RBF Deformation Processing Script

Usage:
python rbf_multithread_processor.py temp_rbf_data.npz

This script reads temporary data files exported from Blender,
performs RBF interpolation using multi-threading, and saves the results.

Required libraries:
- numpy
- scipy
- concurrent.futures (standard library)
- psutil (for memory monitoring)
"""

import numpy as np
import os
import sys
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix  # For future compact RBF units (currently unused)
from scipy.sparse.linalg import gmres, LinearOperator  # spilu has been removed due to a change in diagonal preprocessing
from typing import Tuple, List, Dict, Any

# Check the availability of psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    #print("Warning: psutil is not installed. Memory monitoring will be disabled.")
    #print("To install: pip install psutil")

# Check Numba availability (optional performance optimization)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy definition when Numba is not available (disables decorators)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

def set_cpu_affinity():
    """Configure CPU affinity for the process to utilize all cores"""
    if not PSUTIL_AVAILABLE:
        print("CPU affinity not set: psutil is not available")
        return
    try:
        # Use all logical processors
        all_cpus = list(range(psutil.cpu_count(logical=True)))
        psutil.Process().cpu_affinity(all_cpus)
        print(f"CPU affinity set: using {len(all_cpus)} logical processors.")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

class MemoryMonitor:
    """Class for monitoring memory usage (depends on psutil)"""
    
    def __init__(self, max_memory_gb: float = None):
        if not PSUTIL_AVAILABLE:
            self.enabled = False
            self.initial_memory = 0.0
            return
        
        self.enabled = True
        self.process = psutil.Process()
        self.max_memory_bytes = max_memory_gb * 1024**3 if max_memory_gb else None
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get the current memory usage in GB"""
        if not self.enabled:
            return 0.0
        return self.process.memory_info().rss / 1024**3
    
    def get_memory_increase(self) -> float:
        """Get the amount of memory used since startup in gigabytes"""
        if not self.enabled:
            return 0.0
        return self.get_memory_usage() - self.initial_memory
    
    def is_memory_limit_exceeded(self) -> bool:
        """Check if the memory limit has been exceeded"""
        if not self.enabled or self.max_memory_bytes is None:
            return False
        return self.process.memory_info().rss > self.max_memory_bytes
    
    def get_recommended_batch_size(self, current_batch_size: int, memory_increase: float) -> int:
        """Calculate the recommended batch size based on memory usage"""
        if not self.enabled:
            return current_batch_size
        
        if memory_increase > 2.0:  # If the increase is 2 GB or more
            return max(1000, current_batch_size // 4)
        elif memory_increase > 1.0:  # If the increase is 1 GB or more
            return max(5000, current_batch_size // 2)
        else:
            return current_batch_size


def get_optimal_worker_count(total_items: int, memory_monitor: MemoryMonitor) -> int:
    """Calculate the optimal number of workers (adjusted for the process pool)"""
    # Get the number of CPU cores
    cpu_count = os.cpu_count()
    
    # Perform memory-based tuning only if psutil is available
    if PSUTIL_AVAILABLE:
        # Adjust based on memory usage
        available_memory = psutil.virtual_memory().available / 1024**3  # in GB
    else:
        # If psutil is not available, use conservative values
        available_memory = 8.0  # Assuming 8 GB
    
    # Since each process in a process pool uses memory independently, the settings should be more conservative
    if total_items > 1000000:  # Over 1 million vertices
        max_workers = min(cpu_count, 3)  # Set to a lower value than ThreadPool
    elif total_items > 500000:  # Over 500,000 vertices
        max_workers = min(cpu_count, 4)
    else:
        max_workers = min(cpu_count, 6)
    
    # Adjust based on available memory (more strictly for the process pool)
    if available_memory < 4.0:  # Less than 4GB
        max_workers = min(max_workers, 1)
    elif available_memory < 8.0:  # Less than 8 GB
        max_workers = min(max_workers, 2)
    elif available_memory < 16.0:  # Less than 16 GB
        max_workers = min(max_workers, 4)
    
    return max(1, max_workers)


def multi_quadratic_biharmonic(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Multi-Quadratic Biharmonic RBF Kernel Function"""
    return np.sqrt(r**2 + epsilon**2)


# Default data type (float32 for memory efficiency, float64 for high precision)
DEFAULT_DTYPE = np.float32


# =============================================================================
# Numba JIT Optimization Functions (P2-1)
# =============================================================================

# cache=False: Workaround for an issue where access to the cache directory hangs
# in the Microsoft Store version of Blender's sandbox environment
@jit(nopython=True, parallel=True, fastmath=True, cache=False, nogil=True)
def _cdist_sqeuclidean_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Numba JIT Version: Calculation of Euclidean Squared Distance

    Parameters:
    - A: Array of shape (m, d)
    - B: An array of shape (n, d)

    Returns:
    - A squared distance matrix of shape (m, n)

    Note:
        Setting 'nogil=True' disables the Global Interpreter Lock (GIL) during execution.
        Parallel performance is achieved even when used with a 'ThreadPoolExecutor'.
    """
    m, d = A.shape
    n = B.shape[0]
    result = np.zeros((m, n), dtype=np.float32)

    for i in prange(m):
        for j in range(n):
            dist_sq = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                dist_sq += diff * diff
            result[i, j] = dist_sq

    return result


# cache=False: In the Microsoft Store version of Blender's sandbox environment,
# this avoids an issue where access to the cache directory causes the program to hang
@jit(nopython=True, parallel=True, fastmath=True, cache=False, nogil=True)
def _cdist_euclidean_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Numba JIT Version: Euclidean Distance Calculation

    Parameters:
    - A: Array of shape (m, d)
    - B: An array of shapes (n, d)

    Returns:
    - A distance matrix of shape (m, n)

    Note:
        Setting 'nogil=True' releases the GIL during execution.
        Parallel performance is achieved even when used with a 'ThreadPoolExecutor'.
    """
    m, d = A.shape
    n = B.shape[0]
    result = np.zeros((m, n), dtype=np.float32)

    for i in prange(m):
        for j in range(n):
            dist_sq = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                dist_sq += diff * diff
            result[i, j] = np.sqrt(dist_sq)

    return result


def cdist_fast(A: np.ndarray, B: np.ndarray, metric: str = 'sqeuclidean') -> np.ndarray:
    """
    Fast Distance Calculation (JIT version when Numba is available; otherwise, scipy.cdist)

    Parameters:
    - A: Array of shape (m, d)
    - B: Array of shape (n, d)
    - metric: 'sqeuclidean' (squared Euclidean) or 'euclidean'

    Returns:
    - Distance matrix (float32)

    Note:
        This function always returns float32 (common to both Numba and scipy implementations).
        - Numba JIT version: Internally fixed to float32 (for performance optimization)
        - scipy.cdist version: Casts the result to DEFAULT_DTYPE (usually float32)

        If DEFAULT_DTYPE=float64 is allowed in the future, be aware of the precision limitations of the Numba path.
    """
    # Convert to float32 (The Numba version uses float32 by default, prioritizing performance over precision)
    A_f32 = A.astype(np.float32) if A.dtype != np.float32 else A
    B_f32 = B.astype(np.float32) if B.dtype != np.float32 else B

    if NUMBA_AVAILABLE:
        if metric == 'sqeuclidean':
            return _cdist_sqeuclidean_numba(A_f32, B_f32)
        elif metric == 'euclidean':
            return _cdist_euclidean_numba(A_f32, B_f32)
        else:
            # For unsupported metrics, fall back to scipy.cdist
            return cdist(A, B, metric).astype(DEFAULT_DTYPE)
    else:
        # If Numba is not available, use scipy.cdist
        return cdist(A, B, metric).astype(DEFAULT_DTYPE)


# =============================================================================
# GMRES Iterative Solver (P2-2)
# =============================================================================

# GMRES usage flag (experimental feature; disabled by default)
USE_GMRES_SOLVER = False

# Hybrid Parallelization Flag (P2-3)
# True: Use ThreadPoolExecutor for RBF evaluation (effective for operations where NumPy releases the GIL)
# False: Use the traditional ProcessPoolExecutor (default, prioritizes stability)
USE_HYBRID_PARALLELIZATION = False


# Maximum matrix size when using GMRES (converting dense matrices to sparse matrices is impractical in terms of memory and time)
# This limit can be relaxed after introducing compact RBF
GMRES_MAX_MATRIX_SIZE = 5000


def solve_with_gmres(A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
                     maxiter: int = 500, restart: int = 100) -> Tuple[np.ndarray, bool]:
    """
    GMRES Iterative Solver (Diagonal Preconditioning)

    Parameters:
    - A: Coefficient matrix (n, n)
    - b: Right-hand side vector/matrix (n,) or (n, m)
    - tol: Convergence tolerance
    - maxiter: Maximum number of iterations
    - restart: Restart interval

    Returns:
    - x: Solution vector/matrix
    - success: Whether convergence was achieved,

    Note:
        Since the current RBF matrix is dense, we use diagonalization rather than ILU preprocessing.
        - If we convert the dense matrix to a csc_matrix and apply spilu, nearly all elements become non-zero,
          which causes the algorithm to fail from a memory and time perspective (with n ≈ 15,783, the number of elements in A is ≈ 2.5e8)
        - After introducing a compact basis RBF (when A becomes sparse), ILU preprocessing is effective.

        If the matrix size exceeds GMRES_MAX_MATRIX_SIZE, the algorithm immediately falls back.
    """
    n = A.shape[0]
    b_is_matrix = b.ndim == 2

    # Dense Matrix Size Check: Fall back immediately if the matrix is too large
    if n > GMRES_MAX_MATRIX_SIZE:
        print(f"GMRES: The matrix size {n} exceeds the limit {GMRES_MAX_MATRIX_SIZE} - Falling back to the direct method.")
        print(f"  (Converting a dense matrix to a sparse matrix is impractical in terms of memory and time when n > {GMRES_MAX_MATRIX_SIZE}).")
        return None, False

    try:
        # Diagonalization (safe and effective for dense matrices)
        # Do not use ILU diagonalization because it is impractical for dense matrices
        diag = np.diag(A)
        # Replace small diagonal elements with 1 to avoid division by zero
        diag = np.where(np.abs(diag) < 1e-10, 1.0, diag)

        # Diagonal preprocessing: M^{-1} = diag(A)^{-1}
        # Safe for dense matrices and more efficient than ILU
        diag_inv = 1.0 / diag

        def preconditioner(x):
            return diag_inv * x

        M = LinearOperator((n, n), matvec=preconditioner, dtype=A.dtype)

        if b_is_matrix:
            # When there are multiple right-hand side vectors (x, y, z components)
            m = b.shape[1]
            x = np.zeros_like(b)
            all_converged = True

            for i in range(m):
                x_i, info = gmres(A, b[:, i], M=M, tol=tol, restart=restart, maxiter=maxiter)
                x[:, i] = x_i
                if info != 0:
                    all_converged = False

            return x, all_converged
        else:
            x, info = gmres(A, b, M=M, tol=tol, restart=restart, maxiter=maxiter)
            return x, (info == 0)

    except Exception as e:
        print(f"Error during GMRES processing: {e}")
        return None, False


def calculate_optimal_batch_size(num_control_pts: int, max_workers: int,
                                  available_memory_gb: float = None) -> int:
    """
    Calculating the Optimal Batch Size Considering Memory Constraints

    Parameters:
    - num_control_pts: Number of control points
    - max_workers: Number of workers
    - available_memory_gb: Available memory (GB); if None, it is detected automatically

    Returns:
    - Optimal batch size

    Note:
        If an OOM error occurs, consider the following adjustments:
        - MEMORY_USAGE_RATIO (0.5): Lower the percentage of available memory used (e.g., 0.3)
        - MAX_BATCH_SIZE (20000): Lower the upper limit (e.g., 10000)
        - MIN_BATCH_SIZE (1000): Adjust the lower limit (trade-off with processing speed)
    """
    # Adjustable constants (adjust these if an OOM occurs)
    MEMORY_USAGE_RATIO = 0.5  # Utilization rate of available memory (safety margin)
    MIN_BATCH_SIZE = 1000     # Lower limit (Too small increases communication overhead)
    MAX_BATCH_SIZE = 20000    # Upper limit (Too large risks memory fragmentation)

    # Get available memory
    if available_memory_gb is None:
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / 1024**3
        else:
            available_memory_gb = 8.0  # Default: 8 GB

    # Estimated memory usage per batch
    # - Distance matrix: batch_size × num_control_pts × 4 bytes (float32)
    # - RBF values: batch_size × num_control_pts × 4 bytes (float32)
    # - Polynomial terms: batch_size × 4 × 4 bytes (float32)
    # - Results: batch_size × 3 × 4 bytes (float32)
    bytes_per_vertex = num_control_pts * 4 * 2 + 4 * 4 + 3 * 4  # Based on float32

    # Use the specified memory usage ratio
    target_bytes = available_memory_gb * MEMORY_USAGE_RATIO * 1024**3

    # Divide by the number of workers
    bytes_per_worker = target_bytes / max(1, max_workers)

    # Calculate the optimal batch size
    optimal_batch = int(bytes_per_worker / bytes_per_vertex)

    # Set upper and lower bounds
    result = max(MIN_BATCH_SIZE, min(optimal_batch, MAX_BATCH_SIZE))

    print(f"Calculating batch size dynamically: "
          f"Number of control points={num_control_pts}, number of workers={max_workers}, "
          f"Available memory={available_memory_gb:.1f} GB → batch_size={result}")

    return result


def smooth_step(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """
    Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    """
    # Clamp x to the range [0, 1]
    x = np.maximum(0, np.minimum(1, (x - edge0) / (edge1 - edge0)))
    
    # Apply the smooth step formula: 3x^2 - 2x^3
    return x * x * (3 - 2 * x)


def compute_distances_batch(batch_data: Dict[str, Any]) -> Tuple[int, int, np.ndarray]:
    """
    Batch Processing for Distance Calculation (for Multi-Processing)
    Improve memory efficiency
    """
    start_idx = batch_data['start_idx']
    end_idx = batch_data['end_idx']
    batch_targets = batch_data['batch_targets']
    
    # Retrieve KDTree information and construct a new KDTree (for memory efficiency)
    source_vertices = batch_data['source_vertices']
    kdtree = KDTree(source_vertices)
    
    # Search for nearest neighbors using the KDTree
    distances, _ = kdtree.query(batch_targets)
    
    # Free memory immediately after use
    del kdtree
    
    return start_idx, end_idx, distances


def compute_distances_to_source_mesh(target_vertices: np.ndarray, source_vertices: np.ndarray, 
                                   batch_size: int = 5000, max_workers: int = None) -> np.ndarray:
    """
    Calculates the distance from each vertex of the target mesh to the nearest vertex of the source mesh
    Uses KDTree and parallel processing to improve speed and memory efficiency
    
    Parameters:
    - target_vertices: Array of target vertices
    - source_vertices: Source vertex array  
    - batch_size: Batch size (default: 5000; reduced for memory efficiency)
    - max_workers: Maximum number of workers (automatically set if None)
    
    Returns:
    - Distance array
    """
    num_target = len(target_vertices)
    distances = np.zeros(num_target, dtype=DEFAULT_DTYPE)
    
    # Start memory monitoring
    memory_monitor = MemoryMonitor()
    
    # Calculate the optimal number of workers
    if max_workers is None:
        max_workers = get_optimal_worker_count(num_target, memory_monitor)
    
    print(f"Parallel calculation of the distance from each vertex to its nearest neighbor in progress... (Number of vertices: {num_target:,}, Number of workers: {max_workers})")
    
    # Do not parallelize for small datasets
    if num_target <= batch_size:
        print("Executing as a single process due to small dataset...")
        kdtree = KDTree(source_vertices)
        distances, _ = kdtree.query(target_vertices)
        print("Distance calculation complete")
        return distances
    
    # Dynamically adjust batch size
    memory_increase = memory_monitor.get_memory_increase()
    if memory_increase > 0.5:  # If memory has increased by 500MB or more
        batch_size = memory_monitor.get_recommended_batch_size(batch_size, memory_increase)
        print(f"Adjusting batch size based on memory usage: {batch_size}")
    
    # Prepare batch data
    batch_tasks = []
    for i in range(0, num_target, batch_size):
        end_idx = min(i + batch_size, num_target)
        batch_targets = target_vertices[i:end_idx].copy()  # Create a copy to improve memory efficiency
        
        batch_data = {
            'start_idx': i,
            'end_idx': end_idx,
            'batch_targets': batch_targets,
            'source_vertices': source_vertices  # Pass the source vertex instead of the KDTree
        }
        batch_tasks.append(batch_data)
    
    print(f"The distance calculation will be processed in {len(batch_tasks)} batches using multi-process processing.")
    
    # Calculating distance using parallel processing
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(compute_distances_batch, batch_data): batch_data for batch_data in batch_tasks}
        
        for future in as_completed(future_to_batch):
            try:
                start_idx, end_idx, batch_distances = future.result()
                distances[start_idx:end_idx] = batch_distances
                
                processed_count += (end_idx - start_idx)
                progress_percent = (processed_count / num_target) * 100
                
                # Progress display in the process pool (memory monitoring is independent for each process)
                if processed_count % (batch_size * 5) == 0 or processed_count == num_target:
                    if memory_monitor.enabled:
                        current_memory = memory_monitor.get_memory_usage()
                        print(f"Distance Calculation Progress: {processed_count:,}/{num_target:,} Vertices processed ({progress_percent:.1f}%) [Main Process Memory: {current_memory:.1f} GB]")
                    else:
                        print(f"Distance Calculation Progress: {processed_count:,}/{num_target:,} Vertices processed ({progress_percent:.1f}%)")
                
            except Exception as exc:
                batch_data = future_to_batch[future]
                print(f"An error occurred while calculating the range {batch_data['start_idx']}-{batch_data['end_idx']}: {exc}")
                print("Stack trace:")
                traceback.print_exc()
                raise exc
    
    print("The distance calculation is complete")
    return distances


def falloff_displacements(target_vertices: np.ndarray, target_displacements: np.ndarray, 
                         source_vertices: np.ndarray, max_workers: int = None) -> List[np.ndarray]:
    """
    Apply a falloff to displacement based on distance
    """
    num_vertices = len(target_vertices)
    
    # Calculate the distance from each vertex to the nearest vertex of the source mesh
    print("Calculating distance to the source mesh...")
    distances = compute_distances_to_source_mesh(target_vertices, source_vertices, 
                                               batch_size=5000, max_workers=max_workers)
    
    # Distance-based weighting
    distances = np.maximum(distances - 0.015, 0.0)
    weights = np.minimum(1.0, smooth_step(distances * 4.0, 0.0, 1.0))

    final_displacements = []
    
    for i in range(num_vertices):
        if weights[i] > 0:
            # Apply distance-based weighting
            blend_factor = weights[i]
            next_displacement = (1.0 - blend_factor) * target_displacements[i]
        else:
            next_displacement = target_displacements[i]
        
        final_displacements.append(next_displacement)
    
    return final_displacements


def process_vertex_batch(batch_data: Dict[str, Any]) -> Tuple[int, int, np.ndarray]:
    """
    Function to process vertex batches (for multi-process)
    Improves memory efficiency
    
    Returns:
        Tuple[start_idx, end_idx, displacements]
    """
    start_idx = batch_data['start_idx']
    end_idx = batch_data['end_idx']
    batch_world_vertices = batch_data['batch_world_vertices']
    source_control_points = batch_data['source_control_points']
    rbf_weights = batch_data['rbf_weights']
    poly_weights = batch_data['poly_weights']
    epsilon = batch_data['epsilon']
    dim = batch_data['dim']
    
    current_batch_size = end_idx - start_idx
    
    try:
        # Calculate the distance between the target vertex and the control point
        # Use the JIT version if Numba is available (3-5 times faster)
        batch_dists = cdist_fast(batch_world_vertices, source_control_points, 'sqeuclidean')
        batch_phi = np.sqrt(batch_dists + DEFAULT_DTYPE(epsilon**2))

        # Calculating Polynomial Terms (Unified to DEFAULT_DTYPE)
        batch_P = np.ones((current_batch_size, dim + 1), dtype=DEFAULT_DTYPE)
        batch_P[:, 1:] = batch_world_vertices

        # Calculate the displacement of each target vertex
        batch_displacements = np.dot(batch_phi, rbf_weights) + np.dot(batch_P, poly_weights)
        
        # Free up memory immediately after use
        del batch_dists, batch_phi, batch_P
        
        return start_idx, end_idx, batch_displacements

    except Exception as e:
        print(f"An error occurred during batch processing: {e}")
        raise


def process_vertex_batch_thread(args: Tuple) -> Tuple[int, int, np.ndarray]:
    """
    Function for Processing Batches of Vertices (for ThreadPoolExecutor)

    Since ThreadPoolExecutor shares memory, it can efficiently process large arrays
    by passing by reference without copying them. Because NumPy's BLAS operations release the GIL,
    parallel performance can be achieved even with ThreadPoolExecutor.

    Parameters:
        args: (start_idx, end_idx, target_world_vertices, source_control_points,
               rbf_weights, poly_weights, epsilon, dim)

    Returns:
        Tuple[start_idx, end_idx, displacements]
    """
    (start_idx, end_idx, target_world_vertices, source_control_points,
     rbf_weights, poly_weights, epsilon, dim) = args

    batch_world_vertices = target_world_vertices[start_idx:end_idx]
    current_batch_size = end_idx - start_idx

    try:
        # Calculate the distance between the target vertex and the control point
        # Use the JIT version if Numba is available (3 - 5 times faster)
        batch_dists = cdist_fast(batch_world_vertices, source_control_points, 'sqeuclidean')
        batch_phi = np.sqrt(batch_dists + DEFAULT_DTYPE(epsilon**2))

        # Calculating polynomial terms (unified to DEFAULT_DTYPE)
        batch_P = np.ones((current_batch_size, dim + 1), dtype=DEFAULT_DTYPE)
        batch_P[:, 1:] = batch_world_vertices

        # Calculating polynomial terms (unified to DEFAULT_DTYPE)
        batch_displacements = np.dot(batch_phi, rbf_weights) + np.dot(batch_P, poly_weights)

        # Memory Release
        del batch_dists, batch_phi, batch_P

        return start_idx, end_idx, batch_displacements

    except Exception as e:
        print(f"An error occurred during batch processing: {e}")
        raise


def rbf_interpolation_multithread(source_control_points: np.ndarray, 
                                 source_control_points_deformed: np.ndarray, 
                                 target_world_vertices: np.ndarray,
                                 epsilon: float = 1.0, 
                                 batch_size: int = 10000,  # Reduce the default batch size
                                 max_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates new positions for the target mesh using multi-process RBF interpolation
    Improves memory efficiency
    
    Parameters:
    - source_control_points: Selected control points of the source mesh (reference positions) - world coordinates
    - source_control_points_deformed: Control points of the source mesh after deformation by shape keys - world coordinates
    - target_world_vertices: Vertices of the target mesh (world coordinates)
    - epsilon: RBF parameter
    - batch_size: Number of target vertices processed at a time (reduces the default value)
    - max_workers: Maximum number of workers (if None, based on the number of CPU cores)
    
    Returns:
    - Displacement Vector
    - Final displacement after applying the falloff
    """
    # Start memory monitoring
    memory_monitor = MemoryMonitor()
    total_vertices = len(target_world_vertices)
    
    # Calculate the optimal number of workers
    if max_workers is None:
        max_workers = get_optimal_worker_count(total_vertices, memory_monitor)
    
    print(f"Starting multi-process RBF interpolation (number of workers: {max_workers}, initial memory: {memory_monitor.initial_memory:.1f}GB, dtype: {DEFAULT_DTYPE.__name__}）")

    # Convert input to float32 (for better memory efficiency)
    source_control_points = source_control_points.astype(DEFAULT_DTYPE)
    source_control_points_deformed = source_control_points_deformed.astype(DEFAULT_DTYPE)
    target_world_vertices = target_world_vertices.astype(DEFAULT_DTYPE)

    # Calculate the displacement vector (post-deformation position - original position)
    displacements = source_control_points_deformed - source_control_points
    
    # Calculate the scaling factor: Use a value based on the standard deviation of the distance
    if epsilon <= 0:
        # Calculate an appropriate epsilon based on the average distance
        dists = cdist_fast(source_control_points, source_control_points, 'euclidean')
        mean_dist = np.mean(dists[dists > 0])
        epsilon = mean_dist  # Use the average distance as epsilon
        print(f"Automatically calculated epsilon: {epsilon}")

    # Calculate the distance matrix between control points (use the JIT version if Numba is available)
    print(f"RBFCalculating the array... (Numba: {'Valid' if NUMBA_AVAILABLE else 'Invalid'})）")
    dist_matrix = cdist_fast(source_control_points, source_control_points, 'sqeuclidean')

    # Calculate the RBF matrix
    phi = np.sqrt(dist_matrix + DEFAULT_DTYPE(epsilon**2))

    num_pts, dim = source_control_points.shape
    P = np.ones((num_pts, dim + 1), dtype=DEFAULT_DTYPE)
    P[:, 1:] = source_control_points  # Extended matrix for polynomial terms

    # Build a fully linear system
    A = np.zeros((num_pts + dim + 1, num_pts + dim + 1), dtype=DEFAULT_DTYPE)
    A[:num_pts, :num_pts] = phi
    A[:num_pts, num_pts:] = P
    A[num_pts:, :num_pts] = P.T

    # Set the right side
    b = np.zeros((num_pts + dim + 1, dim), dtype=DEFAULT_DTYPE)
    b[:num_pts] = displacements
    
    # Find the solution
    print(f"I am solving a linear system (matrix size: {A.shape[0]}x{A.shape[1]}, dtype: {A.dtype}）...")
    solve_start = time.time()
    x = None

    # Trying out the GMRES iterative solver (experimental feature; enable with USE_GMRES_SOLVER=True)
    if USE_GMRES_SOLVER:
        print("Testing the GMRES iterative solver... (Experimental feature)")
        gmres_start = time.time()
        x_gmres, gmres_success = solve_with_gmres(A, b)
        gmres_time = time.time() - gmres_start

        if gmres_success and x_gmres is not None:
            x = x_gmres.astype(DEFAULT_DTYPE)
            print(f"Testing the GMRES iterative solver... (Experimental feature)")
        else:
            print(f"GMRES convergence failed ({gmres_time:.2f} seconds) - Falling back to the direct method")

    # Direct Method (LU Decomposition)
    if x is None:
        try:
            # Attempt the standard solution (DEFAULT_DTYPE precision)
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If the operation fails with float32, promote to float64 and retry
            if A.dtype == np.float32:
                print("Solution failed with float32—I'll upgrade to float64 and try again.")
                try:
                    A_f64 = A.astype(np.float64)
                    b_f64 = b.astype(np.float64)
                    x = np.linalg.solve(A_f64, b_f64).astype(DEFAULT_DTYPE)
                    del A_f64, b_f64
                except np.linalg.LinAlgError:
                    # If it fails even with float64, apply regularization and use the pseudo-inverse (maximum stability with float64)
                    print("Fails even with float64 - Apply regularization (float64 precision)")
                    A_f64 = A.astype(np.float64)
                    b_f64 = b.astype(np.float64)
                    reg_f64 = np.eye(A.shape[0], dtype=np.float64) * 1e-6
                    x = np.linalg.lstsq(A_f64 + reg_f64, b_f64, rcond=None)[0].astype(DEFAULT_DTYPE)
                    del A_f64, b_f64, reg_f64
            else:
                # If the matrix is already of type float64, regularize it and use the pseudo-inverse
                print("The matrix is singular - applying regularization")
                reg = np.eye(A.shape[0], dtype=np.float64) * 1e-6
                x = np.linalg.lstsq(A + reg, b, rcond=None)[0]

    solve_time = time.time() - solve_start
    print(f"Linear system solved ({solve_time:.2f} seconds)")

    # Extract weights
    rbf_weights = x[:num_pts]
    poly_weights = x[num_pts:]
    
    # Remove unnecessary variables to free up memory
    del dist_matrix, phi, A, b, x
    
    # Check memory usage and adjust the batch size
    memory_increase = memory_monitor.get_memory_increase()
    if memory_increase > 1.0:  # If the increase is 1 GB or more
        batch_size = memory_monitor.get_recommended_batch_size(batch_size, memory_increase)
        print(f"Adjust the batch size based on memory usage: {batch_size}")
    
    # Initialize the array to store the results
    target_displacements = np.zeros_like(target_world_vertices, dtype=DEFAULT_DTYPE)
    
    # Hybrid Parallelization（P2-3）: ThreadPoolExecutor or ProcessPoolExecutor
    if USE_HYBRID_PARALLELIZATION:
        # ThreadPoolExecutor Usage: Batch tasks in tuple format
        # Since memory sharing is possible, pass by reference instead of copying data
        batch_tasks_thread = []
        for batch_start in range(0, total_vertices, batch_size):
            batch_end = min(batch_start + batch_size, total_vertices)
            # Tuple format: (start_idx, end_idx, target_world_vertices, source_control_points,
            #              rbf_weights, poly_weights, epsilon, dim)
            batch_tasks_thread.append((
                batch_start, batch_end, target_world_vertices, source_control_points,
                rbf_weights, poly_weights, epsilon, dim
            ))

        # ThreadPoolExecutor enables parallelism by releasing the GIL during NumPy BLAS operations
        # Numba JIT functions also release the GIL when 'nogil=True' is specified
        # Prevents oversubscription by limiting the number of threads to the number of CPU cores
        cpu_count = os.cpu_count() or 4
        thread_workers = min(cpu_count, max_workers * 2)  # Threads are lighter than processes, but there is a limit
        print(f"Process the vertices of the target mesh in batches of {len(batch_tasks_thread)}"
              f"Processing using a thread pool (all {total_vertices} vertices, number of workers: {thread_workers})")
        print("Hybrid Parallelization Mode: ThreadPoolExecutor（Utilizing NumPy GIL Release）")

        processed_count = 0
        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            future_to_idx = {executor.submit(process_vertex_batch_thread, task): task[0]
                            for task in batch_tasks_thread}

            for future in as_completed(future_to_idx):
                try:
                    start_idx, end_idx, batch_displacements = future.result()
                    target_displacements[start_idx:end_idx] = batch_displacements

                    processed_count += (end_idx - start_idx)
                    progress_percent = (processed_count / total_vertices) * 100

                    if processed_count % (batch_size * 20) == 0 or processed_count == total_vertices:
                        if memory_monitor.enabled:
                            current_memory = memory_monitor.get_memory_usage()
                            print(f"Progress: {processed_count}/{total_vertices} vertices processed."
                                  f"({progress_percent:.1f}%) [Memory: {current_memory:.1f} GB]")
                        else:
                            print(f"Progress: {processed_count}/{total_vertices} vertices processed ({progress_percent:.1f}%)")

                except Exception as exc:
                    batch_start_idx = future_to_idx[future]
                    print(f"An error occurred at batch start position {batch_start_idx}: {exc}")
                    print("Stack trace:")
                    traceback.print_exc()
                    raise exc

        print("ThreadPool processing is complete")

    else:
        # For ProcessPoolExecutor: Batch tasks in dictionary format (traditional method)
        batch_tasks = []
        for batch_start in range(0, total_vertices, batch_size):
            batch_end = min(batch_start + batch_size, total_vertices)
            batch_world_vertices = target_world_vertices[batch_start:batch_end].copy()  # Create a copy

            batch_data = {
                'start_idx': batch_start,
                'end_idx': batch_end,
                'batch_world_vertices': batch_world_vertices,
                'source_control_points': source_control_points,
                'rbf_weights': rbf_weights,
                'poly_weights': poly_weights,
                'epsilon': epsilon,
                'dim': dim
            }
            batch_tasks.append(batch_data)

        print(f"Process the vertices of the target mesh in batches of {len(batch_tasks)} using multiprocessing (all {total_vertices} vertices)")

        # Limit the number of BLAS threads immediately before starting the ProcessPoolExecutor
        # Since 'np.linalg.solve() ' has already completed, this limit is applied to prevent oversubscription in parallel processing
        # No limit is necessary when 'max_workers == 1' (in low-memory mode or similar scenarios where a single worker is used, all available threads are utilized)
        if max_workers == 1:
            print("Single-worker mode: No BLAS thread limit (full thread utilization)")
        else:
            blas_threads = '2'
            os.environ['OMP_NUM_THREADS'] = blas_threads
            os.environ['OPENBLAS_NUM_THREADS'] = blas_threads
            os.environ['MKL_NUM_THREADS'] = blas_threads
            os.environ['VECLIB_MAXIMUM_THREADS'] = blas_threads
            os.environ['NUMEXPR_NUM_THREADS'] = blas_threads
            print(f"The number of BLAS threads has been limited to {blas_threads} (before starting the ProcessPoolExecutor; number of workers: {max_workers})")

        # Multiprocessing
        processed_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Processing batches in parallel
            future_to_batch = {executor.submit(process_vertex_batch, batch_data): batch_data for batch_data in batch_tasks}

            for future in as_completed(future_to_batch):
                try:
                    start_idx, end_idx, batch_displacements = future.result()
                    target_displacements[start_idx:end_idx] = batch_displacements

                    processed_count += (end_idx - start_idx)
                    progress_percent = (processed_count / total_vertices) * 100

                    # Progress display in the process pool (memory monitoring is independent for each process)
                    if processed_count % (batch_size * 20) == 0 or processed_count == total_vertices:
                        if memory_monitor.enabled:
                            current_memory = memory_monitor.get_memory_usage()
                            print(f"Progress: {processed_count}/{total_vertices} vertices processed ({progress_percent:.1f}%) [Main process memory: {current_memory:.1f} GB]")
                        else:
                            print(f"Progress: {processed_count}/{total_vertices} vertices processed ({progress_percent:.1f}%)")

                except Exception as exc:
                    batch_data = future_to_batch[future]
                    print(f"An error occurred in batch {batch_data['start_idx']}-{batch_data['end_idx']}: {exc}")
                    print("Stack trace:")
                    traceback.print_exc()
                    raise exc

        print("Multi-process processing has been completed")

    # Apply falloff processing
    print("Applying falloff...")
    final_displacements = falloff_displacements(
        target_world_vertices, 
        target_displacements, 
        source_control_points,
        max_workers
    )

    # Note: The number of BLAS threads remains fixed at 2 (no need to reset it after processing)

    if memory_monitor.enabled:
        final_memory = memory_monitor.get_memory_usage()
        print(f"Final memory usage: {final_memory:.1f} GB (Increase: {memory_monitor.get_memory_increase():.1f} GB)")
    else:
        print("Final memory usage: Cannot be displayed because psutil is not available.")
    
    return target_displacements, np.array(final_displacements)


def process_temp_file(temp_file_path: str, max_workers: int = None,
                      old_version: bool = False, batch_size: int = None) -> str:
    """
    Process temporary files to perform multi-process RBF interpolation

    Parameters:
    - temp_file_path: Path to the temporary data file
    - max_workers: Maximum number of workers
    - old_version: Whether to save in the old version format
    - batch_size: Batch size (dynamically optimized if None)
    
    Returns:
    - Path to the output file
    """
    print(f"Loading temporary data file: {temp_file_path}")
    
    # Load temporary data
    data = np.load(temp_file_path, allow_pickle=True)
    
    # Compatible with both the new and old formats
    if 'all_field_world_vertices' in data:
        # New Format: Fields by Step
        all_field_world_vertices = data['all_field_world_vertices']
        print("Detected a new format for temporary data (fields for each step)")
    elif 'field_world_vertices' in data:
        # Old format: Single field
        field_world_vertices = data['field_world_vertices']
        num_steps_temp = int(data['num_steps'])
        all_field_world_vertices = [field_world_vertices for _ in range(num_steps_temp)]
        print("Detect temporary data in the old format (duplicate a single field)")
    else:
        raise ValueError("Field data not found ")
    
    field_world_matrix = data['field_world_matrix']
    all_step_data = data['all_step_data']
    source_world_matrix = data['source_world_matrix']
    epsilon = float(data['epsilon'])
    num_steps = int(data['num_steps'])
    invert = bool(data['invert'])
    source_avatar_name = str(data['source_avatar_name'])
    target_avatar_name = str(data['target_avatar_name'])
    source_shape_key_name = str(data['source_shape_key_name'])
    save_shape_key_mode = bool(data['save_shape_key_mode'])
    
    print(f"Loading complete:")
    print(f"  Number of steps: {num_steps}")
    for step in range(num_steps):
        print(f"  Step {step+1} Number of field vertices: {len(all_field_world_vertices[step])}")
    print(f"  Inverse transformation: {invert}")
    print(f"  Epsilon: {epsilon}")

    # Calculate the optimal number of workers (using data from the first step)
    first_step_data = all_step_data[0]
    num_control_pts = len(first_step_data['control_points_original'])
    total_vertices = len(all_field_world_vertices[0])
    memory_monitor = MemoryMonitor()
    if max_workers is None:
        max_workers = get_optimal_worker_count(total_vertices, memory_monitor)

    # Determining the batch size (user-specified value takes precedence; if not specified, calculated dynamically)
    if batch_size is not None:
        optimal_batch_size = batch_size
        print(f"  Batch size: {optimal_batch_size} (User-specified)")
    else:
        optimal_batch_size = calculate_optimal_batch_size(num_control_pts, max_workers)
        print(f"  Batch size: {optimal_batch_size} (Dynamic Calculations)")

    # Calculate the displacement for each step
    all_displacements = []
    all_target_world_vertices = []
    
    for step in range(num_steps):
        step_data = all_step_data[step]
        
        print(f"\n=== Processing step {step+1}/{num_steps} ===")
        print(f"Shape Key Value: {step_data['step_value']}")
        
        source_control_points = step_data['control_points_original']
        source_control_points_deformed = step_data['control_points_deformed']
        
        # Get the fields for the corresponding step
        current_field_vertices = all_field_world_vertices[step]
        print(f"Number of field vertices used: {len(current_field_vertices)}")

        print(f"current_field_vertices type: {type(current_field_vertices)}")
        print(f"current_field_vertices shape: {current_field_vertices.shape}")
        print(f"current_field_vertices Data type: {current_field_vertices.dtype}")
        print(f"current_field_vertices Number of elements: {len(current_field_vertices)}")
        
        # Check the maximum displacement
        displacements = source_control_points_deformed - source_control_points
        max_disp = np.max(np.linalg.norm(displacements, axis=1))
        print(f"Maximum displacement at the control point: {max_disp}")

        # Perform multi-process RBF interpolation (using dynamically optimized batch sizes)
        target_displacements, final_displacements = rbf_interpolation_multithread(
            source_control_points,
            source_control_points_deformed,
            current_field_vertices,
            epsilon,
            batch_size=optimal_batch_size,
            max_workers=max_workers
        )
        
        all_target_world_vertices.append(current_field_vertices.copy())
        all_displacements.append(final_displacements)
        
        print(f"Displacement calculation for step {step+1} complete")
    
    # Generate the output file path
    base_dir = os.path.dirname(temp_file_path)
    
    if save_shape_key_mode:
        direction_suffix = "_inv" if invert else ""
        output_path = os.path.join(base_dir, f"deformation_{source_avatar_name}_shape_{source_shape_key_name}{direction_suffix}.npz")
    else:
        direction_suffix = "_inv" if invert else ""
        output_path = os.path.join(base_dir, f"deformation_{source_avatar_name}_to_{target_avatar_name}{direction_suffix}.npz")
    
    # Save results
    save_field_data_multi_step(
        field_world_matrix,
        output_path,
        all_target_world_vertices,
        all_displacements,
        num_steps,
        old_version=old_version,
        enable_x_mirror=data.get('enable_x_mirror', False)
    )
    
    print(f"The results have been saved: {output_path}")
    return output_path


def save_field_data_multi_step(world_matrix, filepath, 
                              all_field_points, 
                              all_delta_positions, 
                              num_steps,
                              old_version=False,
                              enable_x_mirror=True):
    """
    Save the difference between the before and after states of a multi-step deformation field directly as a NumPy array
    If 'enable_x_mirror' is enabled, only data with X coordinates of 0 or greater is saved
    """
    
    kdtree_query_k = 27
    
    # Add RBF interpolation parameters
    rbf_epsilon = 0.00001  # Fixed value
    rbf_smoothing = 0.0    # Smoothing parameter
   
    # Save data
    if old_version:
        np.savez(filepath,
                field_points=all_field_points[0],
                delta_positions=all_delta_positions[0],
                num_steps=num_steps,
                world_matrix=world_matrix,
                kdtree_query_k=kdtree_query_k,
                rbf_epsilon=rbf_epsilon,
                rbf_smoothing=rbf_smoothing)
    else:
        # When 'enable_x_mirror' is enabled, only data with X-coordinates of 0 or greater is filtered
        if enable_x_mirror:
            filtered_field_points = []
            filtered_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step]
                delta_positions = all_delta_positions[step]
                
                if len(field_points) > 0:
                    # Get the index where the X-coordinate is 0 or greater
                    x_positive_mask = field_points[:, 0] >= 0.0
                    filtered_field = field_points[x_positive_mask]
                    filtered_delta = delta_positions[x_positive_mask]
                    
                    filtered_field_points.append(filtered_field.astype(np.float32))
                    filtered_delta_positions.append(filtered_delta.astype(np.float32))
                    
                    print(f"Step {step+1}: Original number of vertices {len(field_points)} → After filtering {len(filtered_field)}")
                else:
                    filtered_field_points.append(np.array([]))
                    filtered_delta_positions.append(np.array([]))
                    print(f"Step {step+1}: Number of field vertices: 0")
        else:
            # If the mirror is disabled, only cast to float32
            filtered_field_points = []
            filtered_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step]
                delta_positions = all_delta_positions[step]
                
                if len(field_points) > 0:
                    filtered_field_points.append(field_points.astype(np.float32))
                    filtered_delta_positions.append(delta_positions.astype(np.float32))
                    print(f"Step {step+1}: Number of vertices {len(field_points)} (without mirror filter)")
                else:
                    filtered_field_points.append(np.array([]))
                    filtered_delta_positions.append(np.array([]))
                    print(f"Step {step+1}: Number of field vertices: 0")
        
        np.savez(filepath,
                all_field_points=np.array(filtered_field_points, dtype=object),
                all_delta_positions=np.array(filtered_delta_positions, dtype=object),
                num_steps=num_steps,
                world_matrix=world_matrix,
                kdtree_query_k=kdtree_query_k,
                rbf_epsilon=rbf_epsilon,
                rbf_smoothing=rbf_smoothing,
                enable_x_mirror=enable_x_mirror)
        
    print(f"Deformation Field difference data has been saved: {filepath}")
    print(f"Number of steps: {num_steps}")
    if old_version:
        print(f"Step 1: Number of vertices {len(all_field_points[0])}")
    else:
        for step in range(num_steps):
            if step < len(filtered_field_points):
                print(f"Step {step+1}: Number of vertices {len(filtered_field_points[step])}")
    print(f"RBF function: multi_quadratic_biharmonic, epsilon: {rbf_epsilon}, smoothing: {rbf_smoothing}")


def process_multiple_temp_files(temp_file_pattern: str, max_workers: int = None,
                                old_version: bool = False, batch_size: int = None) -> List[str]:
    """
    Processing Multiple Temporary Files (Forward and Reverse)

    Parameters:
    - temp_file_pattern: Pattern for temporary data files (without the _inv suffix)
    - max_workers: Maximum number of workers
    - old_version: Whether to save in the old version format
    - batch_size: Batch size (dynamically optimized if None)

    Returns:
    - A list of output file paths
    """
    output_paths = []
    
    # Generate paths for forward and reverse files
    temp_files = []
    
    # Generate forward and reverse file paths from the base filename
    base_path = temp_file_pattern
    if base_path.endswith('.npz'):
        base_path = base_path[:-4]
    
    forward_file = f"{base_path}.npz"
    inverse_file = f"{base_path}_inv.npz"
    
    # Include only existing files in the processing
    for temp_file in [forward_file, inverse_file]:
        if os.path.exists(temp_file):
            temp_files.append(temp_file)
        else:
            print(f"Warning: File not found: {temp_file}")
    
    if not temp_files:
        print("Error: The file to be processed cannot be found.")
        return output_paths
    
    total_start_time = time.time()
    
    for i, temp_file in enumerate(temp_files):
        print(f"\n{'='*60}")
        print(f"File {i+1}/{len(temp_files)}: {os.path.basename(temp_file)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            output_path = process_temp_file(temp_file, max_workers, old_version, batch_size)
            output_paths.append(output_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"File {i+1} processed: {processing_time:.2f} seconds")
            print(f"Output file: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"File {i+1} An error has occurred: {e}")
            print("Stack trace:")
            traceback.print_exc()
            continue
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"{'='*60}")
    print(f"Number of processed files: {len(output_paths)}/{len(temp_files)}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    if output_paths:
        print("Output file:")
        for path in output_paths:
            print(f"  - {os.path.basename(path)}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description='External multi-process processing of RBF transformations (with memory optimization) ')
    parser.add_argument('temp_file', help='Path to temporary data files (base filename; _inv files are also processed automatically) ')
    parser.add_argument('--max-workers', type=int, default=16, 
                       help='Maximum number of processes (Default: Automatically set based on the number of CPU cores and memory capacity) ')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (Optimized dynamically if not specified; if specified, that value takes precedence) ')
    parser.add_argument('--single-file', action='store_true',
                       help='Process a single file only (do not automatically detect _inv files) ')
    parser.add_argument('--memory-limit', type=float, default=None,
                       help='Maximum memory usage (in GB; default: no limit) ')
    parser.add_argument('--low-memory', action='store_true',
                       help='Low-memory mode (automatically limits batch size and number of workers) ')
    parser.add_argument('--old-version', action='store_true',
                       help='Save in the old file format (for compatibility) ')
    parser.add_argument('--use-threadpool', action='store_true',
                       help='Hybrid Parallelization: Using ThreadPoolExecutor for RBF Evaluation (Experimental) ')
    parser.add_argument('--use-gmres', action='store_true',
                       help='Use the GMRES iterative solver (experimental; fall back to the direct method if convergence is not achieved) ')

    args = parser.parse_args()

    # Setting Experimental Feature Flags
    global USE_HYBRID_PARALLELIZATION, USE_GMRES_SOLVER
    if args.use_threadpool:
        USE_HYBRID_PARALLELIZATION = True
    if args.use_gmres:
        USE_GMRES_SOLVER = True

    if PSUTIL_AVAILABLE:
        set_cpu_affinity()

    # In low-memory mode, adjust the settings (for the process pool)
    if args.low_memory:
        print("Low-memory mode is enabled. This limits the batch size and the number of processes.")
        if args.batch_size is None or args.batch_size > 2000:
            args.batch_size = 2000
        if args.max_workers is None or args.max_workers > 1:
            args.max_workers = 1  # Limited to one per process pool
    
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Numba JIT: {'Enabled (accelerates distance calculation) ' if NUMBA_AVAILABLE else 'Disabled (can be enabled with pip install numba) '}")
    print(f"GMRES iterative solver: {'Enabled (--use-gmres) ' if USE_GMRES_SOLVER else 'Disabled'}")
    print(f"Hybrid parallelization: {'Enabled (--use-threadpool) ' if USE_HYBRID_PARALLELIZATION else 'Disabled (ProcessPoolExecutor) '}")
    # Note: Set the BLAS thread limit immediately before starting the ProcessPoolExecutor.
    # Since the linear system solver (np.linalg.solve) runs before the ProcessPoolExecutor,
    # do not set the limit here; instead, set it within 'multiprocess_rbf_interpolation()'.
    # This maintains the performance of the linear system solver while
    # preventing oversubscription in the ProcessPoolExecutor
    if not USE_HYBRID_PARALLELIZATION:
        print("BLAS thread count: To be limited after solving the linear system")
    
    np.__config__.show()
    
    # Display memory usage information (only when psutil is available)
    if PSUTIL_AVAILABLE:
        memory_info = psutil.virtual_memory()
        print(f"System Memory Information:")
        print(f"  Total memory: {memory_info.total / 1024**3:.1f}GB")
        print(f"  Available memory: {memory_info.available / 1024**3:.1f}GB")
        print(f"  Memory usage: {memory_info.percent:.1f}%")
        if args.memory_limit:
            print(f"  Memory Limit Settings: {args.memory_limit:.1f}GB")
    else:
        print("System memory information: Cannot be displayed because psutil is not available.")
        if args.memory_limit:
            print(f"Memory Limit Settings: {args.memory_limit:.1f}GB")
    
    if args.single_file:
        print(f"Single-file processing mode")
        if not os.path.exists(args.temp_file):
            print(f"Error: Temporary data file not found: {args.temp_file}")
            sys.exit(1)
        
        start_time = time.time()
        
        try:
            output_path = process_temp_file(args.temp_file, args.max_workers,
                                           args.old_version, args.batch_size)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"\n=== Processing complete ===")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Output file: {output_path}")
            
        except Exception as e:
            print(f"An error has occurred: {e}")
            print("Stack trace:")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Batch processing mode (default)
        try:
            print(f"Batch Processing Mode")
            output_paths = process_multiple_temp_files(args.temp_file, args.max_workers,
                                                      args.old_version, args.batch_size)
            
            if not output_paths:
                print("Error: No files were processed")
                sys.exit(1)
            
        except Exception as e:
            print(f"An error has occurred: {e}")
            print("Stack trace:")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main() 
