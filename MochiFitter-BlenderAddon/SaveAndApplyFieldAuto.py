import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler
import os
import sys
import subprocess
import platform
import math
import bmesh
from mathutils.bvhtree import BVHTree
import time
from math import ceil, sqrt
import json
import traceback
import shutil
from typing import Dict, Optional, Tuple, Set
from bpy_extras.io_utils import ExportHelper

# Importing scipy conditionally
try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist, pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not found. Some features will be limited.")
    print("Please use the dependencies reinstall button to install.")

print(f"SciPy available: {SCIPY_AVAILABLE}")

def get_scene_folder():
    """
    Get the folder path of the current Blender scene file
    If the scene is unsaved, use the current directory
    
    Returns:
        str: Folder path
    """
    blend_filepath = bpy.data.filepath
    if blend_filepath:
        return os.path.dirname(blend_filepath)
    else:
        print("Warning: Blend file is not saved. Using current directory.")
        return os.getcwd()

def load_avatar_data(filename="avatar_data.json"):
    """
    Load Avatar Data
    
    Parameters:
        filename (str): The filename of the avatar data JSON file
    
    Returns:
        dict: A dictionary containing the avatar data
    """
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Avatar data file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        avatar_data = json.load(f)
    
    return avatar_data

def normalize_avatar_name_for_filename(name: str) -> str:
    """
    Normalize Avatar Name for File Names (Convert to Lowercase)

    To ensure compatibility with Unity extensions, output file names are standardized to lowercase.
    Since Linux is case-sensitive, files will be treated as separate files if names are not standardized.

    Parameters:
        name (str): Avatar name

    Returns:
        str: Avatar name converted to lowercase

    See: GitHub Issue #64
    """
    return name.lower() if name else ""

def find_field_data_file(scene_folder: str, source_avatar_name: str, target_avatar_name: str = None,
                         source_shape_key_name: str = None, inverse_suffix: str = "") -> Optional[str]:
    """
    Searching for the Path to Deformed Field Data Files (Backward Compatibility)

    Prioritize the new lowercase filename; if it does not exist, fall back to the original filename containing both uppercase and lowercase letters.
    This ensures that files created in older versions can still be used.

    Parameters:
        scene_folder (str): Path to the folder to search
        source_avatar_name (str): Source avatar name
        target_avatar_name (str, optional): Target avatar name (For avatar-to-avatar transformation)
        source_shape_key_name (str, optional): Shape key name (for shape key mode)
        inverse_suffix (str): Inverse transformation suffix ("_inv" or "")

    Returns:
        Optional[str]: File path of the found file; None if not found

    See: GitHub Issue #64, PR #66 review feedback
    """
    # New lowercase filenames (preferred)
    if source_shape_key_name:
        new_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{inverse_suffix}.npz"
        old_filename = f"deformation_{source_avatar_name}_shape_{source_shape_key_name}{inverse_suffix}.npz"
    else:
        new_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{inverse_suffix}.npz"
        old_filename = f"deformation_{source_avatar_name}_to_{target_avatar_name}{inverse_suffix}.npz"

    new_path = os.path.join(scene_folder, new_filename)
    old_path = os.path.join(scene_folder, old_filename)

    # Search for lowercase files first
    if os.path.exists(new_path):
        return new_path
    # Fallback to legacy files with mixed case
    if os.path.exists(old_path):
        print(f"Note: Using legacy filename '{old_filename}' (consider renaming to '{new_filename}')")
        return old_path

    return None

def build_bone_hierarchy(bone_node: dict, bone_parents: Dict[str, str], current_path: list):
    """
    Recursively construct a parent-child mapping from the bone hierarchy

    Parameters:
        bone_node (dict): The current bone node
        bone_parents (Dict[str, str]): A mapping from bone names to parent bone names
        current_path (list): A list of bone names along the current path
    """
    bone_name = bone_node['name']
    if current_path:
        bone_parents[bone_name] = current_path[-1]
    
    current_path.append(bone_name)
    for child in bone_node.get('children', []):
        build_bone_hierarchy(child, bone_parents, current_path)
    current_path.pop()

def get_humanoid_bone_hierarchy(avatar_data: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Extract the hierarchical relationship of Humanoid bones from avatar data

    Parameters:
        avatar_data (dict): Avatar data

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]: 
            (Dictionary mapping bone names to parents, dictionary mapping Humanoid bone names to bone names, dictionary mapping bone names to Humanoid bone names)
    """
    # Building Parent-Child Relationships in Bone
    bone_parents = {}
    build_bone_hierarchy(avatar_data['boneHierarchy'], bone_parents, [])

    # Create a mapping of Humanoid bone names and bone names
    humanoid_to_bone = {bone_map['humanoidBoneName']: bone_map['boneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    bone_to_humanoid = {bone_map['boneName']: bone_map['humanoidBoneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    
    return bone_parents, humanoid_to_bone, bone_to_humanoid

def find_nearest_parent_with_pose(bone_name: str, 
                                bone_parents: Dict[str, str], 
                                bone_to_humanoid: Dict[str, str],
                                pose_data: dict) -> Optional[str]:
    """
    Traverses the parent hierarchy of the specified bone and returns the name of the nearest Humanoid bone that contains pose data

    Parameters:
        bone_name (str): Starting bone name
        bone_parents (Dict[str, str]): Dictionary of bone parent-child relationships
        bone_to_humanoid (Dict[str, str]): Dictionary mapping bone names to Humanoid bone names
        pose_data (dict): Pose data

    Returns:
        Optional[str]: The Humanoid bone name of the found parent; None if not found
    """
    current_bone = bone_name
    while current_bone in bone_parents:
        parent_bone = bone_parents[current_bone]
        if parent_bone in bone_to_humanoid:
            parent_humanoid = bone_to_humanoid[parent_bone]
            if parent_humanoid in pose_data:
                return parent_humanoid
        current_bone = parent_bone
    return None

def save_armature_pose(armature_obj, filename="pose_data.json", avatar_data_file="avatar_data.json"):
    """
    Save the pose of the active Armature's Humanoid bones in world coordinates to a JSON file

    Parameters:
        filename (str): The name of the JSON file to save
        avatar_data_file (str): The name of the avatar data JSON file
    """
    if not armature_obj:
        raise ValueError("No armature object found")
    
    if armature_obj.type != 'ARMATURE':
        raise ValueError(f"Active object '{armature_obj.name}' is not an armature")
    
    # Load avatar data
    avatar_data = load_avatar_data(avatar_data_file)
    
    # Create a mapping from bone names to Humanoid bone names
    _, _, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
    
    # Create the full path to the destination
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    # A dictionary that stores pose bone information
    pose_data = {}
    
    for bone in armature_obj.pose.bones:
        # Skip if not a humanoid bone
        if bone.name not in bone_to_humanoid:
            continue
            
        humanoid_name = bone_to_humanoid[bone.name]
        base_matrix = armature_obj.data.bones[bone.name].matrix_local
        
        # Calculating matrices in world space
        world_matrix = armature_obj.matrix_world @ bone.matrix
        base_world_matrix = armature_obj.matrix_world @ base_matrix
        
        delta_matrix = world_matrix @ base_world_matrix.inverted()
        
        # Calculate the world coordinates of the bone's head
        head_local = armature_obj.data.bones[bone.name].head_local
        head_world = armature_obj.matrix_world @ head_local
        head_world_transformed = armature_obj.matrix_world @ bone.head
        
        # Get location
        location = head_world_transformed - head_world
        
        # Get rotation (convert to Euler angles)
        rotation = delta_matrix.to_euler('XYZ')
        
        # Get the scale
        scale = delta_matrix.to_scale()

        # Store data in a dictionary (using Humanoid bone names as keys)
        # delta_matrix: Required for compatibility with Unity scripts
        # location/rotation/scale: Reference values (for debugging)
        pose_data[humanoid_name] = {
            'delta_matrix': matrix_to_list(delta_matrix),
            'location': [location.x, location.y, location.z],
            'rotation': [math.degrees(rotation.x),
                        math.degrees(rotation.y),
                        math.degrees(rotation.z)],
            'scale': [scale.x, scale.y, scale.z],
            'head_world': [head_world.x, head_world.y, head_world.z],
            'head_world_transformed': [head_world_transformed.x, head_world_transformed.y, head_world_transformed.z]
        }
    
    # Save to a JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=4)
        
    print(f"Pose data saved for humanoid bones to {filepath}")
    return filepath

def clear_humanoid_bone_relations_preserve_pose(armature_obj, avatar_data_file="avatar_data.json"):
    """
    Preserving poses in world space while un-parenting Humanoid bones
    
    Args:
        armature_obj: bpy.types.Object - Armature object
        avatar_data_file (str): Name of the JSON file containing the avatar data
    """
    if armature_obj.type != 'ARMATURE':
        raise ValueError("Selected object must be an armature")
    
    # Load avatar data
    avatar_data = load_avatar_data(avatar_data_file)
    
    # Create a list of humanoid bones
    humanoid_bones = {bone_map['boneName'] for bone_map in avatar_data['humanoidBones']}
    
    # Get the armature data
    armature = armature_obj.data
    
    # Store original world space matrices for humanoid bones
    original_matrices = {}
    for bone in armature.bones:
        if bone.name in humanoid_bones:
            pose_bone = armature_obj.pose.bones[bone.name]
            original_matrices[bone.name] = armature_obj.matrix_world @ pose_bone.matrix
    
    # Switch to edit mode to modify bone relations
    bpy.context.view_layer.objects.active = armature_obj
    original_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Clear parent relationships for humanoid bones only
    for edit_bone in armature.edit_bones:
        if edit_bone.name in humanoid_bones:
            edit_bone.parent = None
    
    # Return to pose mode
    bpy.ops.object.mode_set(mode='POSE')
    
    # Restore original world space positions for humanoid bones
    for bone_name, original_matrix in original_matrices.items():
        pose_bone = armature_obj.pose.bones[bone_name]
        pose_bone.matrix = armature_obj.matrix_world.inverted() @ original_matrix
    
    # Return to original mode
    bpy.ops.object.mode_set(mode=original_mode)

def is_finger_bone(humanoid_bone: str) -> bool:
    """
    Determine whether a bone is a finger bone
    
    Parameters:
        humanoid_bone (str): Humanoid bone name
        
    Returns:
        bool: True if the bone is a finger bone
    """
    finger_keywords = [
        "Thumb", "Index", "Middle", "Ring", "Little",
        "Toe"
    ]
    return any(keyword in humanoid_bone for keyword in finger_keywords)

def get_next_joint_bone(humanoid_bone: str) -> Optional[str]:
    """
    Get the bone name of the next joint
    
    Parameters:
        humanoid_bone (str): Humanoid bone name
        
    Returns:
        Optional[str]: The bone name of the next joint; None if it does not exist
    """
    joint_mapping = {
        "Proximal": "Intermediate",
        "Intermediate": "Distal",
    }
    
    # Identify the current joint type
    current_joint = None
    for joint_type in joint_mapping.keys():
        if joint_type in humanoid_bone:
            current_joint = joint_type
            break
            
    if not current_joint:
        return None
        
    # Generate the bone name for the next joint
    next_joint = joint_mapping[current_joint]
    return humanoid_bone.replace(current_joint, next_joint)

def apply_finger_bone_adjustments(
    armature_obj: bpy.types.Object,
    humanoid_to_bone: Dict[str, str],
    bone_to_humanoid: Dict[str, str]
) -> None:
    """
    Adjusting Finger Bone Positions
    Adjust so that the Tail of each bone aligns with the Head of the next joint
    
    Parameters:
        armature_obj: Armature object
        humanoid_to_bone: Dictionary mapping Humanoid bone names to regular bone names
        bone_to_humanoid: Dictionary mapping regular bone names to Humanoid bone names
    """
    # Process all finger bones
    for bone_name, pose_bone in armature_obj.pose.bones.items():
        if bone_name not in bone_to_humanoid:
            continue
            
        humanoid_bone = bone_to_humanoid[bone_name]
        if not is_finger_bone(humanoid_bone):
            continue
            
        # Get the next joint
        next_humanoid_bone = get_next_joint_bone(humanoid_bone)
        if not next_humanoid_bone or next_humanoid_bone not in humanoid_to_bone:
            continue
            
        next_bone_name = humanoid_to_bone[next_humanoid_bone]
        if next_bone_name not in armature_obj.pose.bones:
            continue
            
        next_bone = armature_obj.pose.bones[next_bone_name]
        
        # Get the current bone direction vector
        current_dir = ((armature_obj.matrix_world @ pose_bone.tail) - (armature_obj.matrix_world @ pose_bone.head)).normalized()
        
        # Calculate the position in world space
        head_world = armature_obj.matrix_world @ pose_bone.head
        next_head_world = armature_obj.matrix_world @ next_bone.head
        
        # Calculate the new direction vector
        new_dir = (next_head_world - head_world).normalized()
        
        # Calculate the difference in rotation
        #rot_diff = new_dir.rotation_difference(current_dir)
        rot_diff = current_dir.rotation_difference(new_dir)
        
        # Get the current queue
        current_matrix = pose_bone.matrix.copy()
        
        translation, rotation, scale = current_matrix.decompose()
        trans_mat = Matrix.Translation(translation)

        # Create a new matrix with the rotation applied
        rot_matrix = rot_diff.to_matrix().to_4x4()
        new_matrix = trans_mat @ rot_matrix @ trans_mat.inverted() @ current_matrix
        
        print(f"{bone_name} {next_bone_name} \n {head_world} \n {next_head_world} \n {rot_diff.to_euler('XYZ')}")
        
        # Apply a new matrix
        pose_bone.matrix = new_matrix

def matrix_to_list(matrix):
    """
    Convert a Matrix to a List (for saving as JSON)

    Parameters:
        matrix: Matrix - A Blender matrix object

    Returns:
        list: The matrix converted to a 2D list
    """
    return [list(row) for row in matrix]

def list_to_matrix(matrix_list):
    """
    Convert a list to a Matrix type (for loading JSON)

    Parameters:
        matrix_list: list - A two-dimensional list containing matrix data

    Returns:
        Matrix: The converted matrix
    """
    return Matrix(matrix_list)

def add_pose_from_json(filename="pose_data.json", avatar_data_file="avatar_data.json", invert=False):
    """
    Add pose data loaded from a JSON file to the current pose of the active armature
    
    Parameters:
        filename (str): The name of the JSON file to load
        avatar_data_file (str): The name of the avatar data JSON file
        invert (bool): Whether to apply inverse transformation
    """
    # Get the active object
    active_obj = bpy.context.active_object
    
    if not active_obj:
        raise ValueError("No active object found")
    
    if active_obj.type != 'ARMATURE':
        raise ValueError(f"Active object '{active_obj.name}' is not an armature")
    
    # Load avatar data
    avatar_data = load_avatar_data(avatar_data_file)
    
    # Retrieve the hierarchy and transformation map
    bone_parents, humanoid_to_bone, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
    
    # Get the full file path
    scene_folder = get_scene_folder()
    filepath = os.path.join(scene_folder, filename)
    
    # Checking for the existence of a file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pose data file not found: {filepath}")
    
    # Load a JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    
    # Create a step for undo
    bpy.ops.ed.undo_push(message="Add Pose from JSON")

    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Disconnect all edit bones
    for bone in active_obj.data.edit_bones:
        bone.use_connect = False
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # To preserve parent-child relationships during processing, retrieve the bones in hierarchical order
    def get_bone_hierarchy_order():
        """Retrieve Humanoid bones in parent-to-child order"""
        order = []
        visited = set()
        
        def add_bone_and_children(humanoid_bone):
            if humanoid_bone in visited:
                return
            visited.add(humanoid_bone)
            order.append(humanoid_bone)
            
            # Search for child bones
            for child_bone, parent_bone in bone_parents.items():
                if parent_bone == humanoid_bone and child_bone not in visited:
                    add_bone_and_children(child_bone)
        
        # Start at the hip bones
        root_bones = []
        root_bones.append(humanoid_to_bone['Hips'])
        
        for root_bone in root_bones:
            add_bone_and_children(root_bone)
        
        return order
    
    bone_order = get_bone_hierarchy_order()
    
    # A dictionary that stores processed Humanoid bones
    processed_bones = {}
    
    # Save the pre-deformed state of all bones in advance
    original_bone_data = {}
    for humanoid_bone in humanoid_to_bone.keys():
        bone_name = humanoid_to_bone.get(humanoid_bone)
        if bone_name and bone_name in active_obj.pose.bones:
            bone = active_obj.pose.bones[bone_name]
            original_bone_data[humanoid_bone] = {
                'matrix': bone.matrix.copy(),
                'head': bone.head.copy(),
                'tail': bone.tail.copy(),
                'bone_name': bone_name
            }
    
    # Compute pose data in hierarchical order
    for bone_name in bone_order:
        if not bone_name or bone_name not in active_obj.pose.bones:
            continue
   
        humanoid_bone = bone_to_humanoid.get(bone_name)
        if not humanoid_bone:
            continue
        
        # Skip if already processed
        if humanoid_bone in processed_bones:
            continue

        # Determine whether to hold pose data directly or inherit it from a parent
        source_humanoid_bone = humanoid_bone
        if humanoid_bone not in pose_data:
            parent_with_pose = find_nearest_parent_with_pose(
                bone_name, bone_parents, bone_to_humanoid, pose_data)
            if not parent_with_pose:
                continue
            source_humanoid_bone = parent_with_pose
            print(f"Using pose data from parent bone {source_humanoid_bone} for {humanoid_bone}")
        
        # Perform calculations using the saved original data
        if humanoid_bone not in original_bone_data:
            continue
            
        bone = active_obj.pose.bones[bone_name]
        
        original_data = original_bone_data[humanoid_bone]
        
        # Retrieve the matrix in the current world space (using the original data)
        current_world_matrix = active_obj.matrix_world @ original_data['matrix']

        # Construct the transformation matrix
        bone_pose = pose_data[source_humanoid_bone]

        # If a 'delta_matrix' exists, it will be used by default (for compatibility with the old JSON format)
        # Reconstruct from location/rotation/scale only if delta_matrix is missing (new JSON format)
        if 'delta_matrix' in bone_pose:
            # Old format: Use 'delta_matrix' directly (most accurate)
            delta_matrix = list_to_matrix(bone_pose['delta_matrix'])
        elif 'location' in bone_pose and 'rotation' in bone_pose and 'scale' in bone_pose:
            # New format: Reconstruct the matrix from location/rotation/scale
            # Since rotation values are stored in degrees, convert them to radians
            delta_loc = Vector(bone_pose['location'])
            delta_rot = Euler([math.radians(x) for x in bone_pose['rotation']], 'XYZ')
            delta_scale = Vector(bone_pose['scale'])

            delta_matrix = Matrix.Translation(delta_loc) @ \
                        delta_rot.to_matrix().to_4x4() @ \
                        Matrix.Scale(delta_scale.x, 4, (1, 0, 0)) @ \
                        Matrix.Scale(delta_scale.y, 4, (0, 1, 0)) @ \
                        Matrix.Scale(delta_scale.z, 4, (0, 0, 1))
        else:
            print(f"Warning: No valid pose data for {source_humanoid_bone}, skipping.")
            continue
        
        if invert:
            delta_matrix = delta_matrix.inverted()
            
        # Add to the current array
        combined_matrix = delta_matrix @ current_world_matrix
        
        # Convert to local coordinates and apply
        bone.matrix = active_obj.matrix_world.inverted() @ combined_matrix
        
        print(bone_name)
        print(bone.matrix)
        
        # Apply changes immediately (as this affects the calculation of child bones)
        bpy.context.view_layer.update()
        
        # Mark as completed
        processed_bones[humanoid_bone] = True
    
    # Force an update of the final pose
    bpy.context.view_layer.update()
    print(f"Pose data added to armature '{active_obj.name}' from {filepath}")
    
     # Adjusting Finger Bones When Inverting
    if invert:
        #apply_finger_bone_adjustments(active_obj, humanoid_to_bone, bone_to_humanoid)
        # Force an update to the final pose
        bpy.context.view_layer.update()
    
    for bone_name in active_obj.pose.bones.keys():
        if bone_name in bone_to_humanoid:
            humanoid_name = bone_to_humanoid[bone_name]
            if humanoid_name in processed_bones:
                mat = active_obj.pose.bones[bone_name].matrix
                print(f"'{humanoid_name}' ({bone_name}) bone.matrix_final {mat}")

def get_vertices_in_scaled_bbox(source_obj, scale_factor=1.2):
    """
    Scales the bounding box calculated from the selected vertices,
    and retrieves the indices of all vertices contained within that bounding box
    
    Parameters:
    source_obj: Source object
    scale_factor: Scale factor
    
    Returns:
    list: A list of the indices of the vertices contained within the bounding box
    """
    # Calculate the bounding box of the selected vertices
    bounds_min, bounds_max = calculate_target_bounding_box(source_obj, scale_factor, use_selected_vertices=True)
    
    # Get the evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    
    # Convert all vertices to world coordinates and check if they are inside the bounding box
    matrix_world = source_obj.matrix_world
    vertices_in_bbox = []
    
    for i, vertex in enumerate(eval_obj.data.vertices):
        world_pos = matrix_world @ Vector(vertex.co)
        
        # Check if within the bounding box
        if (bounds_min.x <= world_pos.x <= bounds_max.x and
            bounds_min.y <= world_pos.y <= bounds_max.y and
            bounds_min.z <= world_pos.z <= bounds_max.z):
            vertices_in_bbox.append(i)
    
    print(f"Number of vertices in scaled Bounding Box: {len(vertices_in_bbox)}")
    return vertices_in_bbox


def calculate_target_bounding_box(target_obj, scale_factor=1.2, use_selected_vertices=False):
    """
    Calculates the bounding box of the target mesh and scales it to a square
    
    Parameters:
    target_obj: Target mesh object
    scale_factor: Scale factor (default: 1.2x)
    use_selected_vertices: If True, use only the selected vertices
    
    Returns:
    bounds_min, bounds_max: Minimum and maximum coordinates of the square bounding box
    """
    # Check whether the editor is in edit mode and retrieve the selected vertex information
    vertices_world = []
    matrix_world = target_obj.matrix_world
    
    if use_selected_vertices:
        # Save current mode
        current_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
        was_in_edit_mode = current_mode == 'EDIT'
        
        try:
            # If you are not in edit mode, switch to edit mode
            if not was_in_edit_mode:
                bpy.context.view_layer.objects.active = target_obj
                bpy.ops.object.mode_set(mode='EDIT')
            
            # Retrieve selected vertices using bmesh
            bm = bmesh.from_edit_mesh(target_obj.data)
            
            # Get only the selected vertices
            selected_vertices = [v for v in bm.verts if v.select]
            
            if not selected_vertices:
                print("Warning: No vertices selected. Using all vertices.")
                # If no vertices are selected, use all vertices
                vertices_world = [matrix_world @ Vector(v.co) for v in bm.verts]
            else:
                vertices_world = [matrix_world @ Vector(v.co) for v in selected_vertices]
                print(f"Number of selected vertices: {len(selected_vertices)}")
            
            # Update bmesh (not required, but recommended)
            bmesh.update_edit_mesh(target_obj.data)
            
        finally:
            # Return to the original mode
            if not was_in_edit_mode and current_mode == 'OBJECT':
                bpy.context.view_layer.objects.active = target_obj
                bpy.ops.object.mode_set(mode='OBJECT')
    
    else:
        # Use all vertices
        depsgraph = bpy.context.evaluated_depsgraph_get()
        target_eval = target_obj.evaluated_get(depsgraph)
        vertices_world = [matrix_world @ Vector(v.co) for v in target_eval.data.vertices]
        print(f"Using all vertices: {len(vertices_world)}")
    
    if not vertices_world:
        raise ValueError("There are no valid vertices in the target mesh.")
    
    # Convert to a NumPy array for batch processing
    vertices_array = np.array([[v.x, v.y, v.z] for v in vertices_world])
    bounds_min_orig = Vector(vertices_array.min(axis=0))
    bounds_max_orig = Vector(vertices_array.max(axis=0))
    
    # Calculate the center and dimensions of the original bounding box
    center = (bounds_min_orig + bounds_max_orig) * 0.5
    dimensions = bounds_max_orig - bounds_min_orig
    
    # Get the length of the longest side
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    
    # Apply scale
    scaled_half_size = (max_dimension * scale_factor) * 0.5
    
    # Generate a square bounding box (taking X-axis symmetry into account)
    x_extent = max(abs(center.x - scaled_half_size), abs(center.x + scaled_half_size))
    
    bounds_min = Vector((
        -x_extent,
        center.y - scaled_half_size,
        center.z - scaled_half_size
    ))
    bounds_max = Vector((
        x_extent,
        center.y + scaled_half_size,
        center.z + scaled_half_size
    ))
    
    vertex_type = "selected vertices" if use_selected_vertices else "all vertices"
    print(f"Vertices used: {vertex_type} ({len(vertices_world)})")
    print(f"Original dimensions of target mesh: {dimensions}")
    print(f"Maximum dimension: {max_dimension:.4f}")
    print(f"Scaled square size: {max_dimension * scale_factor:.4f}")
    print(f"Generated Bounding Box: Min{bounds_min}, Max{bounds_max}")
    
    return bounds_min, bounds_max


def create_adaptive_deformation_field(target_obj, base_grid_spacing=0.005, surface_distance=2.1, max_distance=2.1, min_distance=0.0036, density_falloff=3.0, bbox_scale_factor=1.2, use_selected_vertices=False):
    """
    Generates a deformation field where density varies based on distance,
    using a bounding box automatically generated from the target mesh.
    
    Parameters:
    target_obj: The target mesh for Surface Deform
    base_grid_spacing: Base grid spacing (in meters)
    surface_distance: Maximum distance from the target mesh surface
    max_distance: Maximum weight distance
    min_distance: Minimum weight distance
    density_falloff: Density decay rate (higher values result in a more rapid gradual change in density)
    bbox_scale_factor: Bounding box scale factor
    use_selected_vertices: If True, calculates the bounding box using only the selected vertices
    """
    start_time = time.time()
    
    # Automatically calculate the bounding box from the target mesh
    bounds_min, bounds_max = calculate_target_bounding_box(target_obj, bbox_scale_factor, use_selected_vertices)
    
    # Calculate the length of each axis
    dimensions = bounds_max - bounds_min
    
    # Create a BVH tree for the target mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_mesh = target_eval.data
    
    bm = bmesh.new()
    bm.from_mesh(target_mesh)
    bm.transform(target_obj.matrix_world)
    
    # Check the "ignore" vertex group and exclude faces containing vertices with weights of 0.5 or higher
    ignore_group = None
    for vg in target_obj.vertex_groups:
        if vg.name == "ignore":
            ignore_group = vg
            break
    
    if ignore_group:
        print(f"Found 'ignore' vertex group. Filtering faces...")
        # Identify vertices with a weight of 0.5 or higher
        ignore_vertices = set()
        for vert in target_mesh.vertices:
            for group in vert.groups:
                if group.group == ignore_group.index and group.weight >= 0.5:
                    ignore_vertices.add(vert.index)
        
        # Delete faces containing excluded vertices
        faces_to_remove = []
        for face in bm.faces:
            for vert in face.verts:
                if vert.index in ignore_vertices:
                    faces_to_remove.append(face)
                    break
        
        for face in faces_to_remove:
            bm.faces.remove(face)
        
        print(f"Removed {len(faces_to_remove)} faces containing ignore vertices (total ignore vertices: {len(ignore_vertices)})")
    
    bvh = BVHTree.FromBMesh(bm)
    
    # Generating Grid Points
    vertices = []
    
    # Pre-computation and Caching
    inv_max_min_diff = 1.0 / (max_distance - min_distance)
    
    # Helper Functions for Adaptive Grid Generation (Optimization)
    def get_adaptive_spacing(distance):
        if distance <= min_distance:
            return 0
        elif distance > surface_distance:
            return float('inf')  # Do not generate points outside the range
        else:
            # Calculate the normalized distance (a value between 0 and 1)
            normalized_distance = (distance - min_distance) * inv_max_min_diff
            normalized_distance = min(1.0, max(0.0, normalized_distance))
            
            # Increase the spacing by a power of 2 based on the distance
            power = sqrt(normalized_distance) * density_falloff
            level = int(power + 1)  # Extract the integer part and convert to a fractional form
            
            # Calculate the value of 2^level (optimized using bit shifts)
            return 1 << level
    
    # Generating a grid that takes into account symmetry along the X-axis
    steps_x_positive = int(ceil(bounds_max.x / base_grid_spacing)) + 1
    steps_y = int(ceil(dimensions.y / base_grid_spacing)) + 1
    steps_z = int(ceil(dimensions.z / base_grid_spacing)) + 1
    
    # Variables for progress display
    total_points = steps_x_positive * steps_y * steps_z
    processed_points = 0
    last_update = time.time()
    update_interval = 2.0  # Update progress every 2 seconds
    
    # Buffer for batch processing
    batch_size = 1000
    batch_positions = []
    batch_mirror_positions = []
    
    # Cache processed adaptive intervals
    spacing_cache = {}
    
    # Functions for step-by-step grid scanning
    def process_cell(x_start, y_start, z_start, cell_size, level=0, max_level=3):
        nonlocal batch_positions, batch_mirror_positions, processed_points, last_update
        
        # Calculate the coordinates of the eight vertices of a cell
        cell_vertices = []
        min_distance_in_cell = float('inf')
        
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    x_pos = x_start + dx * cell_size * base_grid_spacing
                    y_pos = y_start + dy * cell_size * base_grid_spacing
                    z_pos = z_start + dz * cell_size * base_grid_spacing
                    
                    vertex_pos = Vector((x_pos, y_pos, z_pos))
                    location, normal, index, distance = bvh.find_nearest(vertex_pos)
                    
                    if location and distance <= surface_distance:
                        cell_vertices.append((vertex_pos, distance))
                        min_distance_in_cell = min(min_distance_in_cell, distance)
        
        # If there are no valid vertices in the cell, do not process it
        if not cell_vertices:
            return
        
        # Get the adaptive spacing based on the minimum distance within a cell
        if min_distance_in_cell in spacing_cache:
            min_adaptive_spacing = spacing_cache[min_distance_in_cell]
        else:
            min_adaptive_spacing = get_adaptive_spacing(min_distance_in_cell)
            spacing_cache[min_distance_in_cell] = min_adaptive_spacing
        
        # If the cell size is larger than the minimum adaptive spacing and has not reached the maximum level, split it
        if cell_size > min_adaptive_spacing and level < max_level:
            half_size = cell_size // 2
            if half_size > 0:
                # Divide the cell into 8 subcells
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        for dz in [0, 1]:
                            new_x = x_start + dx * half_size * base_grid_spacing
                            new_y = y_start + dy * half_size * base_grid_spacing
                            new_z = z_start + dz * half_size * base_grid_spacing
                            process_cell(new_x, new_y, new_z, half_size, level + 1, max_level)
        else:
            # Add cell center
            x_center = x_start + (cell_size * base_grid_spacing) / 2
            y_center = y_start + (cell_size * base_grid_spacing) / 2
            z_center = z_start + (cell_size * base_grid_spacing) / 2
            
            center_pos = Vector((x_center, y_center, z_center))
            location, normal, index, distance = bvh.find_nearest(center_pos)
            
            if location and distance <= surface_distance:
                # Add to batch
                batch_positions.append(center_pos)
                batch_mirror_positions.append(Vector((-x_center, y_center, z_center)))
                
                # Process when the batch size is reached
                if len(batch_positions) >= batch_size:
                    process_batch(batch_positions, batch_mirror_positions, bvh, vertices)
                    batch_positions = []
                    batch_mirror_positions = []
            
            processed_points += 1
            current_time = time.time()
            if current_time - last_update >= update_interval:
                print(f"Processing: {processed_points} points")
                last_update = current_time
    
    # Generate an initial coarse grid and subdivide it in stages
    initial_cell_size = 2 ** int(density_falloff+1)  # Initial cell size (powers of 2 are efficient)
    
    # Generate grid points in the region where X > 0
    for z in range(0, steps_z, initial_cell_size):
        z_pos = bounds_min.z + z * base_grid_spacing
        
        for y in range(0, steps_y, initial_cell_size):
            y_pos = bounds_min.y + y * base_grid_spacing
            
            for x in range(0, steps_x_positive, initial_cell_size):
                x_pos = x * base_grid_spacing  # The positive X-axis direction
                
                # Process cells
                process_cell(x_pos, y_pos, z_pos, initial_cell_size, 0, int(density_falloff+1))
    
    # Process the remaining batches
    if batch_positions:
        process_batch(batch_positions, batch_mirror_positions, bvh, vertices)
    
    if not vertices:
        bm.free()
        print("Warning: No valid points found in specified range")
        return None

    # Cleanup
    bm.free()

    end_time = time.time()
    print(f"Number of generated points: {len(vertices)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    return vertices


def process_batch(positions, mirror_positions, bvh, vertices):
    """Processing Grid Points in a Batch"""
    for pos, mirror_pos in zip(positions, mirror_positions):
        # Points on the positive X-axis
        location, normal, index, distance = bvh.find_nearest(pos)
        if location:
            vertices.append(pos)
        
        # Points on the negative X-axis
        location, normal, index, distance = bvh.find_nearest(mirror_pos)
        if location:
            vertices.append(mirror_pos)


def compute_distances_to_source_mesh(target_vertices, source_obj):
    """
    Calculates the distance from each vertex of the target mesh to the nearest face of the source mesh
    Uses a BVHTree to calculate distances quickly
    
    Parameters:
    - target_vertices: Vertex coordinates of the target mesh (world coordinates)
    - source_obj: Source mesh object
    
    Returns:
    - An array of distances from each target vertex to the source mesh
    """
    num_vertices = len(target_vertices)
    distances = np.zeros(num_vertices)
    
    print("Building BVH tree for source mesh...")
    
    # Build a BVH tree from a source mesh
    bm_source = bmesh.new()
    bm_source.from_mesh(source_obj.data)
    bm_source.faces.ensure_lookup_table()
    
    # Convert the source mesh to world coordinates
    for v in bm_source.verts:
        v.co = source_obj.matrix_world @ v.co
    
    # Build a BVH tree
    source_bvh = BVHTree.FromBMesh(bm_source)
    
    print("Calculating distance to nearest face for each vertex...")
    for i, vertex in enumerate(target_vertices):
        # Calculate the nearest node and distance using a BVH tree
        closest_point, closest_normal, closest_face_idx, distance = source_bvh.find_nearest(vertex)
        
        if closest_point is not None:
            distances[i] = distance
        else:
            # If you can't find a match, set a higher value
            distances[i] = 9999.0
    
    # Release bmesh
    bm_source.free()
    
    print("Distance calculation complete")
    return distances


def smooth_step(x, edge0, edge1):
    """
    Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    
    Parameters:
    x: The input value to interpolate
    edge0: The lower edge of the interpolation range
    edge1: The upper edge of the interpolation range
    
    Returns:
    A value between 0 and 1, with smooth transitions at the edges
    """
    # Clamp x to the range [0, 1]
    x = np.maximum(0, np.minimum(1, (x - edge0) / (edge1 - edge0)))
    
    # Apply the smooth step formula: 3x^2 - 2x^3
    return x * x * (3 - 2 * x)


def create_partial_mesh_from_vertices(source_obj, vertex_indices):
    """
    Create a submesh object from specified vertex indices
    
    Parameters:
    - source_obj: Source mesh object
    - vertex_indices: List of vertex indices to include
    
    Returns:
    - Submesh object (temporary)
    """
    # Create a submesh using bmesh
    bm = bmesh.new()
    
    # Load the source mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    bm.from_mesh(eval_obj.data)
    
    # Apply World Transformation
    bm.transform(source_obj.matrix_world)
    
    # Delete all vertices except the selected one
    vertex_indices_set = set(vertex_indices)
    verts_to_remove = [v for i, v in enumerate(bm.verts) if i not in vertex_indices_set]
    
    for v in verts_to_remove:
        bm.verts.remove(v)
    
    # Recalculate the surface
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Remove isolated vertices and edges
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    
    # Create a new mesh
    partial_mesh = bpy.data.meshes.new(name="PartialMesh_Temp")
    bm.to_mesh(partial_mesh)
    bm.free()
    
    # Create a new object (since it has already been transformed into the world coordinate system, use the identity matrix)
    partial_obj = bpy.data.objects.new("PartialMesh_Temp", partial_mesh)
    partial_obj.matrix_world = Matrix.Identity(4)
    
    # Add to scene (required for creating a BVHTree)
    bpy.context.scene.collection.objects.link(partial_obj)
    
    return partial_obj


def add_normal_control_points_func(source_obj, control_indices, control_positions_original, control_positions_deformed, normal_distance):
    """
    Generates additional control points in the normal direction of the control points
    
    Parameters:
    - source_obj: Source mesh object
    - control_indices: Indices of the vertices used as control points
    - control_positions_original: Original control point positions (world coordinates)
    - control_positions_deformed: Deformed control point positions (world coordinates)
    - normal_distance: Distance in the normal direction (world coordinate system)
    
    Returns:
    - An array of the original positions of the extended control points
    - An array of the deformed positions of the extended control points
    """
    # Get the world matrix of the source object
    source_world_matrix = source_obj.matrix_world
    
    # Get the evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    
    # Initialize the expanded control point array (original control points + control points in the normal direction)
    extended_original = []
    extended_deformed = []
    
    # Add the original control point
    extended_original.extend(control_positions_original)
    extended_deformed.extend(control_positions_deformed)
    
    # Add control points in the normal direction for each control point
    for i, vertex_index in enumerate(control_indices):
        # Get vertex normal (local coordinates)
        vertex_normal_local = eval_obj.data.vertices[vertex_index].normal.copy()
        
        # Convert normals to world coordinates (rotation only; position is not affected)
        normal_world = source_world_matrix.to_3x3() @ vertex_normal_local
        normal_world.normalize()
        
        # Original control point position
        original_pos = Vector(control_positions_original[i])
        deformed_pos = Vector(control_positions_deformed[i])
        
        # Offset in the normal direction (by a specified distance; the direction is determined by the sign)
        normal_offset = normal_world * normal_distance
        
        # Calculate the position of the control point in the normal direction
        normal_original = original_pos + normal_offset
        normal_deformed = deformed_pos + normal_offset
        
        extended_original.append(normal_original)
        extended_deformed.append(normal_deformed)
    
    # Convert to a NumPy array
    extended_original = np.array([[p[0], p[1], p[2]] for p in extended_original])
    extended_deformed = np.array([[p[0], p[1], p[2]] for p in extended_deformed])
    
    direction_text = "inward" if normal_distance < 0 else "outward"
    print(f"Extended control points: {len(control_indices)} -> {len(extended_original)} (added control points {abs(normal_distance):.5f}m {direction_text} along normals)")
    
    return extended_original, extended_deformed


def falloff_displacements(target_vertices, target_displacements, source_obj):
    """
    Apply a falloff to displacement based on distance
    """
    num_vertices = len(target_vertices)
    
    # Calculate the distance from each vertex to the nearest face of the source mesh
    print("Calculating distance to source mesh...")
    distances = compute_distances_to_source_mesh(target_vertices, source_obj)
    
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


def multi_quadratic_biharmonic(r, epsilon=1.0):
    """Multi-Quadratic Biharmonic RBF Kernel Function"""
    return np.sqrt(r**2 + epsilon**2)


def rbf_interpolation(source_control_points, source_control_points_deformed, target_vertices, source_obj, epsilon=1.0, batch_size=100000, falloff_source_obj=None):
    """
    Calculate New Positions for Target Mesh Using RBF (Batch Version)
    
    Parameters:
    - source_control_points: Selected control points of the source mesh (reference positions) - World coordinates
    - source_control_points_deformed: Control points of the source mesh after deformation via shape keys - World coordinates
    - target_vertices: Vertex coordinates of the target mesh to be deformed (world coordinates)
    - source_obj: Source mesh object (used to calculate the distance to the nearest surface)
    - epsilon: RBF parameter
    - batch_size: Number of target vertices to process at once
    - falloff_source_obj: Source object for falloff calculation (uses source_obj if None)
    
    Returns:
    - Vertex positions of the deformed target mesh (local coordinates)
    - World coordinates of the target mesh
    - Displacement vector
    """
    # Calculate the displacement vector (post-deformation position - original position)
    displacements = source_control_points_deformed - source_control_points
    
    # Check if SciPy is available
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is not available. Please use the 'Reinstall Dependencies' button to install it.")
    
    # Calculate the scaling factor: Use a value based on the standard deviation of the distance
    if epsilon <= 0:
        # Calculate an appropriate epsilon based on the average distance
        dists = cdist(source_control_points, source_control_points)
        mean_dist = np.mean(dists[dists > 0])
        epsilon = mean_dist  # Use the average distance as epsilon
        print(f"Auto-calculated epsilon: {epsilon}")
    
    # Calculate the distance matrix between control points
    dist_matrix = cdist(source_control_points, source_control_points)
    
    # Calculate the RBF matrix
    phi = multi_quadratic_biharmonic(dist_matrix, epsilon)
    
    num_pts, dim = source_control_points.shape
    P = np.ones((num_pts, dim + 1))
    P[:, 1:] = source_control_points  # Extended matrix for polynomial terms
    
    # Build a fully linear system
    A = np.zeros((num_pts + dim + 1, num_pts + dim + 1))
    A[:num_pts, :num_pts] = phi
    A[:num_pts, num_pts:] = P
    A[num_pts:, :num_pts] = P.T
    
    # Set the right-hand side
    b = np.zeros((num_pts + dim + 1, dim))
    b[:num_pts] = displacements
    
    # Find the solution
    try:
        # Try the standard solution
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # If the matrix is singular, regularize it and use the pseudo-inverse
        print("Matrix is singular - applying regularization")
        reg = np.eye(A.shape[0]) * 1e-6
        x = np.linalg.lstsq(A + reg, b, rcond=None)[0]
    
    # Extract weights
    rbf_weights = x[:num_pts]
    poly_weights = x[num_pts:]
    
    # Get the local coordinates of the target vertex
    total_vertices = len(target_vertices)
    
    # Initialize the array to store the results
    target_deformed = np.zeros_like(target_vertices)
    target_world_vertices = np.zeros_like(target_vertices)
    target_displacements = np.zeros_like(target_vertices)
    
    # Process by batch
    print(f"Processing target mesh vertices in batches of {batch_size} (total {total_vertices} vertices)")
    
    # Progress counter
    processed_count = 0
    
    # Process by batch
    for batch_start in range(0, total_vertices, batch_size):
        batch_end = min(batch_start + batch_size, total_vertices)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch: {batch_start} to {batch_end-1} ({current_batch_size} vertices)")
        
        # Current batch coordinates
        batch_world_vertices = target_vertices[batch_start:batch_end]
        
        # Calculate the distance between the target vertex and the control point
        batch_dists = cdist(batch_world_vertices, source_control_points)
        batch_phi = multi_quadratic_biharmonic(batch_dists, epsilon)
        
        # Calculating Polynomial Terms
        batch_P = np.ones((current_batch_size, dim + 1))
        batch_P[:, 1:] = batch_world_vertices
        
        # Calculate the displacement of each target vertex
        batch_displacements = np.dot(batch_phi, rbf_weights) + np.dot(batch_P, poly_weights)
        
        # Apply falloff processing (since this consumes a large amount of memory at once, batch processing is recommended)
        falloff_obj = falloff_source_obj if falloff_source_obj is not None else source_obj
        batch_final_displacements = falloff_displacements(
            batch_world_vertices, 
            batch_displacements, 
            falloff_obj
        )
        
        # Apply displacement to target vertices (world coordinates)
        batch_deformed_world = batch_world_vertices + batch_final_displacements
        
        for i in range(current_batch_size):
            target_deformed[batch_start + i] = batch_deformed_world[i]
            target_world_vertices[batch_start + i] = batch_world_vertices[i]
            target_displacements[batch_start + i] = batch_final_displacements[i]
        
        # Update on Progress
        processed_count += current_batch_size
        progress_percent = (processed_count / total_vertices) * 100
        print(f"Progress: {processed_count}/{total_vertices} vertices processed ({progress_percent:.1f}%)")

    print("All batch processing complete")
    return target_deformed, target_world_vertices, target_displacements


def ensure_objects_visible(objects_to_check):
    """
    If the specified object is hidden, make it visible and record its original state
    
    Parameters:
        objects_to_check: A list of objects to check
    
    Returns:
        dict: A dictionary recording the original visibility state
    """
    original_states = {}
    
    for obj in objects_to_check:
        if obj is None:
            continue
        
        # Record the original state
        original_states[obj.name] = {
            'hide_viewport': obj.hide_viewport,
            'hide_render': obj.hide_render,
            'hide_select': obj.hide_select
        }
        
        # If hidden, show it
        if obj.hide_viewport:
            print(f"Made object '{obj.name}' visible")
            obj.hide_viewport = False
        
        if obj.hide_render:
            obj.hide_render = False
        
        if obj.hide_select:
            obj.hide_select = False
    
    bpy.context.view_layer.update()

    return original_states


def restore_objects_visibility(objects_to_restore, original_states):
    """
    Restore the display state of objects
    
    Parameters:
        objects_to_restore: A list of objects to restore
        original_states: A dictionary of the original display states
    """
    for obj in objects_to_restore:
        if obj is None or obj.name not in original_states:
            continue
        
        state = original_states[obj.name]
        obj.hide_viewport = state['hide_viewport']
        obj.hide_render = state['hide_render']
        obj.hide_select = state['hide_select']
        
        if state['hide_viewport']:
            print(f"Restored visibility state of object '{obj.name}'")
    
    bpy.context.view_layer.update()

def remove_overlapping_vertices(vertices, tolerance=1e-6):
    """
    Exclude overlapping vertices
    
    Parameters:
    - vertices: An array of vertex coordinates (n, 3)
    - tolerance: Threshold for distance at which vertices are considered duplicates
    
    Returns:
    - unique_indices: Indices of non-duplicate vertices
    - duplicate_mask: Mask of duplicate vertices (True = duplicate)
    """
    if len(vertices) <= 1:
        return np.arange(len(vertices)), np.zeros(len(vertices), dtype=bool)
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(vertices)
    
    # Detect overlapping neighborhoods for each vertex
    pairs = kdtree.query_pairs(r=tolerance)
    
    # Create a set of duplicate vertices
    duplicate_indices = set()
    for i, j in pairs:
        # Keep the smaller index and treat the larger one as a duplicate
        duplicate_indices.add(i)
        duplicate_indices.add(j)
    
    # Create a duplicate mask
    duplicate_mask = np.zeros(len(vertices), dtype=bool)
    duplicate_mask[list(duplicate_indices)] = True
    
    # Get the indices of unique vertices
    unique_indices = np.where(~duplicate_mask)[0]
    
    print(f"Total control points: {len(vertices)}, after deduplication: {len(unique_indices)}, removed duplicates: {len(duplicate_indices)}")
    
    return unique_indices, duplicate_mask


def identify_overlapping_control_points_for_shape_keys(
    source_obj, 
    source_shape_key_name, 
    selected_indices, 
    source_world_matrix, 
    add_normal_control_points=False, 
    normal_distance=-0.0002, 
    tolerance=1e-6
):
    """
    Checks the control point positions when the shape key values are set to 0 and 1,
    and identifies the indices of control points that overlap in either case
    
    Parameters:
    - source_obj: Source object
    - source_shape_key_name: Shape key name
    - selected_indices: Vertex indices to use as control points
    - source_world_matrix: World matrix of the source object
    - add_normal_control_points: Whether to add normal direction control points
    - normal_distance: Normal direction distance
    - tolerance: Threshold for duplicate detection
    
    Returns:
    - overlapping_indices: Indices of control points to be excluded (positions in an extended array)
    """
    
    # Save the original shape key values
    original_shape_key_value = source_obj.data.shape_keys.key_blocks[source_shape_key_name].value
    
    overlapping_indices_set = set()
    
    try:
        # Check the control point positions for ShapeKey values 0 and 1
        for shape_key_value in [0.0, 1.0]:
            print(f"Checking for duplicates at shape key value {shape_key_value}")
            
            # Set the ShapeKey value
            source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_value
            bpy.context.view_layer.update()
            
            # Get the object after evaluation
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # Get control point position (local coordinates)
            control_points_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in selected_indices])
            
            # Convert to world coordinates
            control_points_world = np.zeros_like(control_points_local)
            for i, local_co in enumerate(control_points_local):
                local_v = Vector((local_co[0], local_co[1], local_co[2], 1.0))
                world_v = source_world_matrix @ local_v
                control_points_world[i] = np.array([world_v[0], world_v[1], world_v[2]])
            
            # When adding a control point in the normal direction
            if add_normal_control_points:
                control_points_extended, _ = add_normal_control_points_func(
                    source_obj, 
                    selected_indices, 
                    control_points_world, 
                    control_points_world,  # The same before and after transformation
                    normal_distance
                )
            else:
                control_points_extended = control_points_world
            
            # Identify duplicate control points
            _, duplicate_mask = remove_overlapping_vertices(control_points_extended, tolerance)
            
            # Add duplicate indexes to the set
            duplicate_indices = np.where(duplicate_mask)[0]
            overlapping_indices_set.update(duplicate_indices)
            
            print(f"Detected {len(duplicate_indices)} duplicate control points at shape key value {shape_key_value}")
    
    finally:
        # Restore the original shape key value
        source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = original_shape_key_value
        bpy.context.view_layer.update()
    
    overlapping_indices = np.array(sorted(list(overlapping_indices_set)))
    print(f"Total duplicate control points (to be excluded from all steps): {len(overlapping_indices)}")
    
    return overlapping_indices

def create_shape_key_from_rbf(source_obj, source_shape_key_name, selected_only=True, epsilon=0.0, num_steps=1, source_avatar_name="", target_avatar_name="", save_shape_key_mode=False, keep_first_field=False, add_normal_control_points=False, normal_distance=-0.0002, shape_key_start_value=0.0, shape_key_end_value=1.0):
    """
    Perform RBF interpolation based on the source object's shape keys,
    generate a field for each step, and save the deformation field data
    
    
    Parameters:
    - source_obj: Source object (with shape keys)
    - source_shape_key_name: Name of the shape key on the source object
    - selected_only: Whether to use only selected vertices as control points
    - epsilon: RBF parameter (calculated automatically if 0 or negative)
    - num_steps: Number of steps for subdivision
    - source_avatar_name: Name of the source avatar
    - target_avatar_name: Name of the target avatar
    - save_shape_key_mode: Shape key transformation mode (save both forward and reverse directions)
    - keep_first_field: Keep the first transformation field for debugging purposes
    - add_normal_control_points: Whether to place additional control points in the normal direction of the control points
    - normal_distance: Distance in the normal direction (world coordinate system)
    - shape_key_start_value: Shape key start value
    - shape_key_end_value: Shape key end value
    """
    
    # Check the display status of the target object and set it to "Display" if necessary
    armature_obj = get_armature_from_source_object(source_obj)
    objects_to_check = [source_obj]
    if armature_obj:
        objects_to_check.append(armature_obj)
    
    original_visibility_states = ensure_objects_visible(objects_to_check)
    
    try:
        results = []
        
        # List of directions to save (determined based on 'save_shape_key_mode')
        if save_shape_key_mode:
            directions = [False, True]  # Both standard and reverse transformations
        else:
            directions = [False]  # Standard transformations only
        
        for invert in directions:
            direction_suffix = "_inv" if invert else ""
            print(f"\n=== Starting {'inverse' if invert else 'normal'} deformation processing ===")
            
            # Automatically generate the save path for field data
            scene_folder = get_scene_folder()
            
            if save_shape_key_mode:
                # In ShapeKey deformation mode
                field_data_path = os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{direction_suffix}.npz")
            else:
                # In the case of standard avatar transformations
                field_data_path = os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{direction_suffix}.npz")
            
            # Save the ShapeKey values
            original_values = {}
            for key in source_obj.data.shape_keys.key_blocks:
                original_values[key.name] = key.value
                key.value = 0.0
            
            # Set the initial state based on the Invert option
            if invert:
                # Use the end value of the shape key as a reference
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_end_value
            else:
                # Use the initial value of ShapeKey as the reference
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_start_value
            
            # Refresh the scene
            bpy.context.view_layer.update()
            
            # Retrieve the depth graph after evaluation
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            # Get the object resulting from the evaluation of the source object
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # Get the shape keys of the source object
            if source_obj.data.shape_keys is None or source_shape_key_name not in source_obj.data.shape_keys.key_blocks:
                raise ValueError(f"The shape key '{source_shape_key_name}' was not found in the source object.")
            
            # Get the world matrix of the source object
            source_world_matrix = source_obj.matrix_world
            
            # Get the index of the vertex to be used as a control point
            original_selected_vertices = []  # Record the original selected vertices
            if selected_only:
                # Switch to Object Mode to apply the selection made in Edit Mode
                was_edit_mode = False
                if bpy.context.object == source_obj and bpy.context.object.mode == 'EDIT':
                    was_edit_mode = True
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # Check if a vertex has been selected
                original_selected_vertices = [i for i, v in enumerate(source_obj.data.vertices) if v.select]
                
                # Return to edit mode
                if was_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT')
                
                if len(original_selected_vertices) == 0:
                    raise ValueError("No vertices have been selected. Please select at least one vertex.")
                
                # Use all vertices within the scaled bounding box calculated from the selected vertex as control points
                selected_indices = get_vertices_in_scaled_bbox(source_obj, bpy.context.scene.rbf_bbox_scale_factor)
                
                if len(selected_indices) < 4:
                    print(f"Warning: Very few control points ({len(selected_indices)}). Consider selecting more control points.")
            else:
                # Use all vertices
                selected_indices = list(range(len(source_obj.data.vertices)))
            
            # Identify duplicate control points with Shape Key values of 0 and 1 in advance
            print("Pre-checking for duplicate control points at shape key values 0 and 1...")
            overlapping_indices = identify_overlapping_control_points_for_shape_keys(
                source_obj, 
                source_shape_key_name, 
                selected_indices, 
                source_world_matrix, 
                add_normal_control_points, 
                normal_distance
            )
            
            # Calculate the transformation for each step
            all_displacements = []
            all_target_world_vertices = []
            
            for step in range(num_steps):
                print(f"\n=== Step {step+1}/{num_steps} ===")
                
                # Calculate the value of the current step
                progress = (step + 1) / num_steps
                if invert:
                    # In Invert mode, the value changes from the end value to the start value
                    step_value = shape_key_end_value - (shape_key_end_value - shape_key_start_value) * progress
                else:
                    # In normal mode, the value changes from the start value to the end value
                    step_value = shape_key_start_value + (shape_key_end_value - shape_key_start_value) * progress
                
                print(f"Shape key value: {step_value}")
                
                # Filter control points based on vertex groups
                filtered_indices = filter_control_points_by_vertex_groups(source_obj, selected_indices, step_value)
                
                if len(filtered_indices) < 4:
                    print(f"Warning: Very few valid control points at step {step+1} ({len(filtered_indices)}).")
                    if len(filtered_indices) == 0:
                        print(f"Skipping step {step+1}: No valid control points.")
                        continue
                
                print(f"Control points: {len(selected_indices)} -> {len(filtered_indices)} (after vertex group filtering)")
                
                # Retrieve the pre-transformation state (for bounding box calculation)
                current_basis_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in filtered_indices])
                
                # Convert the pre-transformation state to world coordinates
                current_basis = np.zeros_like(current_basis_local)
                for i, basis_co in enumerate(current_basis_local):
                    basis_v = Vector((basis_co[0], basis_co[1], basis_co[2], 1.0))
                    world_basis = source_world_matrix @ basis_v
                    current_basis[i] = np.array([world_basis[0], world_basis[1], world_basis[2]])
                
                # Generate fields for the current step (using the source object before transformation)
                print(f"Generating Deformation Field for step {step+1}...")
                field_vertices = create_adaptive_deformation_field(
                    target_obj=source_obj,
                    base_grid_spacing=bpy.context.scene.rbf_base_grid_spacing,
                    surface_distance=bpy.context.scene.rbf_surface_distance,
                    max_distance=bpy.context.scene.rbf_max_distance,
                    min_distance=bpy.context.scene.rbf_min_distance,
                    density_falloff=bpy.context.scene.rbf_density_falloff,
                    bbox_scale_factor=bpy.context.scene.rbf_bbox_scale_factor,
                    use_selected_vertices=selected_only
                )
                
                if field_vertices is None:
                    print(f"Failed to generate field at step {step+1}")
                    continue
                
                # Update the ShapeKey values to obtain the transformed state
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = step_value
                
                # Refresh the scene
                bpy.context.view_layer.update()
                
                # Retrieve the object after evaluation
                depsgraph.update()
                evaluated_source_deformed = source_obj.evaluated_get(depsgraph)
                
                # Get the vertex positions after transformation
                current_deformed_local = np.array([evaluated_source_deformed.data.vertices[i].co.copy() for i in filtered_indices])
                
                # Convert the transformed position to world coordinates
                current_deformed = np.zeros_like(current_deformed_local)
                for i, deformed_co in enumerate(current_deformed_local):
                    deformed_v = Vector((deformed_co[0], deformed_co[1], deformed_co[2], 1.0))
                    world_deformed = source_world_matrix @ deformed_v
                    current_deformed[i] = np.array([world_deformed[0], world_deformed[1], world_deformed[2]])
                
                # When adding control points in the normal direction
                if add_normal_control_points:
                    current_basis_extended, current_deformed_extended = add_normal_control_points_func(
                        source_obj, 
                        filtered_indices, 
                        current_basis, 
                        current_deformed, 
                        normal_distance
                    )
                else:
                    current_basis_extended = current_basis
                    current_deformed_extended = current_deformed
                
                # Exclude pre-identified duplicate control points
                if len(overlapping_indices) > 0:
                    print(f"Excluding {len(overlapping_indices)} pre-identified duplicate control points.")
                    # Get the indices of non-overlapping control points
                    all_indices = np.arange(len(current_basis_extended))
                    valid_indices = np.setdiff1d(all_indices, overlapping_indices)
                    
                    if len(valid_indices) < len(current_basis_extended):
                        current_basis_extended = current_basis_extended[valid_indices]
                        current_deformed_extended = current_deformed_extended[valid_indices]
                        print(f"Excluded duplicate control points: using {len(valid_indices)} control points.")
                
                # Check the maximum displacement
                displacements = current_deformed_extended - current_basis_extended
                max_disp = np.max(np.linalg.norm(displacements, axis=1))
                print(f"Maximum control point displacement: {max_disp}")
                
                # If 'selected_only' is selected, create a submesh for the falloff
                falloff_source_obj = None
                if selected_only and original_selected_vertices:
                    falloff_source_obj = create_partial_mesh_from_vertices(source_obj, original_selected_vertices)
                    print(f"Created partial mesh for falloff ({len(original_selected_vertices)} vertices)")
                
                # Perform RBF interpolation
                target_deformed, target_world_vertices, target_displacements = rbf_interpolation(
                    current_basis_extended, 
                    current_deformed_extended, 
                    field_vertices, 
                    source_obj, 
                    epsilon,
                    10000,  # batch_size
                    falloff_source_obj
                )
                
                # Cleanup of Submeshes
                if falloff_source_obj:
                    mesh_data = falloff_source_obj.data
                    bpy.data.objects.remove(falloff_source_obj, do_unlink=True)
                    bpy.data.meshes.remove(mesh_data)
                    print("Deleted partial mesh for falloff")
                
                # Save results
                all_target_world_vertices.append(target_world_vertices)
                all_displacements.append(target_displacements)
                
                print(f"Step {step+1} displacement calculation complete.")
            
            # Restore the ShapeKey values
            for key_name, value in original_values.items():
                source_obj.data.shape_keys.key_blocks[key_name].value = value
            
            # Refresh the scene
            bpy.context.view_layer.update()
            
            print(f"Used maximum of {len(current_basis_extended)} vertices as control points.")
            
            # Save Deformation Field Data
            # Use the first field object as the reference
            save_field_data_multi_step(
                field_data_path,
                all_target_world_vertices,  # Save all coordinates for each step
                all_displacements,
                num_steps,
                old_version=False,
                enable_x_mirror=bpy.context.scene.rbf_enable_x_mirror
            )
            print(f"Saved Deformation Field data: {field_data_path}")
            
            # Add the result to the list
            results.append({
                'target_world_vertices': all_target_world_vertices,
                'displacements': all_displacements,
                'filepath': field_data_path,
                'invert': invert
            })
    
    finally:
        # After processing is complete, restore the object's display state
        restore_objects_visibility(objects_to_check, original_visibility_states)
        
    # Return the generated field object as a list (return the first result for backward compatibility)
    # Note: Since the object has already been deleted, this returns an empty list
    if results:
        return [], results[0]['target_world_vertices'], results[0]['displacements']
    else:
        return [], [], []


def save_field_data_multi_step(filepath, all_field_points, all_delta_positions, num_steps, old_version=False, enable_x_mirror=True):
    """
    Save the difference between the pre- and post-deformation states of a multi-step deformation field directly as a NumPy array
    Save the coordinates for each step separately
    If 'enable_x_mirror' is enabled, save only data with X-coordinates of 0 or greater
    """
    
    # Save the object's world matrix
    world_matrix = np.identity(4)
    
    kdtree_query_k = 27
    
    # Add RBF interpolation parameters
    rbf_epsilon = 0.00001  # Fixed values
    rbf_smoothing = 0.0    # Smoothing parameter
    
    # If 'enable_x_mirror' is enabled and 'old_version' is not set, only data with X coordinates of 0 or greater is filtered.
    if not old_version and enable_x_mirror:
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
                
                print(f"Step {step+1}: original vertices {len(field_points)} -> after filter {len(filtered_field)}")
            else:
                filtered_field_points.append(np.array([]))
                filtered_delta_positions.append(np.array([]))
                print(f"Step {step+1}: field vertex count 0")
        
        # Use the filtered data
        all_field_points = filtered_field_points
        all_delta_positions = filtered_delta_positions
    elif not old_version and not enable_x_mirror:
        # If the mirror is disabled, only cast to float32
        filtered_field_points = []
        filtered_delta_positions = []
        
        for step in range(num_steps):
            field_points = all_field_points[step]
            delta_positions = all_delta_positions[step]
            
            if len(field_points) > 0:
                filtered_field_points.append(field_points.astype(np.float32))
                filtered_delta_positions.append(delta_positions.astype(np.float32))
                print(f"Step {step+1}: vertex count {len(field_points)} (no mirror filter)")
            else:
                filtered_field_points.append(np.array([]))
                filtered_delta_positions.append(np.array([]))
                print(f"Step {step+1}: field vertex count 0")
        
        # Use the data after casting
        all_field_points = filtered_field_points
        all_delta_positions = filtered_delta_positions
   
    # Save data
    np.savez(filepath,
             all_field_points=np.array(all_field_points, dtype=object),  # Save the coordinates for each step
             all_delta_positions=np.array(all_delta_positions, dtype=object),
             num_steps=num_steps,
             world_matrix=world_matrix,
             kdtree_query_k=kdtree_query_k,
             rbf_epsilon=rbf_epsilon,
             rbf_smoothing=rbf_smoothing,
             enable_x_mirror=enable_x_mirror)
    
    print(f"Saved Deformation Field differential data: {filepath}")
    print(f"Number of steps: {num_steps}")
    for step in range(num_steps):
        print(f"Step {step+1}: vertex count {len(all_field_points[step])}")
    print(f"RBF function: multi_quadratic_biharmonic, epsilon: {rbf_epsilon}, smoothing: {rbf_smoothing}")


def get_vertex_groups_and_weights(mesh_obj, vertex_index):
    """Get the group and weight of the specified vertex"""
    groups = {}
    for group in mesh_obj.vertex_groups:
        try:
            weight = group.weight(vertex_index)
            groups[group.name] = weight
        except RuntimeError:
            continue
    return groups


def filter_control_points_by_vertex_groups(mesh_obj, selected_indices, step_value):
    """
    Filter control points based on vertex group weights
    
    Parameters:
    - mesh_obj: Mesh object
    - selected_indices: List of vertex indices for potential control points
    - step_value: Current step value
    
    Returns:
    - List of filtered vertex indices
    """
    filtered_indices = []
    
    # Retrieve the exclude_min and exclude_max vertex groups
    exclude_min_group = mesh_obj.vertex_groups.get("exclude_min")
    exclude_max_group = mesh_obj.vertex_groups.get("exclude_max")
    
    for vertex_index in selected_indices:
        should_exclude = False
        
        # Processing the #exclude_min group
        if exclude_min_group and exclude_max_group:
            weight_min = 1.0
            weight_max = 0.0
            try:
                weight_min = exclude_min_group.weight(vertex_index)
                weight_max = exclude_max_group.weight(vertex_index)
                if weight_min < step_value and weight_max > step_value:
                    should_exclude = True
            except RuntimeError:
                # Do not exclude vertices that do not belong to a group
                pass
        
        # If not excluded, use as a control point
        if not should_exclude:
            filtered_indices.append(vertex_index)
    
    return filtered_indices


def get_armature_from_modifier(mesh_obj):
    """Retrieve the armature from the Armature modifier"""
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE':
            return modifier.object
    return None


def calculate_inverse_pose_matrix(mesh_obj, armature_obj, vertex_index):
    """Calculate the inverse matrix of the pose for the specified vertex"""

    # Retrieving Vertex Groups and Weights
    weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
    if not weights:
        raise ValueError(f"No weight has been assigned to vertex {vertex_index}")

    # Initializing the final transformation matrix
    final_matrix = Matrix.Identity(4)
    final_matrix.zero()
    total_weight = 0

    # Calculate the influence of each bone
    for bone_name, weight in weights.items():
        if weight > 0 and bone_name in armature_obj.data.bones:
            bone = armature_obj.data.bones[bone_name]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if bone and pose_bone:
                # Calculate the final matrix of the bone
                mat = armature_obj.matrix_world @ \
                      pose_bone.matrix @ \
                      bone.matrix_local.inverted() @ \
                      armature_obj.matrix_world.inverted()
                
                # Add matrices while taking weights into account
                final_matrix += mat * weight
                total_weight += weight

    # Normalized by the sum of the weights
    if total_weight > 0:
        final_matrix = final_matrix * (1.0 / total_weight)

    # Calculate and return the inverse matrix
    return final_matrix.inverted()


def apply_field_data(target_obj, field_data_path, shape_key_name="RBFDeform"):
    """
    Load and apply saved Deformation Field difference data to the mesh (RBF interpolation version)
    Apply the deformation using the coordinates from each step
    """
    # Loading Data
    data = np.load(field_data_path, allow_pickle=True)
    
    # Verifying and Loading Data Formats
    if 'all_field_points' in data:
        # New format: When the coordinates for each step are saved
        all_field_points = data['all_field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        print(f"Detected multi-step data (new format): {num_steps} steps")

        # Check the mirror settings (if not included in the data, use the existing settings)
        enable_x_mirror = data.get('enable_x_mirror', False)
        print(f"X-axis mirror setting: {'enabled' if enable_x_mirror else 'disabled'}")
        
        if enable_x_mirror:
            # X-axis mirroring: Invert data with X-coordinates greater than 0 to negative values and add the mirrored data
            mirrored_field_points = []
            mirrored_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step].copy()
                delta_positions = all_delta_positions[step].copy()
                
                if len(field_points) > 0:
                    # Search for data with an X-coordinate greater than 0
                    x_positive_mask = field_points[:, 0] > 0.0
                    if np.any(x_positive_mask):
                        # Create a mirror image
                        mirror_field_points = field_points[x_positive_mask].copy()
                        mirror_delta_positions = delta_positions[x_positive_mask].copy()
                        
                        # Reverse the X-coordinate and the X-component of the displacement
                        mirror_field_points[:, 0] *= -1.0
                        mirror_delta_positions[:, 0] *= -1.0
                        
                        # Merge the original data and the mirror data
                        combined_field_points = np.vstack([field_points, mirror_field_points])
                        combined_delta_positions = np.vstack([delta_positions, mirror_delta_positions])
                        
                        mirrored_field_points.append(combined_field_points)
                        mirrored_delta_positions.append(combined_delta_positions)
                        
                        print(f"Step {step+1}: original vertices {len(field_points)} -> after mirror {len(combined_field_points)}")
                    else:
                        mirrored_field_points.append(field_points)
                        mirrored_delta_positions.append(delta_positions)
                        print(f"Step {step+1}: field vertex count {len(field_points)} (no mirror targets)")
                else:
                    mirrored_field_points.append(field_points)
                    mirrored_delta_positions.append(delta_positions)
                    print(f"Step {step+1}: field vertex count 0")
            
            # Use the data after applying the mirror
            all_field_points = mirrored_field_points
            all_delta_positions = mirrored_delta_positions
        else:
            # If mirroring is disabled, use the original data as is
            print("X-axis mirroring is disabled, using original data")
            for step in range(num_steps):
                print(f"Step {step+1}: field vertex count {len(all_field_points[step])}")
        
    else:
        # For backward compatibility, single-step data is also processed
        field_points = data.get('field_points')
        delta_positions = data.get('delta_positions')
        all_field_points = [field_points]
        all_delta_positions = [delta_positions]
        num_steps = 1
        print("Detected single-step data")
    
    field_matrix = Matrix(data['world_matrix'])
    field_matrix_inv = field_matrix.inverted()
    
    # Loading RBF Parameters
    rbf_epsilon = float(data.get('rbf_epsilon', 0.00001))
    
    print(f"RBF interpolation parameters: function=multi_quadratic_biharmonic, epsilon={rbf_epsilon}")
    
    # Get the evaluated mesh (after applying modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = target_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    
    # Preparing Shape Key
    if target_obj.data.shape_keys is None:
        target_obj.shape_key_add(name='Basis')
    
    # Create a shape key
    shape_key = target_obj.shape_key_add(name=shape_key_name)
    shape_key.value = 1.0
    
    # Prepare the vertex data as a NumPy array
    vertices = np.array([v.co for v in eval_mesh.vertices])
    num_vertices = len(vertices)
    
    # Reset cumulative displacement
    cumulative_displacements = np.zeros((num_vertices, 3))
    # Save the current vertex position (world coordinates)
    current_world_positions = np.array([target_obj.matrix_world @ Vector(v) for v in vertices])
    
    # Apply the displacements from each step cumulatively
    for step in range(num_steps):
        field_points = all_field_points[step]
        delta_positions = all_delta_positions[step]
        
        print(f"Applying deformation for step {step+1}/{num_steps}...")
        print(f"Number of field vertices to use: {len(field_points)}")
        
        # Check if SciPy is available
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is not available. Please use the 'Reinstall Dependencies' button to install it.")
        
        # Search for nearest points using a KDTree (constructing a new KDTree at each step)
        kdtree = cKDTree(field_points)
        
        # Calculate new vertex positions using custom RBF interpolation
        batch_size = 1000
        step_displacements = np.zeros((num_vertices, 3))
        
        for start_idx in range(0, num_vertices, batch_size):
            end_idx = min(start_idx + batch_size, num_vertices)
            batch_vertices = vertices[start_idx:end_idx]
            
            # Convert all vertices in the batch to field space (taking into account the current cumulative displacement)
            batch_world = current_world_positions[start_idx:end_idx].copy()
            batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
            
            # Interpolate using the inverse-distance weighting method for each vertex
            batch_displacements = np.zeros((len(batch_field), 3))
            
            for i, point in enumerate(batch_field):
                # Search for nearby points (up to 8 points)
                k = min(8, len(field_points))
                distances, indices = kdtree.query(point, k=k)
                
                # When the distance is 0 (when there is an exact match)
                if distances[0] < 1e-10:
                    batch_displacements[i] = delta_positions[indices[0]]
                    continue
                
                # Calculate the inverse distance weighting
                weights = 1.0 / np.sqrt(distances**2 + rbf_epsilon**2)
                
                # Weight normalization
                weights /= np.sum(weights)
                
                # Calculate displacement using a weighted average
                weighted_deltas = delta_positions[indices] * weights[:, np.newaxis]
                batch_displacements[i] = np.sum(weighted_deltas, axis=0)
            
            # Calculate displacement in world space
            for i, displacement in enumerate(batch_displacements):
                world_displacement = field_matrix.to_3x3() @ Vector(displacement)
                step_displacements[start_idx + i] = world_displacement
                
                # Update the current world position (for the next step)
                current_world_positions[start_idx + i] += world_displacement
        
        # Add the displacement from this step to the cumulative displacement
        cumulative_displacements += step_displacements
        
        print(f"Step {step+1} complete: max displacement {np.max(np.linalg.norm(step_displacements, axis=1)):.6f}")
    
    # Acquisition of Armature
    armature_obj = get_armature_from_modifier(target_obj)
    if not armature_obj:
        print("Armature modifier not found")
    
    # Calculate the final vertex position by applying the cumulative displacement
    results = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        # Convert the position obtained by adding the cumulative displacement to the original world position into local coordinates
        world_pos = target_obj.matrix_world @ Vector(vertices[i])
        final_world_pos = world_pos + Vector(cumulative_displacements[i])
        if armature_obj:
            matrix_armature_inv = calculate_inverse_pose_matrix(target_obj, armature_obj, i)
            undeformed_world_pos = matrix_armature_inv @ Vector(final_world_pos)
        else:
            undeformed_world_pos = Vector(final_world_pos)
        local_pos = target_obj.matrix_world.inverted() @ undeformed_world_pos
        results[i] = local_pos
    
    # Apply the results to ShapeKey
    for i, local_pos in enumerate(results):
        shape_key.data[i].co = local_pos
    
    print(f"Applied cumulative deformation from all steps: {shape_key_name}")
    print(f"Final maximum cumulative displacement: {np.max(np.linalg.norm(cumulative_displacements, axis=1)):.6f}")


# Definition of Properties
def create_field_object_from_data(field_data_path, target_step=1, object_name="FieldVisualization"):
    """
    Load saved Deformation Field difference data and create the field as a Blender object
    Save the displacement for each step as a shape key
    
    Parameters:
    - field_data_path: Path to the field data file
    - target_step: Step to display (starting from 1)
    - object_name: Name of the object to be created
    
    Returns:
    - The created Blender object
    """
    # Loading Data
    data = np.load(field_data_path, allow_pickle=True)
    
    # Verifying and Loading Data Formats
    if 'all_field_points' in data:
        # New format: When the coordinates for each step are saved
        all_field_points = data['all_field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        print(f"Detected multi-step data (new format): {num_steps} steps.")
    elif 'field_points' in data and 'all_delta_positions' in data:
        # Old format: When a single set of coordinates is stored
        field_points = data['field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        
        # In the old format, the same coordinates are used for all steps
        all_field_points = [field_points for _ in range(num_steps)]
        print(f"Detected multi-step data (old format): {num_steps} steps.")
    else:
        # For backward compatibility, single-step data is also processed
        field_points = data.get('field_points', data.get('delta_positions', []))
        delta_positions = data.get('delta_positions', data.get('all_delta_positions', [[]])[0])
        all_field_points = [field_points]
        all_delta_positions = [delta_positions]
        num_steps = 1
        print("Detected single-step data")
    
    # Verification of the number of steps
    if target_step < 1 or target_step > num_steps:
        raise ValueError(f"Step {target_step} is out of range (valid range: 1-{num_steps})")
    
    # Retrieve data for the specified step (convert to 0-based)
    step_index = target_step - 1
    field_points = all_field_points[step_index]
    
    if len(field_points) == 0:
        raise ValueError("The field is empty")
    
    print(f"Field point count for step {target_step}/{num_steps}: {len(field_points)}")
    
    # Create a mesh object
    mesh = bpy.data.meshes.new(object_name + "_mesh")
    obj = bpy.data.objects.new(object_name, mesh)
    
    # Set vertex coordinates
    vertices = []
    for point in field_points:
        if hasattr(point, '__len__') and len(point) >= 3:
            vertices.append([point[0], point[1], point[2]])
        else:
            print(f"Warning: Invalid point data: {point}")
    
    if not vertices:
        raise ValueError("No valid vertices found")
    
    mesh.from_pydata(vertices, [], [])
    mesh.update()
    
    # Add to scene
    bpy.context.scene.collection.objects.link(obj)
    
    # Create a base shape key
    obj.shape_key_add(name='Basis')
    
    # Add the displacement of the specified step as a shape key
    step_name = f"Step_{target_step:02d}_Displacement"
    shape_key = obj.shape_key_add(name=step_name)
    
    # Get the displacement at the specified step
    target_delta_positions = all_delta_positions[step_index]
    
    # Verify that the number of field points matches the number of displacements
    field_count = len(field_points)
    delta_count = len(target_delta_positions)
    
    if field_count != delta_count:
        print(f"Warning: Field point count ({field_count}) and displacement count ({delta_count}) mismatch at step {target_step}")
    else:
        # Apply a displacement to the shape key
        for i in range(min(len(vertices), len(target_delta_positions))):
            if i < len(shape_key.data):
                # Add the displacement to the original vertex position
                original_pos = vertices[i]
                displacement = target_delta_positions[i]
                
                if hasattr(displacement, '__len__') and len(displacement) >= 3:
                    shape_key.data[i].co = [
                        original_pos[0] + displacement[0],
                        original_pos[1] + displacement[1],
                        original_pos[2] + displacement[2]
                    ]
                else:
                    print(f"Warning: Invalid displacement data at step {target_step}: index {i}")
        
        print(f"Created shape key '{step_name}': {len(target_delta_positions)} displacements")
    
    # Select an object to make it active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    print(f"Created field object '{object_name}'")
    print(f"Vertex count: {len(vertices)}, target step: {target_step}/{num_steps}")
    
    return obj


def register_properties():
    bpy.types.Scene.rbf_source_obj = bpy.props.PointerProperty(
        name="Source Mesh",
        description="Source Mesh Object",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    
    bpy.types.Scene.rbf_source_shape_key = bpy.props.StringProperty(
        name="Source Shape Key",
        description="Shape keys of the source object"
    )
    
    # Properties for specifying the range of Shape Key values
    bpy.types.Scene.rbf_shape_key_start_value = bpy.props.FloatProperty(
        name="Shape Key Start Value",
        description="Shape Key start values",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_shape_key_end_value = bpy.props.FloatProperty(
        name="Shape Key End Value", 
        description="Shape Key's final value",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_selected_only = bpy.props.BoolProperty(
        name="Selected Vertices Only",
        description="Use only the selected vertices as control points",
        default=False
    )
    
    bpy.types.Scene.rbf_save_shape_key_mode = bpy.props.BoolProperty(
        name="Save Self Shape Key Transform",
        description="Save the shape key transformations of the source avatar (both normal and inverted)",
        default=False
    )
    
    bpy.types.Scene.rbf_keep_first_field = bpy.props.BoolProperty(
        name="Keep First Field for Debug",
        description="Leave the first transformation field intact for debugging purposes",
        default=False
    )
    
    bpy.types.Scene.rbf_epsilon = bpy.props.FloatProperty(
        name="Epsilon",
        description="RBF parameters (automatically calculated if 0 or less)",
        default=0.00001,
        precision=6
    )
    
    bpy.types.Scene.rbf_num_steps = bpy.props.IntProperty(
        name="Number of Steps",
        description="Number of steps to split the transformation",
        min=1,
        default=1
    )
    
    # Add an avatar name property
    bpy.types.Scene.rbf_source_avatar_name = bpy.props.StringProperty(
        name="Source Avatar Name",
        description="Original avatar name",
        default=""
    )
    
    bpy.types.Scene.rbf_target_avatar_name = bpy.props.StringProperty(
        name="Target Avatar Name",
        description="Name of the avatar to convert to",
        default=""
    )
    
    # Add properties to the avatar data file
    bpy.types.Scene.rbf_source_avatar_data_file = bpy.props.StringProperty(
        name="Source Avatar Data",
        description="Source avatar data file",
        default="avatar_data_template.json",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.rbf_target_avatar_data_file = bpy.props.StringProperty(
        name="Target Avatar Data",
        description="Target avatar data file",
        default="avatar_data_target.json",
        subtype='FILE_PATH'
    )
    
    # Properties of Normal Control Points
    bpy.types.Scene.rbf_add_normal_control_points = bpy.props.BoolProperty(
        name="Add Normal Control Points",
        description="Place additional control points in the normal direction of the control point",
        default=False
    )
    
    bpy.types.Scene.rbf_normal_distance = bpy.props.FloatProperty(
        name="Normal Distance",
        description="Distance along the normal direction (world coordinate system; negative values indicate inward, positive values indicate outward)",
        default=-0.0002,
        min=-0.005,
        max=0.005,
        precision=5
    )
    
    # X Mirror Properties
    bpy.types.Scene.rbf_enable_x_mirror = bpy.props.BoolProperty(
        name="Enable X Mirror",
        description="Enable X-axis mirroring (save only data with X-coordinates of 0 or greater, and automatically mirror it upon loading)",
        default=True
    )
    
    bpy.types.Scene.rbf_apply_shape_key_name = bpy.props.StringProperty(
        name="Apply Shape Key Name",
        description="Name of the shape key to apply",
        default="RBF_Deform"
    )
    
    # Deformation Field Parameters
    bpy.types.Scene.rbf_base_grid_spacing = bpy.props.FloatProperty(
        name="Base Grid Spacing",
        description="Basic grid spacing (in meters)",
        default=0.00250,
        min=0.0001,
        max=0.1,
        precision=5
    )
    
    bpy.types.Scene.rbf_surface_distance = bpy.props.FloatProperty(
        name="Surface Distance",
        description="Maximum distance from the target mesh surface",
        default=2.0,
        min=0.1,
        max=10.0,
        precision=3
    )
    
    bpy.types.Scene.rbf_max_distance = bpy.props.FloatProperty(
        name="Max Distance",
        description="Maximum weight distance",
        default=0.2,
        min=0.001,
        max=1.0,
        precision=4
    )
    
    bpy.types.Scene.rbf_min_distance = bpy.props.FloatProperty(
        name="Min Distance",
        description="Minimum Weighted Distance",
        default=0.005,
        min=0.0001,
        max=0.1,
        precision=5
    )
    
    bpy.types.Scene.rbf_density_falloff = bpy.props.FloatProperty(
        name="Density Falloff",
        description="Density decay rate (increasing this value causes the stages to change more rapidly)",
        default=4.0,
        min=1.0,
        max=10.0,
        precision=2
    )
    
    bpy.types.Scene.rbf_bbox_scale_factor = bpy.props.FloatProperty(
        name="BBox Scale Factor",
        description="Bounding Box Scale Factor",
        default=1.5,
        min=1.0,
        max=5.0,
        precision=2
    )
    
    # Pose-related properties
    bpy.types.Scene.rbf_pose_invert = bpy.props.BoolProperty(
        name="Invert Pose",
        description="Whether to apply the pose using inverse transformation",
        default=False
    )
    
    # Debugging Properties
    bpy.types.Scene.rbf_show_debug_info = bpy.props.BoolProperty(
        name="Show Debug Info",
        description="Display debug information",
        default=False
    )
    
    # Field Visualization Properties
    bpy.types.Scene.rbf_field_step = bpy.props.IntProperty(
        name="Field Step",
        description="Number of steps for the field to be visualized",
        min=1,
        default=1
    )
    
    bpy.types.Scene.rbf_field_use_inverse = bpy.props.BoolProperty(
        name="Use Inverse Data",
        description="Using inverse transform data",
        default=False
    )
    
    bpy.types.Scene.rbf_field_object_name = bpy.props.StringProperty(
        name="Field Object Name",
        description="Name of the field object to create",
        default="FieldVisualization"
    )


def get_armature_from_source_object(source_obj):
    """
    Searches the source object for an Armature modifier and returns the corresponding Armature object
    
    Parameters:
        source_obj: Source mesh object
        
    Returns:
        bpy.types.Object: Armature object; None if not found
    """
    if not source_obj or source_obj.type != 'MESH':
        return None
    
    for modifier in source_obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object:
            return modifier.object
    return None


# Toolbar Settings
class RBF_PT_DeformationPanel(bpy.types.Panel):
    bl_label = "MochiFitter-Kai-EN"
    bl_idname = "RBF_PT_DeformationPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MochiFitter-Kai-EN'  # Configure it to display in the "MochiFitter-Kai-EN" tab
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Avatar Name Settings Section
        box = layout.box()
        box.label(text="Avatar Settings", icon='ARMATURE_DATA')
        
        row = box.row()
        row.prop(scene, "rbf_source_avatar_name")
        
        row = box.row()
        row.prop(scene, "rbf_target_avatar_name")
        
        # Avatar Data File Settings
        col = box.column(align=True)
        col.label(text="Avatar Data File:")
        col.prop(scene, "rbf_source_avatar_data_file", text="Source")
        col.prop(scene, "rbf_target_avatar_data_file", text="Target")
        
        # Avatar Switch Button
        row = box.row()
        row.operator("object.swap_avatar_settings", text="Swap Source and Target", icon='ARROW_LEFTRIGHT')

        # Selecting a Source Object
        row = box.row()
        row.prop(scene, "rbf_source_obj")
        
        # Humanoid Bone "Inherit Scale" Settings Button
        row = box.row()
        humanoid_bone_ready = (context.active_object and 
                              context.active_object.type == 'ARMATURE' and 
                              scene.rbf_source_avatar_data_file)
        if humanoid_bone_ready:
            row.operator("object.set_humanoid_bone_inherit_scale", 
                        text="Humanoid Bones: Inherit Scale → Average", 
                        icon='BONE_DATA')
        else:
            if not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Please select the Armature object.", icon='ERROR')
            elif not scene.rbf_source_avatar_data_file:
                row.label(text="Please configure the source avatar data file.", icon='ERROR')
        
        # Base Pose Variations Section
        box = layout.box()
        box.label(text="Base Pose Variations", icon='ARMATURE_DATA')
        
        # Save Base Pose Button
        row = box.row()
        armature_available = False
        base_pose_save_ready = False
        if scene.rbf_source_avatar_name and context.active_object and context.active_object.type == 'ARMATURE':
            armature_available = True
            if scene.rbf_source_avatar_data_file:
                base_pose_save_ready = True
                base_pose_filename = f"pose_basis_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}.json"
                row.operator("object.save_base_pose_diff", text=f"Save Base Pose", icon='EXPORT')
                row = box.row()
                row.label(text=f"Save to: {base_pose_filename}", icon='FILE')
            else:
                row.label(text="Specify the source avatar data file.", icon='ERROR')
        else:
            if not scene.rbf_source_avatar_name:
                row.label(text="Set the source avatar name.", icon='ERROR')
            elif not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Please select the Armature object", icon='ERROR')
            else:
                row.label(text="Please complete the setup.", icon='ERROR')
        
        # Base Pose Application Section
        row = box.row()
        row.prop(scene, "rbf_pose_invert")
        
        row = box.row()
        base_pose_apply_ready = armature_available and scene.rbf_source_avatar_data_file
        if base_pose_apply_ready:
            row.operator("object.apply_base_pose_diff", text="Apply the base pose", icon='IMPORT')
        else:
            if armature_available:
                row.label(text="Specify the avatar data file.", icon='ERROR')
            else:
                row.label(text="Please complete the setup.", icon='ERROR')

        # Pose Variations Section
        box = layout.box()
        box.label(text="Pose variations", icon='POSE_HLT')
        
        # Save Pose Button
        row = box.row()
        pose_armature_available = False
        pose_save_ready = False
        if scene.rbf_source_avatar_name and scene.rbf_target_avatar_name and context.active_object and context.active_object.type == 'ARMATURE':
            pose_armature_available = True
            if scene.rbf_source_avatar_data_file:
                pose_save_ready = True
                pose_filename = f"posediff_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.json"
                row.operator("object.save_pose_diff", text=f"Save Pose", icon='EXPORT')
                row = box.row()
                row.label(text=f"Save to: {pose_filename}", icon='FILE')
            else:
                row.label(text="Specify the source avatar data file.", icon='ERROR')
        else:
            if not scene.rbf_source_avatar_name or not scene.rbf_target_avatar_name:
                row.label(text="Please set your avatar name.", icon='ERROR')
            elif not context.active_object or context.active_object.type != 'ARMATURE':
                row.label(text="Please select the Armature object.", icon='ERROR')
            else:
                row.label(text="Please complete the setup.", icon='ERROR')
        
        # Apply Pose Section
        row = box.row()
        row.prop(scene, "rbf_pose_invert")
        
        row = box.row()
        pose_apply_ready = pose_armature_available and scene.rbf_target_avatar_data_file
        if pose_apply_ready:
            row.operator("object.apply_pose_diff", text="Apply Pose", icon='IMPORT')
        else:
            if pose_armature_available:
                row.label(text="Specify the target avatar data file.", icon='ERROR')
            else:
                row.label(text="Please complete the setup.", icon='ERROR')
        
        # Dividing line
        layout.separator()
        
        # UI Rendering
        box = layout.box()
        box.label(text="Deformation Field Settings", icon='MESH_DATA')
        
        # If a source object is selected, display the shape key dropdown
        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys:
            row = box.row()
            row.label(text="Shape Key:")
            
            # Dropdown menu for selecting a shape key
            row = box.row()
            shape_keys = [key.name for key in scene.rbf_source_obj.data.shape_keys.key_blocks if key.name != "Basis"]
            if shape_keys:
                op = row.operator("object.select_rbf_shape_key", text=scene.rbf_source_shape_key if scene.rbf_source_shape_key else "Select Shape Key")
            else:
                row.label(text="There are no valid shape keys.")
        elif scene.rbf_source_obj:
            box.label(text="The source object does not have any shape keys.", icon='ERROR')
        
        # Setting the Value Range for Shape Key
        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys and scene.rbf_source_shape_key:
            col = box.column(align=True)
            col.label(text="Range of Shape Key values:")
            row = col.row(align=True)
            row.prop(scene, "rbf_shape_key_start_value", text="Start value")
            row.prop(scene, "rbf_shape_key_end_value", text="End Value")
            
            # Range Validity Check
            if scene.rbf_shape_key_start_value == scene.rbf_shape_key_end_value:
                col.label(text="Please set different values for the start and end values.", icon='ERROR')
        
        # Shape Key Deformation Preservation Option
        col = box.column(align=True)
        col.prop(scene, "rbf_save_shape_key_mode")
        # Whether to use only the selected vertices
        col.prop(scene, "rbf_selected_only")
        # Debugging Options
        col.prop(scene, "rbf_keep_first_field")
        
        col = box.column(align=True)
        col.label(text="RBF Deformation Settings:")
        # Epsilon Settings
        col.prop(scene, "rbf_epsilon")
        # Setting the number of steps
        col.prop(scene, "rbf_num_steps")
        
        # Setting Normal Control Points
        col = box.column(align=True)
        col.label(text="Setting Normal Control Points:")
        col.prop(scene, "rbf_add_normal_control_points")
        if scene.rbf_add_normal_control_points:
            col.prop(scene, "rbf_normal_distance")
        
        # Deformation Field Parameter Section
        # Basic Parameters
        col = box.column(align=True)
        col.label(text="Grid Settings:")
        col.prop(scene, "rbf_base_grid_spacing")
        col.prop(scene, "rbf_bbox_scale_factor")
        
        # Distance parameter
        col = box.column(align=True)
        col.label(text="Distance-dependent attenuation settings:")
        col.prop(scene, "rbf_surface_distance")
        col.prop(scene, "rbf_max_distance")
        col.prop(scene, "rbf_min_distance")
        
        # Density Settings
        col = box.column(align=True)
        col.prop(scene, "rbf_density_falloff")
        
        # Run button
        box = layout.box()
        
        # Display any warning messages
        warning_msg = ""
        if not SCIPY_AVAILABLE:
            warning_msg = "SciPy is not available. Please use the 'Reinstall Dependencies' button."
        elif not scene.rbf_source_avatar_name or not scene.rbf_target_avatar_name:
            warning_msg = "Please set your avatar name"
        elif not scene.rbf_source_obj:
            warning_msg = "Please select the source object"
        elif not scene.rbf_source_obj.data.shape_keys:
            warning_msg = "The source object does not have any shape keys."
        elif not scene.rbf_source_shape_key:
            warning_msg = "Please select a source shape key"
        
        if warning_msg:
            box.label(text=warning_msg, icon='ERROR')
        else:
            # Run Button (Traditional Method)
            row = box.row()
            row.scale_y = 1.2
            op = row.operator("object.create_rbf_deformation", text="Save Transformations (Single-Threaded)", icon='MOD_MESHDEFORM')
            
            # Temporary Data Export Button (New Method)
            row = box.row()
            row.scale_y = 1.5
            op = row.operator("object.export_rbf_temp_data", text="Temporary Data Export & Multithreading", icon='PLAY')
            
            # X Mirror checkbox
            row = box.row()
            row.prop(scene, "rbf_enable_x_mirror", icon='MOD_MIRROR')
            
            # Add a note
            row = box.row()
            row.label(text="* Note: rbf_multithread_processor.py must be in the same folder.", icon='INFO')
            
            # Display the destination file name
            row = box.row()
            if scene.rbf_save_shape_key_mode:
                # For ShapeKey transformation mode (save both the normal and inverted versions)
                base_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"
                row.label(text=f"Default name: {base_filename} + _inv.npz", icon='FILE')
                row = box.row()
                temp_filename = f"temp_rbf_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"
                row.label(text=f"Temporary file: {temp_filename} + _inv.npz", icon='TEMP')
            else:
                # For standard avatar transformations (standard only)
                field_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"
                row.label(text=f"Default name: {field_filename}", icon='FILE')
                row = box.row()
                temp_filename = f"temp_rbf_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"
                row.label(text=f"Temporary file: {temp_filename}", icon='TEMP')
        
        # Dividing line
        layout.separator()
        
        # Applying Saved Field Data Section
        box = layout.box()
        box.label(text="Apply saved transformation data", icon='IMPORT')
        
        row = box.row()
        row.prop(scene, "rbf_apply_shape_key_name", text="Shape Key Name")
        
        # Apply Default Deformation Data (Normal)
        apply_row = box.row(align=True)
        apply_row.operator("rbf.apply_field_data", text="Apply Deformation Data")
        
        # Apply inverse transformation data
        apply_row.operator("rbf.apply_inverse_field_data", text="Apply inverse transformation data")
        
        # Dividing line
        layout.separator()
        
        # Field Visualization Section
        box = layout.box()
        box.label(text="Field Visualization", icon='MESH_ICOSPHERE')
        
        col = box.column(align=True)
        col.prop(scene, "rbf_field_step")
        col.prop(scene, "rbf_field_use_inverse")
        col.prop(scene, "rbf_field_object_name")
        
        # Field Visualization Button
        field_row = box.row()
        field_row.scale_y = 1.2
        
        # Display the filename that will be created with the current settings
        warning_msg = ""
        if not scene.rbf_source_avatar_name:
            warning_msg = "Please set the source avatar name"
        elif scene.rbf_save_shape_key_mode and not scene.rbf_source_shape_key:
            warning_msg = "In Shape Key Deformation Mode, please select a shape key name."
        elif not scene.rbf_save_shape_key_mode and not scene.rbf_target_avatar_name:
            warning_msg = "In Avatar Transformation Mode, please set the target avatar name."
        
        if warning_msg:
            box.label(text=warning_msg, icon='ERROR')
        else:
            field_row.operator("rbf.create_field_visualization", text="Visualize the field", icon='MESH_ICOSPHERE')
            
            # Display the target filename
            row = box.row()
            inverse_suffix = "_inv" if scene.rbf_field_use_inverse else ""
            if scene.rbf_save_shape_key_mode:
                target_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}{inverse_suffix}.npz"
            else:
                target_filename = f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}{inverse_suffix}.npz"
            row.label(text=f"Target: {target_filename}", icon='FILE')
            
            row = box.row()
            direction_text = "Inverse transformation" if scene.rbf_field_use_inverse else "Standard Conversion"
            row.label(text=f"Settings: {direction_text}, Step {scene.rbf_field_step}", icon='INFO')
        
        # Dividing line
        layout.separator()
        
        # Reinstalling NumPy and SciPy Section (Always Show)
        box = layout.box()
        box.label(text="Dependency Management", icon='LIBRARY_DATA_DIRECT')
        
        # Display the current versions of NumPy and SciPy
        col = box.column(align=True)
        try:
            import numpy as np
            numpy_version = np.__version__
            col.label(text=f"Current NumPy: {numpy_version}", icon='CHECKMARK')
        except ImportError:
            col.label(text="NumPy cannot be found.", icon='ERROR')
        
        try:
            import scipy
            scipy_version = scipy.__version__
            col.label(text=f"Current SciPy: {scipy_version}", icon='CHECKMARK')
        except ImportError:
            col.label(text="SciPy cannot be found (it will be installed).", icon='INFO')

        try:
            import numba
            numba_version = numba.__version__
            col.label(text=f"Current Numba: {numba_version}", icon='CHECKMARK')
        except ImportError:
            col.label(text="Numba not found (it will be installed)", icon='INFO')

        row = col.row()
        row.scale_y = 1.2
        row.operator("rbf.reinstall_numpy_scipy_multithreaded", text="Reinstalling Dependency Packages", icon='FILE_REFRESH')
        
        # Dividing line
        layout.separator()
        
        # Debugging Section
        box = layout.box()
        box.label(text="Debugging and Troubleshooting", icon='CONSOLE')
        
        row = box.row()
        row.prop(scene, "rbf_show_debug_info")
        
        if scene.rbf_show_debug_info:
            col = box.column(align=True)
            col.label(text="Python Path Diagnosis:", icon='INFO')
            
            row = col.row(align=True)
            row.operator("rbf.debug_show_python_paths", text="Display Path Information")
            row.operator("rbf.debug_test_external_python", text="External Python Tests")
            
            col.separator()
            col.label(text="Troubleshooting:")
            col.label(text="• If you encounter an import error, run the test above.")
            col.label(text="• Check the path information in the console.")
            col.label(text="• Place rbf_multithread_processor.py in the same folder.")
            col.label(text="• Use the multithreaded version by reinstalling the dependent packages.")


# Operator for selecting shape keys
class SELECT_OT_RBFShapeKey(bpy.types.Operator):
    bl_idname = "object.select_rbf_shape_key"
    bl_label = "Select Shape Key"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        scene = context.scene

        if scene.rbf_source_obj and scene.rbf_source_obj.data.shape_keys:
            # Create a list of ShapeKey
            type(self).shape_keys = [key.name for key in scene.rbf_source_obj.data.shape_keys.key_blocks if key.name != "Basis"]

            if self.shape_keys:
                return context.window_manager.invoke_popup(self, width=200)

        return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        for key_name in type(self).shape_keys:
            op = layout.operator("object.set_rbf_shape_key", text=key_name)
            op.shape_key_name = key_name


# Operator for ShapeKey settings
class SET_OT_RBFShapeKey(bpy.types.Operator):
    bl_idname = "object.set_rbf_shape_key"
    bl_label = "Set Shape Key"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    shape_key_name: bpy.props.StringProperty()
    
    def execute(self, context):
        context.scene.rbf_source_shape_key = self.shape_key_name
        return {'FINISHED'}


# RBF deformation operator
class CREATE_OT_RBFDeformation(bpy.types.Operator, ExportHelper):
    bl_idname = "object.create_rbf_deformation"
    bl_label = "Save Deformation Data"
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper Properties
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        
        # Retrieve the required parameters
        source_obj = scene.rbf_source_obj
        source_shape_key_name = scene.rbf_source_shape_key
        selected_only = scene.rbf_selected_only
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        keep_first_field = scene.rbf_keep_first_field
        epsilon = scene.rbf_epsilon
        num_steps = scene.rbf_num_steps
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        add_normal_control_points = scene.rbf_add_normal_control_points
        normal_distance = scene.rbf_normal_distance
        shape_key_start_value = scene.rbf_shape_key_start_value
        shape_key_end_value = scene.rbf_shape_key_end_value
        
        # Avatar Name Validation
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        # Validation of ShapeKey Value Ranges
        if shape_key_start_value == shape_key_end_value:
            self.report({'ERROR'}, "Shape key start and end values must be different")
            return {'CANCELLED'}
        
        default_paths = []
        scene_folder = get_scene_folder()
        
        if scene.rbf_save_shape_key_mode:
            # In ShapeKey deformation mode
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"))
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}_inv.npz"))
        else:
            # In the case of standard avatar transformations
            default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"))
        
        try:
            # Generate a field using RBF interpolation and save the deformation field data
            field_objects, target_world_vertices, displacements = create_shape_key_from_rbf(
                source_obj, 
                source_shape_key_name, 
                selected_only,
                epsilon,
                num_steps,
                source_avatar_name,
                target_avatar_name,
                save_shape_key_mode,
                keep_first_field,
                add_normal_control_points,
                normal_distance,
                shape_key_start_value,
                shape_key_end_value
            )

            filelist = []
            if default_paths[0] and os.path.exists(default_paths[0]):
                if os.path.abspath(default_paths[0]) != os.path.abspath(self.filepath):
                    shutil.copy2(default_paths[0], self.filepath)
                filelist.append(self.filepath)
            if scene.rbf_save_shape_key_mode and default_paths[1] and os.path.exists(default_paths[1]):
                inv_filepath = self.filepath[:-4] + "_inv.npz"
                if os.path.abspath(default_paths[1]) != os.path.abspath(inv_filepath):
                    shutil.copy2(default_paths[1], inv_filepath)
                filelist.append(self.filepath[:-4] + "_inv.npz")

            self.report({'INFO'}, f"Deformation data saved: {', '.join(filelist)}")
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"An error has occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        # Set the default filename
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_shape_key_name = scene.rbf_source_shape_key
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        
        filename = "deformation.npz"
        if source_avatar_name:
            if save_shape_key_mode:
                # In Shape Key mode
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}"
            elif target_avatar_name:
                # In the case of the standard transformation mode
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}"
            self.filepath = filename + ".npz"

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Operator for applying saved field data
class APPLY_OT_FieldData(bpy.types.Operator):
    bl_idname = "rbf.apply_field_data"
    bl_label = "Apply Field Data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        # Generate file paths based on the current settings (backward compatibility supported)
        scene_folder = get_scene_folder()
        if save_shape_key_mode:
            # In ShapeKey deformation mode
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name
            )
        else:
            # In the case of standard avatar transformations
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            target_obj = context.active_object
            if not target_obj or target_obj.type != 'MESH':  
                self.report({'ERROR'}, "Please select a Mesh object")
                return {'CANCELLED'}
            
            shape_key_name = scene.rbf_apply_shape_key_name if scene.rbf_apply_shape_key_name else "RBFDeform"
            apply_field_data(target_obj, field_data_path, shape_key_name)
            self.report({'INFO'}, f"{display_name} applied: {os.path.basename(field_data_path)}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Base Pose Difference Saving Operator
class SAVE_OT_BasePoseDiff(bpy.types.Operator, ExportHelper):
    bl_idname = "object.save_base_pose_diff"
    bl_label = "Save Base Pose"
    bl_options = {'REGISTER', 'UNDO'}
    
    # ExportHelper Properties
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        
        # Avatar Name Validation
        if not source_avatar_name:
            self.report({'ERROR'}, "Please set source avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # Get the active object
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # Check if the active object is an Armature
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # Save to the specified path
        filepath = self.filepath
        
        # Convert the path of the avatar data file to an absolute path
        avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            # Base Pose Difference Saving Operator
            filename = os.path.basename(filepath)
            temp_dir = os.path.dirname(filepath)
            
            # Save the data using the original function
            saved_filepath = save_armature_pose(armature_obj, filename, avatar_data_filename)
            
            # Move to the specified location
            if saved_filepath != filepath and os.path.abspath(saved_filepath) != os.path.abspath(filepath):
                shutil.copy2(saved_filepath, filepath)

            self.report({'INFO'}, f"Base pose data saved: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        # Set the default filename
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        if source_avatar_name:
            self.filepath = f"pose_basis_{normalize_avatar_name_for_filename(source_avatar_name)}.json"
        else:
            self.filepath = "pose_basis.json"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Operator for Applying Base Pose Variations
class APPLY_OT_BasePoseDiff(bpy.types.Operator):
    bl_idname = "object.apply_base_pose_diff"
    bl_label = "Apply Base Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        invert = scene.rbf_pose_invert
        
        # Avatar Name Validation
        if not source_avatar_name:
            self.report({'ERROR'}, "Please set source avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # Get the active object
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # Check if the active object is an Armature
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # Automatically generate file names
        pose_filename = f"pose_basis_{normalize_avatar_name_for_filename(source_avatar_name)}.json"
        
        # Convert the path of the avatar data file to an absolute path
        if invert:
            avatar_data_filename = bpy.path.abspath(target_avatar_data_file)
        else:
            avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            add_pose_from_json(pose_filename, avatar_data_filename, invert)
            action = "inverse applied" if invert else "applied"
            self.report({'INFO'}, f"Base pose data {action}: {pose_filename}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Pose Save Operator
class SAVE_OT_PoseDiff(bpy.types.Operator, ExportHelper):
    bl_idname = "object.save_pose_diff"
    bl_label = "Save Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    # ExportHelper Properties
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        
        # Avatar Name Validation
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # Get the active object
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # Check if the active object is an Armature
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # Save to the specified path
        filepath = self.filepath
        
        # Convert the path of the avatar data file to an absolute path
        avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            # Retrieve the filename to use the original 'save_armature_pose' function
            filename = os.path.basename(filepath)
            temp_dir = os.path.dirname(filepath)
            
            # Save the data using the original function
            saved_filepath = save_armature_pose(armature_obj, filename, avatar_data_filename)
            
            # Move to the specified location
            if saved_filepath != filepath and os.path.abspath(saved_filepath) != os.path.abspath(filepath):
                shutil.copy2(saved_filepath, filepath)

            self.report({'INFO'}, f"Pose data saved: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}

    def invoke(self, context, event):
        # Set the default filename
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        if source_avatar_name and target_avatar_name:
            self.filepath = f"posediff_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}.json"
        else:
            self.filepath = "posediff.json"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Pose Application Operator
class APPLY_OT_PoseDiff(bpy.types.Operator):
    bl_idname = "object.apply_pose_diff"
    bl_label = "Apply Pose Difference"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        invert = scene.rbf_pose_invert
        
        # Avatar Name Validation
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}
        
        if not source_avatar_data_file:
            self.report({'ERROR'}, "Please specify source avatar data file")
            return {'CANCELLED'}
        
        # Get the active object
        active_obj = context.active_object
        if not active_obj:
            self.report({'ERROR'}, "Please select an object")
            return {'CANCELLED'}
        
        # Check if the active object is an Armature
        if active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = active_obj
        
        # Automatically generate file names
        pose_filename = f"posediff_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}.json"
        
        # Convert the path of the avatar data file to an absolute path
        if invert:
            avatar_data_filename = bpy.path.abspath(target_avatar_data_file)
        else:
            avatar_data_filename = bpy.path.abspath(source_avatar_data_file)
        
        try:
            add_pose_from_json(pose_filename, avatar_data_filename, invert)
            action = "inverse applied" if invert else "applied"
            self.report({'INFO'}, f"Pose data {action}: {pose_filename}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Avatar Settings Switcher
class SWAP_OT_AvatarSettings(bpy.types.Operator):
    bl_idname = "object.swap_avatar_settings"
    bl_label = "Swap Avatar Settings"
    bl_description = "Swap the source and target avatar settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Get the current value
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_avatar_data_file = scene.rbf_source_avatar_data_file
        target_avatar_data_file = scene.rbf_target_avatar_data_file
        
        # Swap the values
        scene.rbf_source_avatar_name = target_avatar_name
        scene.rbf_target_avatar_name = source_avatar_name
        scene.rbf_source_avatar_data_file = target_avatar_data_file
        scene.rbf_target_avatar_data_file = source_avatar_data_file
        
        self.report({'INFO'}, "Avatar settings swapped")
        return {'FINISHED'}


class SET_OT_HumanoidBoneInheritScale(bpy.types.Operator):
    bl_idname = "object.set_humanoid_bone_inherit_scale"
    bl_label = "Set Humanoid Bone Inherit Scale"
    bl_description = "Set the 'Inherit Scale' property of the selected humanoid bones in the Armature to 'Average'"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        
        # Check if the active object is an Armature
        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select an Armature object")
            return {'CANCELLED'}
        
        armature_obj = context.active_object
        
        # Check if the source avatar data file is configured
        if not scene.rbf_source_avatar_data_file:
            self.report({'ERROR'}, "Please set source avatar data file")
            return {'CANCELLED'}
        
        try:
            # Load avatar data
            avatar_data = load_avatar_data(scene.rbf_source_avatar_data_file)
            
            # Retrieve humanoid bone information
            bone_parents, humanoid_to_bone, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)
            
            # Switch to Edit Mode
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='EDIT')
            
            modified_count = 0
            
            # Set "Inherit Scale" for each Humanoid bone
            for humanoid_bone_name, bone_name in humanoid_to_bone.items():
                if bone_name in armature_obj.data.edit_bones:
                    edit_bone = armature_obj.data.edit_bones[bone_name]
                    
                    # Set only if 'Inherit Scale' is not 'None'
                    if edit_bone.inherit_scale != 'NONE':
                        # Set the humanoid bones for the upper chest, chest, toes, and toes to "Full"
                        if 'Breast' in humanoid_bone_name or 'UpperChest' in humanoid_bone_name or 'Toe' in humanoid_bone_name or ('Foot' in humanoid_bone_name and ('Index' in humanoid_bone_name or 'Little' in humanoid_bone_name or 'Middle' in humanoid_bone_name or 'Ring' in humanoid_bone_name or 'Thumb' in humanoid_bone_name)):
                            edit_bone.inherit_scale = 'FULL'
                        else:
                            edit_bone.inherit_scale = 'AVERAGE'
                        modified_count += 1
            
            # Return to ObjectMode
            bpy.ops.object.mode_set(mode='OBJECT')
            
            if modified_count > 0:
                self.report({'INFO'}, f"Set Inherit Scale to Average for {modified_count} Humanoid bones")
            else:
                self.report({'INFO'}, "No bones needed modification")

            return {'FINISHED'}

        except Exception as e:
            # If an error occurs, return to ObjectMode
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except:
                pass
            self.report({'ERROR'}, f"An error occurred: {str(e)}")
            return {'CANCELLED'}


# Registered Functions
def register():
    bpy.utils.register_class(RBF_PT_DeformationPanel)
    bpy.utils.register_class(SELECT_OT_RBFShapeKey)
    bpy.utils.register_class(SET_OT_RBFShapeKey)
    bpy.utils.register_class(CREATE_OT_RBFDeformation)
    bpy.utils.register_class(EXPORT_OT_RBFTempData)
    bpy.utils.register_class(APPLY_OT_FieldData)
    bpy.utils.register_class(APPLY_OT_InverseFieldData)
    bpy.utils.register_class(CREATE_OT_FieldVisualization)
    bpy.utils.register_class(SAVE_OT_BasePoseDiff)
    bpy.utils.register_class(APPLY_OT_BasePoseDiff)
    bpy.utils.register_class(SAVE_OT_PoseDiff)
    bpy.utils.register_class(APPLY_OT_PoseDiff)
    bpy.utils.register_class(SWAP_OT_AvatarSettings)
    bpy.utils.register_class(SET_OT_HumanoidBoneInheritScale)
    bpy.utils.register_class(DEBUG_OT_ShowPythonPaths)
    bpy.utils.register_class(DEBUG_OT_TestExternalPython)
    bpy.utils.register_class(REINSTALL_OT_NumpyScipyMultithreaded)
    register_properties()


# Unregister function
def unregister():
    bpy.utils.unregister_class(RBF_PT_DeformationPanel)
    bpy.utils.unregister_class(SELECT_OT_RBFShapeKey)
    bpy.utils.unregister_class(SET_OT_RBFShapeKey)
    bpy.utils.unregister_class(CREATE_OT_RBFDeformation)
    bpy.utils.unregister_class(EXPORT_OT_RBFTempData)
    bpy.utils.unregister_class(APPLY_OT_FieldData)
    bpy.utils.unregister_class(APPLY_OT_InverseFieldData)
    bpy.utils.unregister_class(CREATE_OT_FieldVisualization)
    bpy.utils.unregister_class(SAVE_OT_BasePoseDiff)
    bpy.utils.unregister_class(APPLY_OT_BasePoseDiff)
    bpy.utils.unregister_class(SAVE_OT_PoseDiff)
    bpy.utils.unregister_class(APPLY_OT_PoseDiff)
    bpy.utils.unregister_class(SWAP_OT_AvatarSettings)
    bpy.utils.unregister_class(SET_OT_HumanoidBoneInheritScale)
    bpy.utils.unregister_class(DEBUG_OT_ShowPythonPaths)
    bpy.utils.unregister_class(DEBUG_OT_TestExternalPython)
    bpy.utils.unregister_class(REINSTALL_OT_NumpyScipyMultithreaded)
    
    # Deleting a Property
    del bpy.types.Scene.rbf_source_obj
    del bpy.types.Scene.rbf_source_shape_key
    del bpy.types.Scene.rbf_selected_only
    del bpy.types.Scene.rbf_save_shape_key_mode
    del bpy.types.Scene.rbf_keep_first_field
    del bpy.types.Scene.rbf_epsilon
    del bpy.types.Scene.rbf_num_steps
    del bpy.types.Scene.rbf_apply_shape_key_name
    del bpy.types.Scene.rbf_base_grid_spacing
    del bpy.types.Scene.rbf_surface_distance
    del bpy.types.Scene.rbf_max_distance
    del bpy.types.Scene.rbf_min_distance
    del bpy.types.Scene.rbf_density_falloff
    del bpy.types.Scene.rbf_bbox_scale_factor
    del bpy.types.Scene.rbf_source_avatar_name
    del bpy.types.Scene.rbf_target_avatar_name
    del bpy.types.Scene.rbf_pose_invert
    del bpy.types.Scene.rbf_source_avatar_data_file
    del bpy.types.Scene.rbf_target_avatar_data_file
    del bpy.types.Scene.rbf_add_normal_control_points
    del bpy.types.Scene.rbf_normal_distance
    del bpy.types.Scene.rbf_show_debug_info
    del bpy.types.Scene.rbf_field_step
    del bpy.types.Scene.rbf_field_use_inverse
    del bpy.types.Scene.rbf_field_object_name


class APPLY_OT_InverseFieldData(bpy.types.Operator):
    """Operator for applying inverse transformation data"""
    bl_idname = "rbf.apply_inverse_field_data"
    bl_label = "Apply Inverse Field Data"
    bl_description = "Apply inverse transformation data"
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        # Generate file paths based on the current settings (reverse transformation, backward compatibility supported)
        scene_folder = get_scene_folder()
        if save_shape_key_mode:
            # In ShapeKey deformation mode
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Inverse shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name,
                inverse_suffix="_inv"
            )
        else:
            # In the case of standard avatar transformations
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inverse inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name,
                inverse_suffix="_inv"
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}_inv.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Inverse deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            target_obj = context.active_object
            if not target_obj or target_obj.type != 'MESH':
                self.report({'ERROR'}, "Please select a Mesh object")
                return {'CANCELLED'}
            
            shape_key_name = scene.rbf_apply_shape_key_name if scene.rbf_apply_shape_key_name else "RBFDeform_inv"
            apply_field_data(target_obj, field_data_path, shape_key_name)
            self.report({'INFO'}, f"{display_name} applied: {os.path.basename(field_data_path)}")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


def export_rbf_temp_data(source_obj, source_shape_key_name, selected_only=True, epsilon=0.0, num_steps=1, source_avatar_name="", target_avatar_name="", save_shape_key_mode=False, add_normal_control_points=False, normal_distance=-0.0002, shape_key_start_value=0.0, shape_key_end_value=1.0, enable_x_mirror=True):
    """
    Export the temporary data required for RBF processing
    
    Parameters:
    - source_obj: Source object (with shape keys)
    - source_shape_key_name: Name of the shape key on the source object
    - selected_only: Whether to use only selected vertices as control points
    - epsilon: RBF parameter (calculated automatically if 0 or negative)
    - num_steps: Number of steps for subdivision
    - source_avatar_name: Name of the source avatar
    - target_avatar_name: Name of the target avatar
    - save_shape_key_mode: Shape key deformation mode (save both forward and reverse directions)
    - add_normal_control_points: Whether to place additional control points in the normal direction of the control points
    - normal_distance: Distance in the normal direction (world coordinate system)
    - shape_key_start_value: The starting value of the shape key
    - shape_key_end_value: The ending value of the shape key
    
    Returns:
    - The path to the temporary file
    """
    
    # Check the display status of the target object and set it to "Display" if necessary
    armature_obj = get_armature_from_source_object(source_obj)
    objects_to_check = [source_obj]
    if armature_obj:
        objects_to_check.append(armature_obj)
    
    original_visibility_states = ensure_objects_visible(objects_to_check)
    
    try:
        # List of directions to save (determined based on 'save_shape_key_mode')
        if save_shape_key_mode:
            directions = [False, True]  # Both standard and reverse transformations
        else:
            directions = [False]  # Standard transformations only
        
        results = []
        
        for invert in directions:
            direction_suffix = "_inv" if invert else ""
            print(f"\n=== Starting temporary data preparation for {'inverse' if invert else 'normal'} deformation ===")
            
            # Automatically generate a temporary data storage path
            scene_folder = get_scene_folder()
            
            if save_shape_key_mode:
                # In ShapeKey deformation mode
                temp_data_path = os.path.join(scene_folder, f"temp_rbf_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}{direction_suffix}.npz")
            else:
                # In the case of standard avatar transformations
                temp_data_path = os.path.join(scene_folder, f"temp_rbf_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}{direction_suffix}.npz")
            
            # Save the ShapeKey values
            original_values = {}
            for key in source_obj.data.shape_keys.key_blocks:
                original_values[key.name] = key.value
                key.value = 0.0
            
            # Set the initial state based on the Invert option
            if invert:
                # Use the end value of the shape key as a reference
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_end_value
            else:
                # Use the initial value of ShapeKey as the reference
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = shape_key_start_value
            
            # Refresh the scene
            bpy.context.view_layer.update()
            
            # Retrieve the depth graph after evaluation
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            # Get the object resulting from the evaluation of the source object
            evaluated_source = source_obj.evaluated_get(depsgraph)
            
            # Get the shape keys of the source object
            if source_obj.data.shape_keys is None or source_shape_key_name not in source_obj.data.shape_keys.key_blocks:
                raise ValueError(f"The shape key '{source_shape_key_name}' was not found in the source object.")
            
            # Get the world matrix of the source object
            source_world_matrix = source_obj.matrix_world
            
            # Get the index of the vertex to be used as a control point
            original_selected_vertices = []  # Record the original selected vertices
            if selected_only:
                # Switch to Object Mode to apply the selection made in Edit Mode
                was_edit_mode = False
                if bpy.context.object == source_obj and bpy.context.object.mode == 'EDIT':
                    was_edit_mode = True
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # Check if a vertex has been selected
                original_selected_vertices = [i for i, v in enumerate(source_obj.data.vertices) if v.select]
                
                # Return to edit mode
                if was_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT')
                
                if len(original_selected_vertices) == 0:
                    raise ValueError("No vertices have been selected. Please select at least one vertex.")
                
                # Use all vertices within the scaled bounding box calculated from the selected vertex as control points
                selected_indices = get_vertices_in_scaled_bbox(source_obj, bpy.context.scene.rbf_bbox_scale_factor)
                
                if len(selected_indices) < 4:
                    print(f"Warning: Very few control points ({len(selected_indices)}). Consider selecting more control points.")
            else:
                # Use all vertices
                selected_indices = list(range(len(source_obj.data.vertices)))
            
            # Identify duplicate control points with Shape Key values of 0 and 1 in advance
            print("Pre-checking for duplicate control points at shape key values 0 and 1...")
            overlapping_indices = identify_overlapping_control_points_for_shape_keys(
                source_obj, 
                source_shape_key_name, 
                selected_indices, 
                source_world_matrix, 
                add_normal_control_points, 
                normal_distance
            )
            
            # Collect field data and transformation data at each step
            all_step_data = []
            all_field_world_vertices = []
            
            for step in range(num_steps):
                print(f"\n=== Collecting data for step {step+1}/{num_steps} ===")
                
                # Calculate the value of the current step
                progress = (step + 1) / num_steps
                if invert:
                    # In Invert mode, the value changes from the end value to the start value
                    step_value = shape_key_end_value - (shape_key_end_value - shape_key_start_value) * progress
                else:
                    # In normal mode, the value changes from the start value to the end value
                    step_value = shape_key_start_value + (shape_key_end_value - shape_key_start_value) * progress
                
                print(f"Shape key value: {step_value}")
                
                # Filter control points based on vertex groups
                filtered_indices = filter_control_points_by_vertex_groups(source_obj, selected_indices, step_value)
                
                if len(filtered_indices) < 4:
                    print(f"Warning: Very few valid control points at step {step+1} ({len(filtered_indices)}).")
                    if len(filtered_indices) == 0:
                        print(f"Skipping step {step+1}: No valid control points.")
                        continue
                
                print(f"Control points: {len(selected_indices)} -> {len(filtered_indices)} (after vertex group filtering)")
                
                # Retrieve the pre-transformation state (for bounding box calculation)
                current_basis_local = np.array([evaluated_source.data.vertices[i].co.copy() for i in filtered_indices])
                
                # Convert the pre-transformation state to world coordinates
                current_basis = np.zeros_like(current_basis_local)
                for i, basis_co in enumerate(current_basis_local):
                    basis_v = Vector((basis_co[0], basis_co[1], basis_co[2], 1.0))
                    world_basis = source_world_matrix @ basis_v
                    current_basis[i] = np.array([world_basis[0], world_basis[1], world_basis[2]])
                
                # Generate fields for the current step (using the source object before transformation)
                print(f"Generating Deformation Field for step {step+1}...")
                field_vertices = create_adaptive_deformation_field(
                    target_obj=source_obj,
                    base_grid_spacing=bpy.context.scene.rbf_base_grid_spacing,
                    surface_distance=bpy.context.scene.rbf_surface_distance,
                    max_distance=bpy.context.scene.rbf_max_distance,
                    min_distance=bpy.context.scene.rbf_min_distance,
                    density_falloff=bpy.context.scene.rbf_density_falloff,
                    bbox_scale_factor=bpy.context.scene.rbf_bbox_scale_factor,
                    use_selected_vertices=selected_only
                )
                
                if field_vertices is None:
                    print(f"Failed to generate field at step {step+1}")
                    continue
                
                # Convert a Vector object to a Python array (to make it pickleable)
                field_vertices_array = np.array([[v.x, v.y, v.z] for v in field_vertices])
                all_field_world_vertices.append(field_vertices_array)
                
                # Update the ShapeKey values to obtain the transformed state
                source_obj.data.shape_keys.key_blocks[source_shape_key_name].value = step_value
                
                # Refresh the scene
                bpy.context.view_layer.update()
                
                # Retrieve the object after evaluation
                depsgraph.update()
                evaluated_source_deformed = source_obj.evaluated_get(depsgraph)
                
                # Get the vertex positions after transformation
                current_deformed_local = np.array([evaluated_source_deformed.data.vertices[i].co.copy() for i in filtered_indices])
                
                # Convert the transformed position to world coordinates
                current_deformed = np.zeros_like(current_deformed_local)
                for i, deformed_co in enumerate(current_deformed_local):
                    deformed_v = Vector((deformed_co[0], deformed_co[1], deformed_co[2], 1.0))
                    world_deformed = source_world_matrix @ deformed_v
                    current_deformed[i] = np.array([world_deformed[0], world_deformed[1], world_deformed[2]])
                
                # When adding control points in the normal direction
                if add_normal_control_points:
                    current_basis_extended, current_deformed_extended = add_normal_control_points_func(
                        source_obj, 
                        filtered_indices, 
                        current_basis, 
                        current_deformed, 
                        normal_distance
                    )
                else:
                    current_basis_extended = current_basis
                    current_deformed_extended = current_deformed
                
                # Exclude pre-identified duplicate control points
                selected_indices_updated = filtered_indices
                if len(overlapping_indices) > 0:
                    print(f"Excluding {len(overlapping_indices)} pre-identified duplicate control points")
                    # Get the indices of non-overlapping control points
                    all_indices = np.arange(len(current_basis_extended))
                    valid_indices = np.setdiff1d(all_indices, overlapping_indices)
                    
                    if len(valid_indices) < len(current_basis_extended):
                        # You also need to update 'filtered_indices' in the same way.
                        if len(filtered_indices) == len(current_basis_extended):
                            selected_indices_updated = np.array(filtered_indices)[valid_indices].tolist()
                        else:
                            selected_indices_updated = filtered_indices
                        
                        current_basis_extended = current_basis_extended[valid_indices]
                        current_deformed_extended = current_deformed_extended[valid_indices]
                        print(f"Excluded duplicate control points: using {len(valid_indices)} control points")
                
                # Save step data
                step_data = {
                    'step_value': step_value,
                    'control_points_original': current_basis_extended,
                    'control_points_deformed': current_deformed_extended,
                    'selected_indices': selected_indices_updated
                }
                
                all_step_data.append(step_data)
                
                print(f"Step {step+1} data collection complete")
            
            # Restore the ShapeKey values
            for key_name, value in original_values.items():
                source_obj.data.shape_keys.key_blocks[key_name].value = value
            
            # Refresh the scene
            bpy.context.view_layer.update()
            
            # Save temporary data
            temp_data = {
                'all_field_world_vertices': all_field_world_vertices if len(all_field_world_vertices) == 1 else np.array(all_field_world_vertices, dtype=object),  # Field coordinates for each step
                'field_world_matrix': np.identity(4),
                'all_step_data': np.array(all_step_data, dtype=object),
                'source_world_matrix': np.array(source_world_matrix),
                'epsilon': epsilon,
                'num_steps': num_steps,
                'invert': invert,
                'source_avatar_name': source_avatar_name,
                'target_avatar_name': target_avatar_name,
                'source_shape_key_name': source_shape_key_name,
                'save_shape_key_mode': save_shape_key_mode,
                'add_normal_control_points': add_normal_control_points,
                'normal_distance': normal_distance,
                'shape_key_start_value': shape_key_start_value,  # Add shape key start value
                'shape_key_end_value': shape_key_end_value,      # Add shape key end value
                'original_selected_vertices': original_selected_vertices,  # Add original selected vertices
                'selected_only': selected_only,  # Add the selected_only flag
                'rbf_base_grid_spacing': bpy.context.scene.rbf_base_grid_spacing,
                'rbf_surface_distance': bpy.context.scene.rbf_surface_distance,
                'rbf_max_distance': bpy.context.scene.rbf_max_distance,
                'rbf_min_distance': bpy.context.scene.rbf_min_distance,
                'rbf_density_falloff': bpy.context.scene.rbf_density_falloff,
                'rbf_bbox_scale_factor': bpy.context.scene.rbf_bbox_scale_factor,
                'enable_x_mirror': enable_x_mirror
            }
            
            # Save in NumPy format
            np.savez(temp_data_path, **temp_data)
            
            print(f"Saved temporary data: {temp_data_path}")
            results.append(temp_data_path)
        
        return results
    
    finally:
        # After processing is complete, restore the object's display state
        restore_objects_visibility(objects_to_check, original_visibility_states)

# Operator for Temporary Data Export
class EXPORT_OT_RBFTempData(bpy.types.Operator, ExportHelper):
    bl_idname = "object.export_rbf_temp_data"
    bl_label = "Save Deformation Data"
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper Properties
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,
    )

    # Class variable for modal handling
    _timer = None
    _thread = None
    _process = None
    _queue = None
    _progress = 0.0
    _status_message = ""
    _default_paths = None
    _save_shape_key_mode = False
    _dot_count = 0
    _temp_file_paths = None  # Phase 2: Save the temporary file path (for cleanup in case of cancellation)
    # Phase 3: For UI Enhancements Related to Progress
    _current_phase = ""  # Current phase name (Distance Calculation/Deformation/Falloff)
    _progress_started = False  # Has 'window_manager.progress' started?

    def modal(self, context, event):
        import queue as queue_module
        import re

        if event.type == 'TIMER':
            # If the queue is 'None' after cancellation, skip it
            if not self._queue:
                return {'PASS_THROUGH'}

            # Animation Update
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * (self._dot_count + 1)

            # Retrieve logs from the queue in a non-blocking manner
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item[0] == 'LOG':
                        line = item[1]
                        print(f"[RBF Processing] {line}")

                        # Phase 3: Phase Detection
                        if 'Calculate Distance Progress' in line or 'Calculate the distance' in line:
                            self._current_phase = "Distance Calculation"
                        elif 'Start multi-process RBF interpolation' in line:
                            self._current_phase = "Deformation"
                        elif 'Applying falloff' in line:
                            self._current_phase = "Falloff"
                            self._progress = 95.0  # The falloff is in its final stages
                            # Phase 3: Update the progress bar immediately when the falloff begins
                            if self._progress_started:
                                context.window_manager.progress_update(95)

                        # Progress Report (e.g., "Progress: 1,000/10,000 vertices processed (10.0%)")
                        match = re.search(r'\((\d+\.?\d*)%\)', line)
                        if match:
                            raw_progress = float(match.group(1))
                            # Phase 3: Calculate overall progress based on the phase
                            # Distance calculation: 0 - 30%, Deformation: 30- 95%, Falloff: 95 - 100%
                            if self._current_phase == "Distance Calculation":
                                self._progress = raw_progress * 0.30
                            elif self._current_phase == "Deformation":
                                self._progress = 30.0 + raw_progress * 0.65
                            else:
                                self._progress = raw_progress

                            # Phase 3: window_manager.progress Update
                            if self._progress_started:
                                context.window_manager.progress_update(int(self._progress))

                        # Update status message (last 50 characters)
                        self._status_message = line[-50:] if len(line) > 50 else line
                    elif item[0] == 'DONE':
                        returncode = item[1]
                        # Phase 3: Display 100% upon completion
                        if self._progress_started:
                            self._progress = 100.0
                            context.window_manager.progress_update(100)
                        self._finish(context, returncode == 0)
                        return {'FINISHED'}
                    elif item[0] == 'ERROR':
                        error_msg = item[1]
                        self._finish_with_error(context, error_msg)
                        return {'CANCELLED'}
            except queue_module.Empty:
                pass

            # Phase 3: UI Update (including phase name)
            if self._progress > 0:
                phase_str = f"[{self._current_phase}] " if self._current_phase else ""
                context.workspace.status_text_set(
                    f"RBF Processing{dots} {phase_str}{self._progress:.1f}%"
                )
            else:
                context.workspace.status_text_set(f"RBF Processing{dots}")

        elif event.type == 'ESC':
            self._cancel_process(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        import threading
        import queue as queue_module

        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        scene = context.scene

        # Retrieve the required parameters
        source_obj = scene.rbf_source_obj
        source_shape_key_name = scene.rbf_source_shape_key
        selected_only = scene.rbf_selected_only
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        epsilon = scene.rbf_epsilon
        num_steps = scene.rbf_num_steps
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        add_normal_control_points = scene.rbf_add_normal_control_points
        normal_distance = scene.rbf_normal_distance
        shape_key_start_value = scene.rbf_shape_key_start_value
        shape_key_end_value = scene.rbf_shape_key_end_value
        enable_x_mirror = scene.rbf_enable_x_mirror

        # Avatar Name Validation
        if not source_avatar_name or not target_avatar_name:
            self.report({'ERROR'}, "Please set avatar name")
            return {'CANCELLED'}

        # Validation of ShapeKey Value Ranges
        if shape_key_start_value == shape_key_end_value:
            self.report({'ERROR'}, "Shape key start and end values must be different")
            return {'CANCELLED'}


        self._default_paths = []
        scene_folder = get_scene_folder()
        self._save_shape_key_mode = save_shape_key_mode

        if scene.rbf_save_shape_key_mode:
            # In ShapeKey deformation mode
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}.npz"))
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_shape_{scene.rbf_source_shape_key}_inv.npz"))
        else:
            # In the case of standard avatar transformations
            self._default_paths.append(os.path.join(scene_folder, f"deformation_{normalize_avatar_name_for_filename(scene.rbf_source_avatar_name)}_to_{normalize_avatar_name_for_filename(scene.rbf_target_avatar_name)}.npz"))

        try:
            # Export temporary data (synchronization process, no issues due to high speed)
            self._temp_file_paths = export_rbf_temp_data(
                source_obj,
                source_shape_key_name,
                selected_only,
                epsilon,
                num_steps,
                source_avatar_name,
                target_avatar_name,
                save_shape_key_mode,
                add_normal_control_points,
                normal_distance,
                shape_key_start_value,
                shape_key_end_value,
                enable_x_mirror
            )

            # Generate information about the saved file
            file_list = ", ".join([os.path.basename(path) for path in self._temp_file_paths])
            self.report({'INFO'}, f"Temporary data exported: {file_list}")

            base_temp_path = self._temp_file_paths[0]

            print(f"\n{'='*60}")
            print(f"RBF processing started: {os.path.basename(base_temp_path)}")
            print(f"{'='*60}")

            # Initialize Queue
            self._queue = queue_module.Queue()
            self._progress = 0.0
            self._status_message = ""
            self._dot_count = 0
            # Phase 3: Initialization for Progress UI Enhancements
            self._current_phase = ""
            self._progress_started = True
            context.window_manager.progress_begin(0, 100)

            # Values that must be retrieved in advance on the main thread
            python_path = get_blender_python_path()
            processor_path = get_rbf_processor_script_path()
            blender_lib_paths = get_blender_python_lib_paths()
            user_site_packages = get_blender_python_user_site_packages(python_path)
            blender_deps_path = os.path.join(os.path.dirname(__file__), 'deps')
            filepath = self.filepath  # The file path configured in ExportHelper

            # Functions to run in a background thread
            def run_rbf_background():
                # To avoid conflicts during cancellation, store the queue as a local reference
                # (This ensures safe operation even if 'self._queue = None' is set in '_cleanup() ')
                q = self._queue

                try:
                    # Verifying the existence of a path
                    if not os.path.exists(python_path):
                        if q:
                            q.put(('ERROR', f"Python binary not found: {python_path}"))
                        return

                    if not os.path.exists(processor_path):
                        if q:
                            q.put(('ERROR', f"RBF processor script not found: {processor_path}"))
                        return

                    # Set environment variables
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
                    env['PYTHONUNBUFFERED'] = '1'

                    # Set PYTHONPATH
                    pythonpath_parts = []
                    if 'PYTHONPATH' in env:
                        pythonpath_parts.append(env['PYTHONPATH'])
                    if blender_deps_path:
                        pythonpath_parts.append(blender_deps_path)
                    if user_site_packages:
                        pythonpath_parts.append(user_site_packages)
                    pythonpath_parts.extend(blender_lib_paths)
                    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)

                    # Phase 1-B: Limit the number of BLAS threads to a fixed value (to prevent oversubscription)
                    env['OMP_NUM_THREADS'] = '2'
                    env['OPENBLAS_NUM_THREADS'] = '2'
                    env['MKL_NUM_THREADS'] = '2'
                    env['VECLIB_MAXIMUM_THREADS'] = '2'
                    env['NUMEXPR_NUM_THREADS'] = '2'

                    # Phase 1-B: Limit the number of workers
                    max_workers = min(4, os.cpu_count() or 4)

                    # Limit the number of workers
                    cmd = [python_path, '-u', processor_path, base_temp_path,
                           '--max-workers', str(max_workers)]

                    print(f"Executing command: {' '.join(cmd)}")
                    print(f"max_workers: {max_workers}, OMP_NUM_THREADS: 2")

                    # Start the process
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        cwd=os.path.dirname(base_temp_path),
                        env=env,
                        bufsize=1,
                        universal_newlines=True
                    )

                    # Read from stdout and send to the queue
                    for line in iter(self._process.stdout.readline, ''):
                        if line and q:
                            q.put(('LOG', line.rstrip('\n\r')))

                    # Wait for the process to complete
                    self._process.wait()

                    # Copy the file if successful
                    if self._process.returncode == 0:
                        if self._default_paths[0] and os.path.exists(self._default_paths[0]):
                            if os.path.abspath(self._default_paths[0]) != os.path.abspath(filepath):
                                shutil.copy2(self._default_paths[0], filepath)
                        if self._save_shape_key_mode and len(self._default_paths) > 1 and self._default_paths[1] and os.path.exists(self._default_paths[1]):
                            inv_filepath = filepath[:-4] + "_inv.npz"
                            if os.path.abspath(self._default_paths[1]) != os.path.abspath(inv_filepath):
                                shutil.copy2(self._default_paths[1], inv_filepath)

                    if q:
                        q.put(('DONE', self._process.returncode))

                except Exception as e:
                    error_msg = f"Error during RBF processing: {str(e)}"
                    print(error_msg)
                    print(traceback.format_exc())
                    if q:
                        q.put(('ERROR', error_msg))

            # Start a background thread
            self._thread = threading.Thread(target=run_rbf_background)
            self._thread.start()

            # Set the timer (check every 0.1 seconds)
            self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)

            # Start displaying in the status bar
            context.workspace.status_text_set("Starting RBF processing...")

            self.report({'INFO'}, "Multiprocess processing started (running in background)")

            return {'RUNNING_MODAL'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            # Phase 3: Ensure the progress bar closes even when an exception occurs
            if self._progress_started:
                context.window_manager.progress_end()
                self._progress_started = False
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}

    def _finish(self, context, success):
        """Action upon completion"""
        # If the program exits normally, do not delete temporary files; only clear the references.
        self._temp_file_paths = None
        self._cleanup(context)

        if success:
            self.report({'INFO'}, "RBF processing completed successfully")
            print("RBF processing completed successfully")

            # Display a success pop-up
            def draw_success_popup(self, context):
                self.layout.label(text="Deformation Field generation completed.")

            context.window_manager.popup_menu(draw_success_popup, title="Complete", icon='CHECKMARK')
        else:
            self.report({'ERROR'}, "RBF processing failed")
            print("RBF processing failed")

        # Update the UI
        for area in context.screen.areas:
            area.tag_redraw()

    def _finish_with_error(self, context, error_msg):
        """Handling Error Termination"""
        # Clear only references even when the program terminates with an error (leave temporary files intact for debugging purposes)
        self._temp_file_paths = None
        self._cleanup(context)
        self.report({'ERROR'}, error_msg)
        print(f"RBF processing error: {error_msg}")

        # Update the UI
        for area in context.screen.areas:
            area.tag_redraw()

    def _cancel_process(self, context):
        """Cancel the process"""
        import sys

        if self._process:
            try:
                pid = self._process.pid
                print(f"Cancelling RBF processing (PID: {pid})...")

                # Phase 2: On Windows, use 'taskkill' to terminate processes, including child processes
                if sys.platform == 'win32':
                    try:
                        # /T: Child process has also terminated; /F: Force termination
                        kill_cmd = ['taskkill', '/T', '/F', '/PID', str(pid)]
                        result = subprocess.run(kill_cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            print(f"Terminated including child processes with taskkill (PID: {pid})")
                        else:
                            # If 'taskkill' fails (e.g., because the process has already terminated)
                            stderr_msg = result.stderr.decode('utf-8', errors='replace').strip()
                            print(f"taskkill exited with returncode={result.returncode}: {stderr_msg}")
                            print("Trying terminate()...")
                            self._process.terminate()
                    except subprocess.TimeoutExpired:
                        print("taskkill timed out. Trying terminate()...")
                        self._process.terminate()
                    except Exception as e:
                        print(f"Error with taskkill: {e}. Trying terminate()...")
                        self._process.terminate()
                else:
                    # On Unix-like systems, use 'terminate() '
                    self._process.terminate()

                print("RBF processing cancelled")
            except Exception as e:
                print(f"Error while terminating process: {e}")

        # Phase 2: Clean up temporary files
        self._cleanup_temp_files()

        self._cleanup(context)
        self.report({'WARNING'}, "RBF processing was cancelled")

    def _cleanup_temp_files(self):
        """Clean up temporary files (only when canceling)"""
        if self._temp_file_paths:
            for temp_path in self._temp_file_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Deleted temporary file: {os.path.basename(temp_path)}")
                except Exception as e:
                    print(f"Error deleting temporary file ({os.path.basename(temp_path)}): {e}")
            self._temp_file_paths = None

    def _cleanup(self, context):
        """Resource Cleanup"""
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        # Phase 3: Close the progress bar
        if self._progress_started:
            context.window_manager.progress_end()
            self._progress_started = False
        self._current_phase = ""
        self._process = None
        self._thread = None
        self._queue = None

    def cancel(self, context):
        """Handling Cancellations (Called from Blender)"""
        self._cancel_process(context)

    def invoke(self, context, event):
        # Set the default filename
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name
        target_avatar_name = scene.rbf_target_avatar_name
        source_shape_key_name = scene.rbf_source_shape_key
        save_shape_key_mode = scene.rbf_save_shape_key_mode

        filename = "deformation.npz"
        if source_avatar_name:
            if save_shape_key_mode:
                # In Shape Key mode
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_shape_{source_shape_key_name}"
            elif target_avatar_name:
                # In the case of the standard transformation mode
                filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_to_{normalize_avatar_name_for_filename(target_avatar_name)}"
            self.filepath = filename + ".npz"

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def safe_decode(data):
    """
    Safely decode binary data into text.
    To avoid UnicodeDecodeErrors that occur in Windows console output,
    it falls back in the following order: UTF-8 → CP932 → UTF-8 (replacement mode).

    Parameters:
        data (bytes): The binary data to be decoded

    Returns:
        str: The decoded text
    """
    if not data:
        return ""
    # First, try UTF-8
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    # Next, try CP932 (Shift-JIS)
    try:
        return data.decode('cp932')
    except UnicodeDecodeError:
        pass
    # Finally, in replacement mode, UTF-8
    return data.decode('utf-8', errors='replace')


def run_subprocess_safe(cmd, env=None, timeout=None, cwd=None):
    """
    Run 'subprocess' while avoiding 'UnicodeDecodeError'.
    On Windows, run in binary mode instead of text mode
    and decode manually.

    Parameters:
        cmd (list): Command to execute
        env (dict, optional): Environment variables
        timeout (int, optional): Timeout in seconds
        cwd (str, optional): Working directory

    Returns:
        tuple: (returncode, stdout_text, stderr_text)
    """
    creationflags = 0
    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NO_WINDOW

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        creationflags=creationflags
    )

    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_bytes, stderr_bytes = process.communicate()
        return -1, "", "Timeout"

    stdout_text = safe_decode(stdout_bytes)
    stderr_text = safe_decode(stderr_bytes)

    return process.returncode, stdout_text, stderr_text


def get_blender_python_path():
    """
    Get the path to the Python binary included in Blender
    
    Returns:
        str: The path to the Python binary
    """
    # Path to the Blender executable file
    blender_binary = bpy.app.binary_path
    blender_dir = os.path.dirname(blender_binary)
    
    # Setting up Python binary paths by OS
    system = platform.system()
    
    if system == "Windows":
        # Windows: Blender/{version}/python/bin/python.exe
        version = bpy.app.version_string[:4]  # "4.2" format
        python_path = os.path.join(blender_dir, version, "python", "bin", "python.exe")
        
        # Backup path (for a different directory structure)
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python", "bin", "python.exe")
            
        # If you still can't find it, look in the same directory as Blender
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python.exe")
            
    elif system == "Darwin":  # macOS
        # macOS: Blender.app/Contents/Resources/{version}/python/bin/python
        version = bpy.app.version_string[:4]
        python_path = os.path.join(blender_dir, "..", "Resources", version, "python", "bin", "python")
        
        # Backup path
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "..", "Resources", "python", "bin", "python")
            
    else:  # Linux
        # Linux: blender/{version}/python/bin/python
        version = bpy.app.version_string[:4]
        python_path = os.path.join(blender_dir, version, "python", "bin", "python")
        
        # Backup path
        if not os.path.exists(python_path):
            python_path = os.path.join(blender_dir, "python", "bin", "python")
    
    # Path Normalization
    python_path = os.path.abspath(python_path)
    
    return python_path


def get_rbf_processor_script_path():
    """
    Get the path to the rbf_multithread_processor.py script
    
    Returns:
        str: The path to rbf_multithread_processor.py
    """
    # Assuming it is in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processor_path = os.path.join(current_dir, "rbf_multithread_processor.py")
    
    # If you can't find it, check the directory where the Blender file is located
    if not os.path.exists(processor_path):
        scene_folder = get_scene_folder()
        processor_path = os.path.join(scene_folder, "rbf_multithread_processor.py")
    
    return processor_path


def get_blender_python_user_site_packages(python_path=None):
    """
    Retrieving the location of packages installed via --user in Blender Python
    
    Parameters:
        python_path (str, optional): Path to the Python binary
    
    Returns:
        str: Path to the user-site packages; None if not found
    """
    try:
        if python_path is None:
            python_path = get_blender_python_path()
        
        if not os.path.exists(python_path):
            return None
            
        # Getting the user's site package directory in Python
        cmd = [python_path, '-c', 'import site; print(site.getusersitepackages())']

        try:
            returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=10)

            if returncode == 0:
                user_site_path = stdout.strip()
                if user_site_path and os.path.exists(user_site_path):
                    return user_site_path
        except:
            pass

        # Fallback: Manually construct the path
        if platform.system() == "Windows":
            # Get the Python version
            try:
                version_cmd = [python_path, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")']
                returncode, stdout, stderr = run_subprocess_safe(version_cmd, timeout=5)

                if returncode == 0:
                    python_version = stdout.strip()
                    # Windows: %APPDATA%\Python\PythonXX\site-packages
                    appdata = os.environ.get('APPDATA', '')
                    if appdata:
                        user_site_path = os.path.join(appdata, 'Python', f'Python{python_version.replace(".", "")}', 'site-packages')
                        if os.path.exists(user_site_path):
                            return user_site_path
            except:
                pass
                
    except Exception as e:
        print(f"Failed to get user site packages: {e}")
        
    return None


def get_blender_python_lib_paths():
    """
    Get the Python library path in Blender
    
    Returns:
        list: A list of Python library paths
    """
    import site
    
    # Path to the Blender executable file
    blender_binary = bpy.app.binary_path
    blender_dir = os.path.dirname(blender_binary)
    
    # Setting up Python library paths by OS
    system = platform.system()
    lib_paths = []
    
    # Get the Blender version
    version = bpy.app.version_string[:3]  # "Version 4.0"
    
    # Get the Blender path in the user directory
    def get_user_blender_path():
        if system == "Windows":
            appdata = os.environ.get('APPDATA', '')
            if appdata:
                return os.path.join(appdata, "Blender Foundation", "Blender", version)
        elif system == "Darwin":  # macOS
            home = os.path.expanduser("~")
            return os.path.join(home, "Library", "Application Support", "Blender", version)
        else:  # Linux
            home = os.path.expanduser("~")
            return os.path.join(home, ".config", "blender", version)
        return None
    
    user_blender_path = get_user_blender_path()
    
    if system == "Windows":
        # Windows: Supports the specified directory structure
        lib_paths.extend([
            # Scripts (Blender installation directory)
            os.path.join(blender_dir, version, "scripts", "startup"),
            os.path.join(blender_dir, version, "scripts", "modules"), 
            os.path.join(blender_dir, version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons"),
            os.path.join(blender_dir, version, "scripts", "addons_contrib"),
            
            # Related to Python
            os.path.join(blender_dir, f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, version, "python", "DLLs"),
            os.path.join(blender_dir, version, "python", "lib"),
            os.path.join(blender_dir, version, "python", "bin"),
            os.path.join(blender_dir, version, "python"),
            os.path.join(blender_dir, version, "python", "lib", "site-packages"),
            
            # Backup path (no version)
            os.path.join(blender_dir, "scripts", "startup"),
            os.path.join(blender_dir, "scripts", "modules"),
            os.path.join(blender_dir, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "scripts", "addons"),
            os.path.join(blender_dir, "scripts", "addons_contrib"),
            os.path.join(blender_dir, "python", "lib", "site-packages"),
            os.path.join(blender_dir, "python", "lib"),
            os.path.join(blender_dir, "python")
        ])
        
        # Blender path in the user directory (Windows)
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
        
    elif system == "Darwin":  # macOS
        # macOS: Blender.app/Contents/Resources/{version}/
        lib_paths.extend([
            # Scripts-related
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "startup"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "modules"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons"),
            os.path.join(blender_dir, "..", "Resources", version, "scripts", "addons_contrib"),
            
            # Related to Python
            os.path.join(blender_dir, "..", "Resources", f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "lib", "python3.11", "site-packages"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "lib"),
            os.path.join(blender_dir, "..", "Resources", version, "python", "bin"),
            os.path.join(blender_dir, "..", "Resources", version, "python"),
            
            # Backup path
            os.path.join(blender_dir, "..", "Resources", "scripts", "startup"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "modules"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons", "modules"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons"),
            os.path.join(blender_dir, "..", "Resources", "scripts", "addons_contrib"),
            os.path.join(blender_dir, "..", "Resources", "python", "lib", "python3.11", "site-packages")
        ])
        
        # Blender path in the user directory (macOS)
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
        
    else:  # Linux
        # Linux: blender/{version}/
        lib_paths.extend([
            # Scripts-related
            os.path.join(blender_dir, version, "scripts", "startup"),
            os.path.join(blender_dir, version, "scripts", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons", "modules"),
            os.path.join(blender_dir, version, "scripts", "addons"),
            os.path.join(blender_dir, version, "scripts", "addons_contrib"),
            
            # Related to Python
            os.path.join(blender_dir, f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
            os.path.join(blender_dir, version, "python", "lib", "python3.11", "site-packages"),
            os.path.join(blender_dir, version, "python", "lib"),
            os.path.join(blender_dir, version, "python", "bin"),
            os.path.join(blender_dir, version, "python"),
            
            # Backup path
            os.path.join(blender_dir, "scripts", "startup"),
            os.path.join(blender_dir, "scripts", "modules"),
            os.path.join(blender_dir, "scripts", "addons", "modules"),
            os.path.join(blender_dir, "scripts", "addons"),
            os.path.join(blender_dir, "scripts", "addons_contrib"),
            os.path.join(blender_dir, "python", "lib", "python3.11", "site-packages")
        ])
        
        # Blender path in the user directory (Linux)
        if user_blender_path:
            lib_paths.extend([
                os.path.join(user_blender_path, "scripts", "startup"),
                os.path.join(user_blender_path, "scripts", "modules"),
                os.path.join(user_blender_path, "scripts", "addons", "modules"),
                os.path.join(user_blender_path, "scripts", "addons"),
                os.path.join(user_blender_path, "scripts", "addons_contrib")
            ])
    
    # Add the current Blender site-packages path as well
    for path in site.getsitepackages():
        if path not in lib_paths:
            lib_paths.append(path)
    
    # Dynamically search for individual dependency paths within the add-on directory
    def find_addon_deps_paths():
        addon_deps_paths = []
        addon_dirs = []
        
        # The addons path in the installation directory
        addon_dirs.append(os.path.join(blender_dir, version, "scripts", "addons"))
        
        # The addons directory in the user directory
        if user_blender_path:
            addon_dirs.append(os.path.join(user_blender_path, "scripts", "addons"))
        
        for addon_dir in addon_dirs:
            if os.path.exists(addon_dir):
                try:
                    for addon_name in os.listdir(addon_dir):
                        addon_path = os.path.join(addon_dir, addon_name)
                        if os.path.isdir(addon_path):
                            # Check if the deps directory exists
                            deps_path = os.path.join(addon_path, "deps")
                            if os.path.exists(deps_path):
                                addon_deps_paths.append(deps_path)
                except (OSError, PermissionError):
                    # Skip if you do not have permission
                    continue
        
        return addon_deps_paths
    
    # Add add-on dependency paths
    addon_deps_paths = find_addon_deps_paths()
    lib_paths.extend(addon_deps_paths)
    
    # Remove duplicates and return only the existing paths
    unique_paths = []
    for path in lib_paths:
        if path not in unique_paths and os.path.exists(path):
            unique_paths.append(path)
    
    return unique_paths

def run_rbf_processor(temp_file_path, python_path=None, processor_path=None, old_version=False):
    """
    Run rbf_multithread_processor.py
    
    Parameters:
        temp_file_path (str): Path to the temporary data file
        python_path (str, optional): Path to the Python binary
        processor_path (str, optional): Path to the processor script
        old_version (bool, optional): Whether to output in the old version format
    
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    try:
        np.show_config()
        
        # Get the default path
        if python_path is None:
            python_path = get_blender_python_path()
        
        if processor_path is None:
            processor_path = get_rbf_processor_script_path()
        
        # Verifying the existence of a path
        if not os.path.exists(python_path):
            return False, "", f"Python binary not found: {python_path}"
        
        if not os.path.exists(processor_path):
            return False, "", f"The RBF processor script cannot be found: {processor_path}"
        
        # Get the Python library path in Blender
        blender_lib_paths = get_blender_python_lib_paths()
        
        print(f"Detected Blender library paths:")
        for path in blender_lib_paths:
            print(f"  - {path}")

        # --Get the path of packages installed by the user
        user_site_packages = get_blender_python_user_site_packages(python_path)
        if user_site_packages:
            print(f"Detected user site packages path: {user_site_packages}")
        else:
            print("User site packages path not found")

        blender_deps_path = os.path.join(os.path.dirname(__file__), 'deps')
        print(f"Blender deps path: {blender_deps_path}")
        
        # Method 1: Set the library path using environment variables
        env = os.environ.copy()
        
        # Avoiding Windows-specific character encoding issues
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        
        # Add Blender library paths and user site package paths to PYTHONPATH
        pythonpath_parts = []
        if 'PYTHONPATH' in env:
            pythonpath_parts.append(env['PYTHONPATH'])
        
        # Add the user site package first (set it as a high priority)
        if blender_deps_path:
            pythonpath_parts.append(blender_deps_path)
        if user_site_packages:
            pythonpath_parts.append(user_site_packages)
            
        pythonpath_parts.extend(blender_lib_paths)
        env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
        
        # Disable standard output buffering in Python
        env['PYTHONUNBUFFERED'] = '1'
        
        print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
        
        # Build the command (disable buffering with the -u flag)
        cmd = [python_path, '-u', processor_path, temp_file_path]
        
        # Add an option for the old version format
        if old_version:
            cmd.append('--old-version')
        
        max_workers = os.cpu_count()
        env['OMP_NUM_THREADS'] = str(max_workers)
        env['OPENBLAS_NUM_THREADS'] = str(max_workers)
        env['MKL_NUM_THREADS'] = str(max_workers)
        env['VECLIB_MAXIMUM_THREADS'] = str(max_workers)
        env['NUMEXPR_NUM_THREADS'] = str(max_workers)
        cmd.append('--max-workers')
        cmd.append(str(max_workers))

        print(f"Configured environment variables: {env}")

        print(f"Executing command: {' '.join(cmd)}")

        # Run the process (real-time output)
        print("Starting RBF processing...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect standard error to standard output
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=os.path.dirname(temp_file_path),
                env=env,
                bufsize=1,  # Line buffering
                universal_newlines=True
            )
            
            # Read and display output in real time
            import select
            import sys
            output_lines = []
            
            # Real-time reading in a Windows environment
            if sys.platform == "win32":
                # On Windows, since 'select' cannot be used, a different approach is needed
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line = line.rstrip('\n\r')
                        print(f"[RBF Processing] {line}")
                        output_lines.append(line)
            else:
                # For Unix-like systems, use 'select'
                while True:
                    if process.poll() is not None:
                        # If the process has finished, read the remaining output
                        remaining = process.stdout.read()
                        if remaining:
                            for line in remaining.splitlines():
                                print(f"[RBF Processing] {line}")
                                output_lines.append(line)
                        break
                    
                    # Check if it is readable
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            line = line.rstrip('\n\r')
                            print(f"[RBF Processing] {line}")
                            output_lines.append(line)
            
            # Wait for the process to finish
            process.wait()
            success = process.returncode == 0
            output = '\n'.join(output_lines)
            
            if success:
                print("RBF processing completed successfully")
            else:
                print(f"RBF processing failed (return code: {process.returncode})")

        except Exception as e:
            print(f"Error during process execution: {e}")
            success = False
            output = ""
        
        return success, output, ""
        
    except Exception as e:
        error_msg = f"An error occurred while performing RBF processing.: {str(e)}"
        print(error_msg)
        return False, "", error_msg


# Debugging Operator: Display Python Path Information
class DEBUG_OT_ShowPythonPaths(bpy.types.Operator):
    bl_idname = "rbf.debug_show_python_paths"
    bl_label = "Show Python Paths"
    bl_description = "Display the Python path and library path"
    
    def execute(self, context):
        try:
            # Blender Python binary path
            python_path = get_blender_python_path()
            print(f"\n{'='*60}")
            print(f"PYTHON Path Information")
            print(f"{'='*60}")
            print(f"Python binary path: {python_path}")
            print(f"Exists: {os.path.exists(python_path)}")

            # User Site Package Path
            user_site_packages = get_blender_python_user_site_packages(python_path)
            print(f"\nUser site packages path:")
            if user_site_packages:
                print(f"  {user_site_packages}")
                print(f"  Exists: {'Yes' if os.path.exists(user_site_packages) else 'No'}")
            else:
                print("  Not found")

            # Blender Python library path
            lib_paths = get_blender_python_lib_paths()
            print(f"\nBlender library paths:")
            for i, path in enumerate(lib_paths, 1):
                print(f"  {i}. {path}")
                print(f"     Exists: {'Yes' if os.path.exists(path) else 'No'}")

            # RBF Processor Script Path
            processor_path = get_rbf_processor_script_path()
            print(f"\nRBF processor script path: {processor_path}")
            print(f"Exists: {os.path.exists(processor_path)}")

            # Current PYTHONPATH
            current_pythonpath = os.environ.get('PYTHONPATH', 'Not set')
            print(f"\nCurrent PYTHONPATH: {current_pythonpath}")

            # Checking scipys within Blender
            try:
                import scipy
                print(f"\nSciPy in Blender: Available (version: {scipy.__version__})")
                print(f"SciPy path: {scipy.__file__}")
            except ImportError as e:
                print(f"\nSciPy in Blender: Not available ({e})")
            
            print(f"{'='*60}")
            
            self.report({'INFO'}, "Debug info printed to console")
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"Failed to retrieve debug information: {str(e)}"
            print(error_msg)
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Field Visualization Operator
class CREATE_OT_FieldVisualization(bpy.types.Operator):
    bl_idname = "rbf.create_field_visualization"
    bl_label = "Create Field Visualization"
    bl_description = "Visualize fields as Blender objects from existing deformation data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Switch to Object Mode
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = context.scene
        source_avatar_name = scene.rbf_source_avatar_name.strip()
        target_avatar_name = scene.rbf_target_avatar_name.strip()
        save_shape_key_mode = scene.rbf_save_shape_key_mode
        source_shape_key_name = scene.rbf_source_shape_key
        field_step = scene.rbf_field_step
        use_inverse = scene.rbf_field_use_inverse
        object_name = scene.rbf_field_object_name.strip()
        
        if not source_avatar_name:
            self.report({'ERROR'}, "Please specify source avatar name")
            return {'CANCELLED'}
        
        if not object_name:
            object_name = "FieldVisualization"
        
        # Generate file paths based on the current settings (backward compatibility supported)
        scene_folder = get_scene_folder()
        inverse_suffix = "_inv" if use_inverse else ""

        if save_shape_key_mode:
            # In ShapeKey deformation mode
            if not source_shape_key_name:
                self.report({'ERROR'}, "Please specify shape key name in shape key mode")
                return {'CANCELLED'}
            display_name = "Shape key deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                source_shape_key_name=source_shape_key_name,
                inverse_suffix=inverse_suffix
            )
        else:
            # In the case of standard avatar transformations
            if not target_avatar_name:
                self.report({'ERROR'}, "Please specify target avatar name")
                return {'CANCELLED'}
            display_name = "Inter-avatar deformation data"
            field_data_path = find_field_data_file(
                scene_folder, source_avatar_name,
                target_avatar_name=target_avatar_name,
                inverse_suffix=inverse_suffix
            )

        if not field_data_path:
            expected_filename = f"deformation_{normalize_avatar_name_for_filename(source_avatar_name)}_{f'shape_{source_shape_key_name}' if save_shape_key_mode else f'to_{normalize_avatar_name_for_filename(target_avatar_name)}'}{inverse_suffix}.npz"
            self.report({'ERROR'}, f"{display_name} file not found: {expected_filename}")
            print(f"Deformation data file not found in: {scene_folder}")
            return {'CANCELLED'}
        
        try:
            # Create a field object
            field_obj = create_field_object_from_data(
                field_data_path=field_data_path,
                target_step=field_step,
                object_name=object_name
            )
            
            direction_text = "inverse" if use_inverse else "normal"
            self.report({'INFO'}, f"Field object '{field_obj.name}' created ({direction_text}, step {field_step})")
            return {'FINISHED'}

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Functions for Reinstalling NumPy and SciPy
def get_numpy_version():
    """
    Get the version of numpy currently installed

    Use 'importlib.metadata' to retrieve the version without loading the module.
    This prevents file locking.
    """
    try:
        from importlib.metadata import version
        return version("numpy")
    except Exception:
        return None

def get_scipy_version():
    """
    Get the version of SciPy currently installed

    Use 'importlib.metadata' to retrieve the version without loading the module.
    This prevents file locking.
    """
    try:
        from importlib.metadata import version
        return version("scipy")
    except Exception:
        return None

def _rmtree_onerror(func, path, exc_info):
    """
    Get the version of SciPy currently installed

    Use 'importlib.metadata' to retrieve the version without loading the module.
    This prevents file locking.
    """
    import stat
    # Remove the read-only attribute and try again
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise exc_info[1]

def safe_rmtree(path: str) -> tuple:
    """
    Safely Delete a Directory

    Args:
        path: The path to the directory to be deleted

    Returns:
        tuple: (success: bool, error_type: str, error_message: str)
            error_type: "LOCK", "ERROR", "" (on success)
    """
    if not os.path.exists(path):
        return True, "", ""

    try:
        shutil.rmtree(path, onerror=_rmtree_onerror)
        return True, "", ""
    except PermissionError as e:
        return False, "LOCK", f"The file is locked. Please restart Blender and try again.: {e}"
    except OSError as e:
        return False, "ERROR", f"Failed to delete the directory: {str(e)}"

def safe_rename(src: str, dst: str) -> tuple:
    """
    Rename a directory safely

    Args:
        src: Original path
        dst: New path

    Returns:
        tuple: (success: bool, error_type: str, error_message: str)
    """
    try:
        if os.path.exists(dst):
            # Delete an existing dst
            success, err_type, err_msg = safe_rmtree(dst)
            if not success:
                return False, err_type, err_msg

        os.rename(src, dst)
        return True, "", ""
    except PermissionError as e:
        return False, "LOCK", f"The file is locked. Please restart Blender and try again.: {e}"
    except OSError as e:
        return False, "ERROR", f"Failed to rename: {str(e)}"

def reinstall_numpy_scipy_multithreaded(python_path, numpy_version, scipy_version):
    """
    Force a reinstallation of NumPy and SciPy using the multithreaded versions

    Safe installation method:
    1. Install to a temporary directory (deps_new)
    2. If successful, rename the existing 'deps' directory to 'deps_old'
    3. Rename 'deps_new' to 'deps'
    4. Delete 'deps_old'

    This ensures that the existing dependencies are preserved even if pip fails.

    Parameters:
        python_path (str): The path to Blender's Python binary (retrieved in advance by the main thread)
        numpy_version (str): The NumPy version
        scipy_version (str or None): The SciPy version (None if not installed)

    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    try:
        # Parameter validation (values dependent on bpy have already been retrieved on the main thread)
        if not numpy_version:
            return False, "", "numpy not found"

        if not python_path or not os.path.exists(python_path):
            return False, "", f"The Python path cannot be found: {python_path}"

        # Create a list of packages to install
        packages = [f"numpy=={numpy_version}"]
        if scipy_version:
            packages.append(f"scipy=={scipy_version}")
        else:
            # If scipy is not installed, install the latest version
            packages.append("scipy")
        # Install psutil as well (for monitoring memory)
        packages.append("psutil")
        # Try Numba in a separate step (optional; execution continues even if it fails)

        addon_dir = os.path.dirname(__file__)
        deps_path = os.path.join(addon_dir, 'deps')
        deps_new_path = os.path.join(addon_dir, 'deps_new')
        deps_old_path = os.path.join(addon_dir, 'deps_old')

        print(f"\n{'='*60}")
        print(f"Dependencies Reinstallation Starting")
        print(f"{'='*60}")
        print(f"NumPy version: {numpy_version}")
        if scipy_version:
            print(f"SciPy version: {scipy_version}")
        else:
            print("SciPy: Not installed (will install new)")
        print("psutil: Latest version (for memory monitoring)")
        print("Numba: Latest version (for JIT optimization)")

        # Clean up the temporary directory (delete leftover files from the previous failure)
        # Note: On Windows, the file system state may be delayed, so
        # a file may actually exist even if os.path.exists() returns False
        # Therefore, always attempt to delete the file without checking for its existence
        for tmp_path in [deps_new_path, deps_old_path]:
            print(f"Cleaning up temporary path: {tmp_path}")
            try:
                # First, try deleting it as a file
                try:
                    os.remove(tmp_path)
                    print(f"  Deleted file")
                    continue
                except IsADirectoryError:
                    # For directories, use 'rmtree'
                    pass
                except FileNotFoundError:
                    # If it doesn't exist, skip it
                    print(f"  Does not exist (skipping)")
                    continue
                except PermissionError:
                    # Since there may be directories, use 'rmtree'
                    pass

                # Attempt to delete as a directory
                try:
                    shutil.rmtree(tmp_path, onerror=_rmtree_onerror)
                    print(f"  Deleted directory")
                except FileNotFoundError:
                    print(f"  Does not exist (skipping)")
                except PermissionError as e:
                    err_msg = f"The file is locked. Please restart Blender and try again.: {e}"
                    print(f"  {err_msg}")
                    return False, "", err_msg
                except OSError as e:
                    err_msg = f"Failed to delete: {e}"
                    print(f"  {err_msg}")
                    return False, "", err_msg

            except Exception as e:
                err_msg = f"Unexpected error: {e}"
                print(f"  {err_msg}")
                return False, "", err_msg

        # Create a temporary directory
        # Note: Since os.makedirs() fails in the Microsoft Store version of Blender,
        # on Windows, use 'cmd /c mkdir' instead
        print(f"Creating temporary directory: {deps_new_path}")

        def create_directory(path: str) -> tuple:
            """
            Create a directory across platforms.
            Windows: To support the sandbox environment of the Blender Store version,
                     prioritize using 'cmd /c mkdir'.
            Linux/macOS: Use ' os.makedirs() '.
            """
            import sys

            # For Windows: Use 'cmd /c mkdir' (for Blender from the Microsoft Store)
            if sys.platform == 'win32':
                try:
                    # Since no output is needed, use 'DEVNULL' (to avoid a 'UnicodeDecodeError')
                    result = subprocess.run(
                        ['cmd', '/c', 'mkdir', path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=False
                    )
                    if result.returncode == 0:
                        return True, "cmd"
                    # Error code 1 is returned even if it already exists
                    if os.path.isdir(path):
                        return True, "cmd (already exists)"
                except Exception as e:
                    print(f"  cmd /c mkdir exception: {e}")

            # os.makedirs (Linux/macOS, or as a fallback if 'cmd' fails on Windows)
            try:
                os.makedirs(path, exist_ok=True)
                return True, "os.makedirs"
            except OSError as e:
                print(f"  os.makedirs failed: {e}")

            return False, ""

        success, method = create_directory(deps_new_path)
        if success:
            print(f"Created temporary directory ({method})")
        else:
            return False, "", f"Failed to create temporary directory: {deps_new_path}"

        # pip download + Manual extraction method
        # Since 'pip install --target' in the Microsoft Store version of Blender
        # causes a cross-drive access error (WinError 17),
        # download the wheel file and extract it manually
        import zipfile

        wheels_path = os.path.join(deps_new_path, '_wheels')
        success, method = create_directory(wheels_path)
        if not success:
            return False, "", f"Failed to create wheel download directory: {wheels_path}"
        print(f"Created wheel download directory: {wheels_path}")

        # Step 1: Download the wheel file using 'pip download'
        cmd = [python_path, "-m", "pip", "download",
               "--no-cache-dir",
               "--only-binary=:all:",  # Avoid building from source
               "--dest", wheels_path] + packages
        print(f"Executing command: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        env['PIP_NO_CACHE_DIR'] = '1'

        # Since setting 'capture_output=True' in ' subprocess.run() ' can sometimes cause a 'UnicodeDecodeError' on Windows,
        # run it in binary mode and decode the output manually
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout_bytes, stderr_bytes = process.communicate()

        # Decode using ' safe_decode() ' at the module level
        stdout_text = safe_decode(stdout_bytes)
        stderr_text = safe_decode(stderr_bytes)

        class Result:
            returncode = process.returncode
            stdout = stdout_text
            stderr = stderr_text

        result = Result()

        print(f"Execution result (return code: {result.returncode}):")
        print(f"Output:\n{result.stdout}")

        if result.stderr:
            print(f"Error output:\n{result.stderr}")

        if result.returncode != 0:
            print("pip download failed. Existing deps will be kept.")
            safe_rmtree(deps_new_path)
            return False, result.stdout, result.stderr

        # Step 2: Extract the wheel file
        # Since ' zipfile.extractall() ' uses ' os.makedirs() ' internally,
        # it causes WinError 183 in the Microsoft Store version of Blender.
        # Therefore, extract the files one by one and use
        # 'cmd /c mkdir' to create directories.
        print("Extracting wheel files...")
        wheel_files = [f for f in os.listdir(wheels_path) if f.endswith('.whl')]

        if not wheel_files:
            print("Error: No wheel files found")
            safe_rmtree(deps_new_path)
            return False, result.stdout, "No wheel files found"

        # Track existing directories (to avoid creating duplicates)
        created_dirs = set()

        for wheel_file in wheel_files:
            wheel_path = os.path.join(wheels_path, wheel_file)
            print(f"  Extracting: {wheel_file}")
            try:
                with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # Expanded path
                        target_path = os.path.join(deps_new_path, member)

                        # For directory entries
                        if member.endswith('/'):
                            if target_path not in created_dirs:
                                create_directory(target_path)
                                created_dirs.add(target_path)
                            continue

                        # For files: Create the parent directory
                        parent_dir = os.path.dirname(target_path)
                        if parent_dir and parent_dir not in created_dirs:
                            create_directory(parent_dir)
                            created_dirs.add(parent_dir)

                        # Extract the file
                        with zip_ref.open(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())

            except Exception as e:
                print(f"  Extraction error: {e}")
                import traceback
                traceback.print_exc()
                safe_rmtree(deps_new_path)
                return False, result.stdout, f"Wheel extraction error: {e}"

        # Step 3: Delete the "wheels" directory
        print("Cleaning up wheel files...")
        safe_rmtree(wheels_path)

        # Step 4: Install Numba separately (optional)
        # Even if the Numba installation fails, the main package will install successfully
        numba_success = False
        print("\n" + "="*60)
        print("Optional: Attempting Numba installation (JIT optimization)")
        print("="*60)
        try:
            # Recreate the wheels directory for Numba
            success, method = create_directory(wheels_path)
            if success:
                numba_cmd = [python_path, "-m", "pip", "download",
                           "--no-cache-dir",
                           "--only-binary=:all:",
                           "--dest", wheels_path, "numba"]
                print(f"Executing: {' '.join(numba_cmd)}")

                numba_process = subprocess.Popen(
                    numba_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
                numba_stdout, numba_stderr = numba_process.communicate()

                if numba_process.returncode == 0:
                    # Unpack Numba wheel (Skip this step since numpy and scipy are already installed)
                    all_wheels = [f for f in os.listdir(wheels_path) if f.endswith('.whl')]
                    # Filter numpy-*.whl and scipy-*.whl (retain pinned versions)
                    numba_wheels = [f for f in all_wheels
                                   if not f.startswith('numpy-') and not f.startswith('scipy-')]
                    skipped = [f for f in all_wheels if f not in numba_wheels]
                    if skipped:
                        print(f"  Skipping (already installed): {', '.join(skipped)}")
                    for wheel_file in numba_wheels:
                        wheel_path_full = os.path.join(wheels_path, wheel_file)
                        print(f"  Extracting: {wheel_file}")
                        try:
                            with zipfile.ZipFile(wheel_path_full, 'r') as zip_ref:
                                for member in zip_ref.namelist():
                                    target_path = os.path.join(deps_new_path, member)
                                    if member.endswith('/'):
                                        if target_path not in created_dirs:
                                            create_directory(target_path)
                                            created_dirs.add(target_path)
                                        continue
                                    parent_dir = os.path.dirname(target_path)
                                    if parent_dir and parent_dir not in created_dirs:
                                        create_directory(parent_dir)
                                        created_dirs.add(parent_dir)
                                    with zip_ref.open(member) as source:
                                        with open(target_path, 'wb') as target:
                                            target.write(source.read())
                        except Exception as e:
                            print(f"  Warning: Failed to extract {wheel_file}: {e}")
                    numba_success = True
                    print("Numba installation successful")
                else:
                    numba_error = safe_decode(numba_stderr)
                    print(f"Numba download failed (optional, continuing without it): {numba_error}")

                # Delete the Numba wheels directory
                safe_rmtree(wheels_path)
        except Exception as e:
            print(f"Numba installation skipped due to error (optional): {e}")
            safe_rmtree(wheels_path)

        if not numba_success:
            print("Note: Numba not installed. RBF processing will use SciPy fallback.")
        print("="*60 + "\n")

        # pip successful: directory overwritten
        print("Installation successful. Replacing directory...")

        # If there are existing 'deps' files, rename them to 'deps_old'
        if os.path.exists(deps_path):
            print(f"Moving existing deps to deps_old...")
            success, err_type, err_msg = safe_rename(deps_path, deps_old_path)
            if not success:
                print(f"Failed to rename deps: {err_msg}")
                # Keep the new 'deps_new' file even if it fails (for manual recovery)
                return False, result.stdout, err_msg

        # Rename deps_new to deps
        print(f"Moving deps_new to deps...")
        success, err_type, err_msg = safe_rename(deps_new_path, deps_path)
        if not success:
            print(f"Failed to rename deps_new: {err_msg}")
            # Change "deps_old" back to "deps"
            if os.path.exists(deps_old_path):
                safe_rename(deps_old_path, deps_path)
            return False, result.stdout, err_msg

        # Remove #deps_old (only a warning if it fails)
        if os.path.exists(deps_old_path):
            print(f"Deleting old deps_old...")
            success, _, err_msg = safe_rmtree(deps_old_path)
            if not success:
                print(f"Warning: Failed to delete deps_old (please delete manually): {err_msg}")

        print("Directory replacement complete")
        return True, result.stdout, result.stderr

    except Exception as e:
        error_msg = f"Error occurred during dependencies reinstallation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return False, "", error_msg


# Numpy and SciPy Reinstallation Guide (Modal Version - Avoiding UI Freezes)
class REINSTALL_OT_NumpyScipyMultithreaded(bpy.types.Operator):
    bl_idname = "rbf.reinstall_numpy_scipy_multithreaded"
    bl_label = "Reinstall Dependencies"
    bl_description = "Reinstall NumPy, SciPy, psutil, Numba"

    # Preserve the state of the installation thread
    _timer = None
    _thread = None
    _result = None  # (success, output, error)
    _numpy_version = None
    _scipy_version = None
    _dot_count = 0  # Animation Counter

    def modal(self, context, event):
        if event.type == 'TIMER':
            # Updating the status bar animation
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * (self._dot_count + 1)
            context.workspace.status_text_set(f"Installing dependencies{dots}")

            # Check if the thread has finished
            if self._thread is not None and not self._thread.is_alive():
                # Stop the timer
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None

                # Clear the status bar
                context.workspace.status_text_set(None)

                # Get the results
                success, output, error = self._result if self._result else (False, "", "Unknown error")

                if success:
                    packages_info = f"NumPy {self._numpy_version}"
                    if self._scipy_version:
                        packages_info += f", SciPy {self._scipy_version}"
                    else:
                        packages_info += ", SciPy (new installation)"

                    self.report({'WARNING'}, f"{packages_info} reinstalled. Please restart Blender.")
                    print(f"Dependencies reinstall succeeded. Please restart Blender.")

                    # Display a success pop-up
                    def draw_success_popup(self, context):
                        self.layout.label(text="Dependencies installation complete.")
                        self.layout.label(text="")
                        self.layout.label(text="Please restart Blender.", icon='ERROR')

                    context.window_manager.popup_menu(draw_success_popup, title="Installation Complete", icon='CHECKMARK')
                else:
                    if error:
                        self.report({'ERROR'}, error)
                    else:
                        self.report({'ERROR'}, "Dependencies reinstallation failed")

                    # Display an error pop-up
                    def draw_error_popup(self, context):
                        self.layout.label(text="Installation failed.")
                        self.layout.label(text="")
                        if error:
                            # Display a shorter error message
                            short_error = error[:80] + "..." if len(error) > 80 else error
                            self.layout.label(text=short_error)
                        self.layout.label(text="See console for details.", icon='INFO')

                    context.window_manager.popup_menu(draw_error_popup, title="Installation Error", icon='ERROR')

                # Update the UI
                for area in context.screen.areas:
                    area.tag_redraw()

                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        import threading

        # Get the current version (on the main thread)
        self._numpy_version = get_numpy_version()
        self._scipy_version = get_scipy_version()

        if not self._numpy_version:
            self.report({'ERROR'}, "numpy not found")
            return {'CANCELLED'}

        # Pre-fetch values dependent on bpy on the main thread (for thread safety)
        python_path = get_blender_python_path()
        numpy_version = self._numpy_version
        scipy_version = self._scipy_version

        if not python_path:
            self.report({'ERROR'}, "Python path not found")
            return {'CANCELLED'}

        # Run the installation in a separate thread (passing only pure Python data)
        def run_install():
            try:
                self._result = reinstall_numpy_scipy_multithreaded(
                    python_path, numpy_version, scipy_version
                )
            except Exception as e:
                self._result = (False, "", str(e))

        self._thread = threading.Thread(target=run_install)
        self._thread.start()

        # Set the timer (check every 0.5 seconds)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)

        # Start displaying in the status bar
        self._dot_count = 0
        context.workspace.status_text_set("Installing dependencies.")

        self.report({'INFO'}, "Installing... (running in background)")

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # Display a confirmation dialog before execution
        numpy_version = get_numpy_version()
        scipy_version = get_scipy_version()

        if numpy_version:
            return context.window_manager.invoke_confirm(self, event)
        else:
            self.report({'ERROR'}, "numpy not found")
            return {'CANCELLED'}

    def cancel(self, context):
        # Clean up the timer upon cancellation
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        # Clear the status bar
        context.workspace.status_text_set(None)


# Debugging operator: Testing SciPy using external Python
class DEBUG_OT_TestExternalPython(bpy.types.Operator):
    bl_idname = "rbf.debug_test_external_python"
    bl_label = "Test External Python"
    bl_description = "Testing the import of SciPy in an external Python script"
    
    def execute(self, context):
        try:
            python_path = get_blender_python_path()
            blender_lib_paths = get_blender_python_lib_paths()
            
            # Create a test script (written in English to avoid encoding issues)
            test_script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Add Blender library paths
blender_lib_paths = {repr(blender_lib_paths)}

print("Python executable path:", sys.executable)
print("Python version:", sys.version)
print()

print("Adding library paths:")
for lib_path in blender_lib_paths:
    exists = "YES" if os.path.exists(lib_path) else "NO"
    print(f"  - {{lib_path}} (exists: {{exists}})")
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

print()
print("Current sys.path:")
for i, path in enumerate(sys.path):
    print(f"  {{i+1}}. {{path}}")

print()
print("scipy import test:")
try:
    import scipy
    print(f"scipy: SUCCESS (version: {{scipy.__version__}})")
    print(f"scipy path: {{scipy.__file__}}")
    
    from scipy.spatial import cKDTree
    print("cKDTree: import SUCCESS")
    
    import numpy as np
    print(f"numpy: SUCCESS (version: {{np.__version__}})")
    
    import mathutils
    print(f"mathutils: SUCCESS (version: {{mathutils.__version__}})")
    
    from mathutils.bvhtree import BVHTree
    print("BVHTree: import SUCCESS")
    
except ImportError as e:
    print(f"import FAILED: {{e}}")

print("\\nTest completed")
'''
            
            # Save the test script to a temporary file
            scene_folder = get_scene_folder()
            test_script_path = os.path.join(scene_folder, "test_scipy_import.py")
            
            with open(test_script_path, 'w', encoding='utf-8') as f:
                f.write(test_script_content)
            
            # Set environment variables
            env = os.environ.copy()
            
            # Avoiding Windows-specific character encoding issues
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
            
            pythonpath_parts = []
            if 'PYTHONPATH' in env:
                pythonpath_parts.append(env['PYTHONPATH'])
            pythonpath_parts.extend(blender_lib_paths)
            env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
            
            # Run the test script
            cmd = [python_path, test_script_path]
            print(f"\n{'='*60}")
            print(f"External Python Test Execution")
            print(f"{'='*60}")
            print(f"Executing command: {' '.join(cmd)}")

            returncode, stdout, stderr = run_subprocess_safe(cmd, env=env, cwd=scene_folder)

            print(f"Execution result (return code: {returncode}):")
            print(f"Output:\n{stdout}")

            if stderr:
                print(f"Error output:\n{stderr}")
            
            # Delete test script
            try:
                os.remove(test_script_path)
            except:
                pass
            
            if returncode == 0:
                self.report({'INFO'}, "External Python test succeeded")
            else:
                self.report({'WARNING'}, "External Python test detected issues")
            
            return {'FINISHED'}
        
        except Exception as e:
            error_msg = f"External Python test failed: {str(e)}"
            print(error_msg)
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}


# Do not automatically register here if loaded as an add-on
# register() and unregister() are called from __init__.py
