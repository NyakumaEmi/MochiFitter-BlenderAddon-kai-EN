# -*- coding: utf-8 -*-
"""
MochiFitter-Kai-EN - Advanced Avatar Outfit Retargeting System for Blender
"""

bl_info = {
    "name": "MochiFitter-Kai-EN",
    "author": "Community Fork (Original: MochiFitter Development Team)."
    "DeepL Japanese to English Translation by NyakumaEmi.",
    "version": (0, 2, 20, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MochiFitter-Kai-EN",
    "description": "Community-optimized fork of MochiFitter - Avatar Outfit Retargeting System using RBF interpolation."
    "Compatible with Blender 4.0.0 up to the latest 4.5.X LTS." 
    "Creating MochiFitter profiles with Blender 5.X or later with this add-on is NOT recommended, as 5.X+ introduces breaking Python and API changes.",
    "warning": "Unofficial version - This is an unofficial community fork."
    "Machine translated using DeepL from Japanese to English." 
    "Mistranslations can be reported to the github issues page.",
    "doc_url": "https://github.com/NyakumaEmi/MochiFitter-BlenderAddon-kai-EN",
    "tracker_url": "https://github.com/NyakumaEmi/MochiFitter-BlenderAddon-kai-EN/issues",
    "category": "Mesh",
    "support": "COMMUNITY"
}

import bpy
import sys
import os
import importlib

# Add the 'deps' directory to the path (for dependencies such as scipy)
# Note: Defer the import of scipy (to prevent file locking)
libs_path = os.path.join(os.path.dirname(__file__), 'deps')
if libs_path not in sys.path:
    sys.path.append(libs_path)

# Get the add-on directory path
addon_dir = os.path.dirname(__file__)

# Dynamic module import
def reload_modules():
    """Reload the add-on module"""
    import importlib
    
    # If any modules have already been imported, reload them
    if "SaveAndApplyFieldAuto" in locals():
        importlib.reload(SaveAndApplyFieldAuto)
    
    # Import the main module
    from . import SaveAndApplyFieldAuto

def register():
    """Register an add-on"""
    # Reload the main module
    reload_modules()
    
    # Registering the Main Module
    from . import SaveAndApplyFieldAuto
    
    try:
        SaveAndApplyFieldAuto.register()
        print("MochiFitter-Kai-EN - The add-on has been successfully installed.")
    except Exception as e:
        print(f"MochiFitter-Kai-EN - An error occurred while registering the add-on: {e}")
        import traceback
        traceback.print_exc()

def unregister():
    """Uninstall the add-on"""
    from . import SaveAndApplyFieldAuto
    
    try:
        SaveAndApplyFieldAuto.unregister()
        print("MochiFitter-Kai-EN - The add-on has been uninstalled")
    except Exception as e:
        print(f"MochiFitter-Kai-EN - An error occurred while unregistering the add-on: {e}")
        import traceback
        traceback.print_exc()

# Processing when executed directly as a script
if __name__ == "__main__":
    register() 
