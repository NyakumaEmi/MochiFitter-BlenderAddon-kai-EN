# MochiFitter-BlenderAddon-kai-EN

MochiFitter Blender Add-on Optimization and Refactoring Project.

This is an English Translation of the original Japanese [Mega-Gorilla/MochiFitter-BlenderAddon-kai](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai) repository.

⚠️ **Blender Version Warning: MochiFitter Profiles generated with Blender 5.X and later are likely to encounter bugs due to breaking Python version differences and API changes between Blender 4.X and Blender 5.X.**
- Use Blender 4.0.0 up to [the latest 4.5.X LTS](https://www.blender.org/download/releases/4-5/) for the most stable experience.
- Blender 4.0.2 is the official version recommended by MochiFitter.
- See 'Supported Blender Versions' below for more detailed Blender version links.

## Overview

This repository is a project dedicated to optimizing and refactoring the code for the open-source (GPLv3 license) Blender add-on portion of [MochiFitter](https://yamirin.booth.pm/items/7657840).

MochiFitter is a paid costume retargeting system for VRChat/VRM avatars. It uses RBF (Radial Basis Function) interpolation and deformation fields to transfer mesh deformations between different avatars.

## Installation

The packaged .zip version of the MochiFitter-BlenderAddon-kai-EN is available for download here:

**[MochiFitter-BlenderAddon-kai-EN Releases (Github)](https://github.com/NyakumaEmi/MochiFitter-BlenderAddon-kai-EN/releases)**

The MochiFitter-BlenderAddon-kai-EN is a free add-on. It is compatible with Blender 4.0.0 up to the latest 4.5.X LTS.

## Purpose of This Repository

- Improve code readability
- Optimize performance
- Fix bugs
- Explore and implement feature improvements

## What Is Included

- `MochiFitter-BlenderAddon/` - The Blender add-on itself (GPLv3 license)

## Not Included

- [MochiFitter Unity add-on (2,500 JPY)](https://yamirin.booth.pm/items/7657840) - Not publicly available as it is paid content.
- The paid MochiFitter add-on provides the Blender file templates you will need for creating MochiFitter profiles.

## System Requirements

- Blender 4.0.0 up to the latest 4.5.X (**Blender 5.X or later is NOT recommended.**)
- NumPy
- SciPy (can be installed using the "Reinstall" button within the add-on)
- Numba (can be installed using the "Reinstall" button within the add-on)

## Supported Blender Versions

| Supported Blender Versions                                                      |
|---------------------------------------------------------------------------------|
| [Blender 4.0.0 - 4.0.2](https://download.blender.org/release/Blender4.0/)       |
| [Blender 4.1.0 - 4.1.1](https://download.blender.org/release/Blender4.1/)       |
| [Blender 4.2.0 - 4.2.21+ LTS](https://download.blender.org/release/Blender4.2/) |
| [Blender 4.3.0 - 4.3.2](https://download.blender.org/release/Blender4.3/)       |
| [Blender 4.4.0 - 4.4.3](https://download.blender.org/release/Blender4.4/)       |
| [Blender 4.5.0 - 4.5.10+ LTS](https://download.blender.org/release/Blender4.5/) |

## Supported Platforms

| Platform | Support Status    | Notes                                          |
|----------|-------------------|------------------------------------------------|
| Windows  | ✅ Fully supported | Also supports Blender from the Microsoft Store|
| Linux    | ✅ Supported       | Manual installation of psutil recommended     |
| macOS    | ⚠️ Untested       | May work                                       |

### Notes for Linux Users

In a Linux environment, the `psutil` module used for the memory monitoring feature does not work with the bundled version.
To enable the memory monitoring feature, please install psutil using Blender's built-in Python:

```bash
# Check the path to Blender's built-in Python (example)
/path/to/blender/4.x/python/bin/python3 -m pip install psutil
```

Basic functionality will work even without psutil, but memory monitoring and CPU affinity settings will be disabled.

## Development Environment Setup

To perform local development and E2E testing, please set up your Blender environment using the following steps.

### 1. Installing Blender

```bash
python scripts/setup_blender.py
```

This will install the following:
- Blender 4.0.2 (the official version recommended by MochiFitter)
- scipy, numpy (installed in Blender's built-in Python)

### 2. Copying the robust-weight-transfer Add-on (Required)

To run the E2E tests (`run_retarget.py`), you must manually copy the `robust-weight-transfer` add-on included in the MochiFitter Unity package.

```
Source: <MochiFitter Unity Project>/BlenderTools/blender-4.0.2-windows-x64/4.0/scripts/addons/robust-weight-transfer/
Destination: MochFitter-unity-addon/BlenderTools/blender-4.0.2-windows-x64/4.0/scripts/addons/robust-weight-transfer/
```

> **Note**: This add-on is not included in the GitHub repository. Please copy it from the Unity project after [purchasing MochiFitter (2,500 JPY)](https://yamirin.booth.pm/items/7657840).

### 3. Running E2E Tests

```bash
cd MochFitter-unity-addon/OutfitRetargetingSystem
python run_retarget.py --preset beryl_to_mao
```

## License

This project is released under the GNU General Public License v3.0. For details, please refer to [LICENSE.txt](MochiFitter-BlenderAddon/LICENSE.txt).

## Related Links

If this English translation of [the original 'MochiFitter-Kai'](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai) has been useful to you, consider supporting the original creator, Mega-Gorilla on Booth. The link below is 500 JPY. Before checkout, you can click 'Add a tip to support this creator' if you would like to leave a bigger tip to Mega-Gorilla.
- [MochiFitter-Kai (BOOTH), 500 JPY](https://megagorilla.booth.pm/items/7807826) - Original Japanese Package Distribution Page

The paid MochiFitter add-on provides the Blender file templates you will need for creating MochiFitter profiles.
- [MochiFitter (BOOTH), 2,500 JPY](https://yamirin.booth.pm/items/7657840) - Original Product Page.

## Important Notes

This repository is an unofficial community project. For support or inquiries regarding the original MochiFitter, please [visit the product page](https://yamirin.booth.pm/items/7657840) on BOOTH. There is an official NINE GATES Discord server linked there. 

⚠️ **!! Support is provided only in Japanese in the official NINE GATES / MochiFitter Discord server.**
