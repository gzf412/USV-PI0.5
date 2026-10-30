# Simulation assets

`worlds/simple_waterway.sdf` is the daylight scene template with eight buoy
variants and one cylinder template. The static image-only ego boat and two
redundant cylinder instances were removed. `worlds/sensor_systems.sdf` contains
only the three required sensor system declarations: IMU, NavSat, and contact.
The training-world generator adds the dynamic boat and chosen targets.

`models/` contains the CoRa hull, the two actual buoy mesh families, Sydney
shore/building/tree geometry and referenced textures, and the coast water mesh,
shaders, and textures. All eight buoy appearances reuse two mesh families and
are colored in the world template. No Fuel thumbnails, unrelated cached
models, or full VRX checkout are included.

Set `GZ_SIM_RESOURCE_PATH` to this folder's `models/`. Scene mesh/material paths
use `model://` or model-local relative paths and do not depend on a home folder.
The water shader plugin resolves its model-local shader and texture files.

Origins and attribution:

- CoRa hull: copied from the existing VORC CoRa-derived boat; original Apache
  2.0 license retained in `models/small_cora_float/LICENSE` and `licenses/`.
- Water mesh/shaders/textures and VRX code: existing VRX checkout, original
  Apache 2.0 text in `licenses/VRX_APACHE_2_0.txt`; original source copyright
  headers retained in `../sim_runtime/src/`.
- Sydney Regatta and buoy meshes/textures: existing OpenRobotics Gazebo Fuel
  cache, exact cached versions 3, 5, and 5. Original `model.config` author
  attribution is retained in each model folder. Those cached files do not
  contain a license declaration; this package does not relabel them under the
  code license. Confirm upstream asset terms before external redistribution.

`SOURCE_MANIFEST.json` records original copied-file provenance, including old
source paths; these paths are audit metadata only and are never loaded by the
simulator. The legacy workspace source directories have been removed; their
tools, licenses, and source attribution are retained in the
[legacy source archive](/home/gzf/robot_ws/usv_vla_project/reports/history/legacy_tools_20261026.tar.gz).
All required simulation resources remain local in this project.
`PORTABILITY_MANIFEST.json` records final packaged file hashes.
`python3 assets/check_resources.py` validates local XML and COLLADA image
references without launching Gazebo or downloading anything.
