# Third-party notices

NavOL is distributed under the BSD 3-Clause License in `LICENSE`. Portions of
this repository are derived from or vendor third-party projects whose original
notices and licenses remain in effect.

## Isaac Lab

The Isaac Lab extension scaffold, task launchers, wrappers, and files carrying
an Isaac Lab copyright header are derived from the
[Isaac Lab project](https://github.com/isaac-sim/IsaacLab), which is licensed
under the BSD 3-Clause License. Copyright notices in individual source files
must be retained.

Isaac Sim and other NVIDIA Omniverse components are external runtime
dependencies and are subject to NVIDIA's own license terms. They are not
redistributed by this repository.

## RSL-RL

`source/rsl_rl/` contains a NavOL-compatible fork of
[RSL-RL](https://github.com/leggedrobotics/rsl_rl).

- License: BSD 3-Clause
- Copyright: ETH Zurich and NVIDIA CORPORATION & AFFILIATES
- Full text: `source/rsl_rl/LICENSE`
- Dependency license texts retained upstream: `source/rsl_rl/licenses/dependencies/`

NavOL-specific modifications do not remove or replace the upstream license.

## torchinterp1d

`source/torchinterp1d/` vendors
[torchinterp1d](https://github.com/aliutkus/torchinterp1d).

- License: BSD 3-Clause
- Copyright: Inria (Antoine Liutkus)
- Full text: `source/torchinterp1d/LICENSE`

## Python and simulator dependencies

The packages installed from PyPI, Conda, Isaac Lab, Habitat-Sim, or other
upstream channels retain their own licenses. Declaring a dependency in NavOL's
package metadata does not redistribute or relicense that dependency.

## Models, robot assets, and datasets

Large files hosted outside Git are not automatically covered by NavOL's code
license. Users must follow the license or dataset-card terms attached to each
checkpoint, the Dingo robot asset, 3D-FRONT-derived training data, and benchmark
source data. If an asset's redistribution status is unclear, do not publish it
until its provenance and license have been confirmed.
