import os
import trimesh
import argparse

parser = argparse.ArgumentParser(description="Convert z3d obj to standard obj.")
parser.add_argument("input", type=str, default="z_3d/bedroom.obj", help="Path to the input z3d obj file.")
args = parser.parse_args()
mesh = trimesh.load(args.input, force='mesh')
mesh.vertices = mesh.vertices[:, [2, 0, 1]]
mesh.export(args.input.replace(".obj", "_z_up.obj"))