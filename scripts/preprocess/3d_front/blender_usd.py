import bpy
import sys
import os
import argparse

def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def convert_glb_to_usd(input_path, output_path):
    """Convert GLB to USD while preserving materials"""

    print(f"Starting conversion: {input_path}")
    clear_scene()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Import GLB file
        bpy.ops.import_scene.gltf(filepath=input_path)
        print("GLB imported successfully")
    except Exception as e:
        print(f"Error importing GLB: {e}")
        return False

    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    try:
        # Export as USD with materials preserved
        bpy.ops.wm.usd_export(
            filepath=output_path,
            export_materials=True,
            export_textures=True,
            export_normals=True,
        )
        print(f"Successfully converted and saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error exporting USD: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GLB to USD while preserving materials.")
    parser.add_argument("--input", type=str, required=True, help="Input GLB file path")
    parser.add_argument("--output", type=str, required=True, help="Output USD file path")

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = parser.parse_args(argv)

    success = convert_glb_to_usd(args.input, args.output)
    sys.exit(0 if success else 1)
