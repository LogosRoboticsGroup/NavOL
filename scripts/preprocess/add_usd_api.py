import os
from pxr import Usd, UsdGeom, UsdPhysics

def add_rigid_body_api_to_usd(usd_path, prim_path="/World/Scene"):
    """
    Adds USD RigidBodyAPI to the specified prim in the USD file.
    """
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        # Apply RigidBodyAPI
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # Optionally, set properties if needed
        # rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        # rigid_body_api.CreateRigidBodyEnabledAttr(True)
        stage.Save()
        print(f"Applied RigidBodyAPI to prim '{prim_path}' in {usd_path}")
    else:
        print(f"Prim '{prim_path}' not found in {usd_path}")

if __name__ == "__main__":
    # Example usage for your USD files
    usd_files = [
        "/SSD_DISK/users/weixiaofei/Embodied_AI/new_dog/navol/z_3d_f/usd/DiningRoom.usd",
        "/SSD_DISK/users/guchun/embodied/navigation/Replica-Dataset/data/apartment_0/mesh.usd"
    ]
    for usd_file in usd_files:
        if os.path.exists(usd_file):
            add_rigid_body_api_to_usd(usd_file)
        else:
            print(f"USD file not found: {usd_file}")