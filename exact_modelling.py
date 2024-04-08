"""

Make sure you are in the correct directory to run this script, e.g.,:

.. code-block:: python

    import os
    os.chdir("C:/Code/visualization-tool/data/blender")

To run this file type:
.. code-block:: python

    exec(open("create_plane_from_edges.py").read())

"""

import bpy
import mathutils
import numpy as np


def get_rotation_to_vector_as_euler_xyz(
    vector: np.ndarray, *, start_vector: np.ndarray = np.array([0, 0, 1.0])
) -> mathutils.Quaternion:
    if not (input_norm := np.linalg.norm(start_vector)):
        raise ValueError("Zero input vector.")
    start_vector = start_vector / input_norm

    if not (output_norm := np.linalg.norm(vector)):
        raise ValueError("Zero input vector.")
    vector = vector / output_norm

    cross_product_vector = np.cross(start_vector, vector)
    if not (angle := np.linalg.norm(cross_product_vector)):
        return np.zeros_like(vector)
    axis = cross_product_vector / angle

    return mathutils.Quaternion(axis, angle)


def create_plane_from_edges():
    # if True:
    print("Creating plane from edges...")
    obj = bpy.context.active_object

    # Do quick toggle to update. Not sure why...
    for ii in range(2):
        bpy.ops.object.editmode_toggle()

    vertices = [vv.co for vv in bpy.context.active_object.data.vertices if vv.select]
    if len(vertices) != 3:
        raise ValueError(f"Got {len(vertices)} points. Wrong amount to define a plane.")

    center = np.mean(vertices, axis=0)

    vector1 = vertices[1] - vertices[0]
    vector2 = vertices[2] - vertices[0]
    normal = np.cross(vector1, vector2)

    max_dist = max(np.linalg.norm(vector1), np.linalg.norm(vector2))

    quaternion = get_rotation_to_vector_as_euler_xyz(normal)

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.primitive_plane_add(
        enter_editmode=False,
        align="WORLD",
        location=center,
        rotation=quaternion.to_euler(),
        # scale=tuple([max_dist * 1.5] * 3),
    )

    for ii in range(3):
        # Stretching needs to be done afterwards (?!)
        bpy.context.object.scale[ii] = max_dist * 2.0

    # bpy.ops.object.editmode_toggle()
    print(f"New plane of size {bpy.context.object.scale[ii]} created")


# def bisect_object_with_plane():
if True:
    print("Cutting object with plane.")
    obj = bpy.context.active_object

    print("Cut completed created")
    pass


if __name__ == "__main__":
    # create_plane_from_edges()
    # bisect_object_with_plane()
    print("Script finished.")
