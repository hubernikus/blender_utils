"""

Make sure you are in the correct directory to run this script, e.g.,:

.. code-block:: python

    import os
    os.chdir("C:/Code/blender_utils")

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

    vertices = []

    for vv in bpy.context.active_object.data.vertices:
        if not vv.select:
            continue
        vertices.append(obj.matrix_world @ vv.co)

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


def bisect_object_with_plane():
    print("Cutting object with plane...")
    scene_objects = bpy.context.selected_objects
    if len(scene_objects) != 2:
        raise ValueError("Not exactly 2 many objects selected.")

    if len(scene_objects[0].data.vertices) == 4:
        plane = scene_objects[0]
        # obj = scene_objects[1]
    elif len(scene_objects[1].data.vertices) == 4:
        plane = scene_objects[1]
        # obj = scene_objects[0]
    else:
        raise ValueError(
            "Active object does not have 4 vertices as expected of a plane."
        )

    plane.select_set(False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action="SELECT")

    base_normal = mathutils.Vector([0, 0, 1.0])
    plane_normal = plane.matrix_world.to_3x3().normalized() @ base_normal

    bpy.ops.mesh.bisect(
        plane_co=plane.matrix_world.translation,
        plane_no=plane_normal,
        use_fill=True,
        clear_outer=False,
    )

    bpy.ops.object.editmode_toggle()
    print("Cutting succesful.")


if __name__ == "__main__":
    # create_plane_from_edges()
    # bisect_object_with_plane()
    print("Script finished.")
