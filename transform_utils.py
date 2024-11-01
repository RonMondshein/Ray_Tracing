import numpy as np
from surfaces.surface import Epsilon as Epsilon


def create_translation_matrix(point) -> np.ndarray:
    """""""""""""""""
    create_translation_matrix
    input - plane_point
    output - translation_matrix
    help function to create a translation matrix for plane_mapping_matrix
    """""""""""""""""
    # Create a transformation matrix with only translation
    transformation_matrix = np.eye(4)
    # Set the translation part (last column) to the plane point
    transformation_matrix[:3, 3] = point
    return transformation_matrix


def plane_mapping_matrix(normal, point) -> np.ndarray: 
    """"""""""
    plane_mapping_matrix
    input - plane_normal, plane_point
    output - transformation_matrix
    creates a transformation matrix that maps points from the XY plane (the plane where z=0) to a specified plane in 3D space. The specified plane is defined by a normal vector (normal) and a point on the plane (point).
    """""""""""
    # Define the z-axis unit vector
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    # Check if the plane normal is nearly the same as the z-axis (only needs translation)
    if np.allclose(z_axis, np.abs(normal), atol=Epsilon):
        return create_translation_matrix(point)
    
    # Calculate the rotation matrix using an alternate method
    # Normalize the plane normal
    normal = normal / np.linalg.norm(normal)
    
    # Compute the angle between z_axis and plane_normal
    angle = np.arccos(np.dot(z_axis, normal))

    # Compute the rotation axis (cross product of z_axis and plane_normal)
    rotation_axis = np.cross(z_axis, normal)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # If the rotation axis norm is very small, set the rotation axis to z_axis
    if rotation_axis_norm < Epsilon:
        rotation_axis = z_axis
    else:
        # Normalize the rotation axis
        rotation_axis = rotation_axis / rotation_axis_norm

    # Compute the rotation matrix using the angle and normalized rotation axis
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ], dtype=np.float64)

    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Create a 4x4 identity matrix for the final rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R

    # Create a 4x4 identity matrix for translation
    translation_matrix = np.eye(4)
    # Set the translation part (last column) to the plane point
    translation_matrix[:3, 3] = point

    # Combine the translation and rotation into a single transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)

    return transformation_matrix
