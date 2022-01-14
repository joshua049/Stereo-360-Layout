"""
This module provides utilities to handle the various coordinate system transformations:
1. Spherical to/from cartesian
2. 3D room layout to/from pano pixels
3. 3D room floor_plan_layouts to/from 2D top-down merged floor_plan_layouts
"""
import collections
import logging
import math
import sys
from typing import List, Dict, Any

import numpy as np
import torch
from utils import Point2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


class Transformation2D(
    collections.namedtuple("Transformation", "rotation_matrix scale translation")
):
    """
    Class to handle relative translation/rotation/scale of room shape coordinates
    to transform them from local to the global frame of reference.
    """

    @classmethod
    def from_translation_rotation_scale(
        cls, *, position: Point2D, rotation: float, scale: float
    ):
        """
        Create a transformation object from the ZInD merged top-down geometry data
        based on the given 2D translation (position), rotation angle and scale.

        :param position: 2D translation (in the x-y plane)
        :param rotation: Rotation angle in degrees (in the x-y plane)
        :param scale: Scale factor for all the coordinates

        :return: A transformation object that can later be applied on a list of
        coordinates in local frame of reference to move them into the global
        (merged floor map) frame of reference.
        """
        translation = torch.Tensor([position.x, position.y]).reshape(1, 2)
        rotation_angle = np.deg2rad(rotation)

        rotation_matrix = torch.Tensor(
            [
                [np.cos(rotation_angle), np.sin(rotation_angle)],
                [-np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )

        return cls(
            rotation_matrix=rotation_matrix, scale=torch.Tensor([scale]), translation=translation
        )

    @classmethod
    def from_zind_data(cls, zind_transformation: Dict[str, Any]):
        """
        Create a transformation object from the ZInD JSON blob.

        :param zind_transformation: Dict with "translation", "rotation" and "scale" fields.

        :return: A transformation object that can later be applied on a list of
        coordinates in local frame of reference to move them into the global
        (merged floor map) frame of reference.
        """
        return Transformation2D.from_translation_rotation_scale(
            position=Point2D.from_tuple(zind_transformation["translation"]),
            rotation=zind_transformation["rotation"],
            scale=zind_transformation["scale"],
        )

    def to_global(self, coordinates):
        """
        Apply transformation on a list of 2D points to transform them from local to global frame of reference.

        :param coordinates: List of 2D coordinates in local frame of reference.

        :return: The transformed list of 2D coordinates.
        """
        if len(self.rotation_matrix.shape) == 3:
            coordinates = coordinates.bmm(self.rotation_matrix) * self.scale
            coordinates += self.translation

        else:
            coordinates = coordinates.matmul(self.rotation_matrix) * self.scale
            coordinates += self.translation

        return coordinates

    def apply_inverse(self, coordinates):

        if len(self.rotation_matrix.shape) == 3:
            coordinates -= self.translation
            coordinates = coordinates.bmm(torch.transpose(self.rotation_matrix, 1, 2)) / self.scale
        else:
            coordinates -= self.translation
            coordinates = coordinates.matmul(self.rotation_matrix.T) / self.scale

        return coordinates


class TransformationSpherical:
    """
    Class to handle various spherical transformations.
    """

    EPS = np.deg2rad(1)  # Absolute precision when working with radians.

    def __init__(self):
        pass

    @classmethod
    def rotate(cls, input_array: torch.Tensor):
        return input_array.matmul(cls.ROTATION_MATRIX)

    @staticmethod
    def normalize(points_cart: torch.Tensor) -> torch.Tensor:
        """
        Normalize a set of 3D vectors.
        """
        num_points = points_cart.shape[0]
        assert num_points > 0

        num_coords = points_cart.shape[1]
        assert num_coords == 3

        rho = torch.sqrt(torch.sum(torch.square(points_cart), axis=1))
        return points_cart / rho.reshape(num_points, 1)

    @staticmethod
    def cartesian_to_sphere(points_cart: torch.Tensor) -> torch.Tensor:
        """
        Convert cartesian to spherical coordinates.
        """
        output_shape = (points_cart.shape[0], 3)  # type: ignore

        num_points = points_cart.shape[0]
        assert num_points > 0

        num_coords = points_cart.shape[1]
        assert num_coords == 3

        x_arr = points_cart[:, 0]
        y_arr = points_cart[:, 1]
        z_arr = points_cart[:, 2]

        # Azimuth angle is in [-pi, pi].
        # Note the x-axis flip to align the handedness of the pano and room shape coordinate systems.
        theta = torch.atan2(-x_arr, y_arr)

        # Radius can be anything between (0, inf)
        rho = torch.sqrt(torch.sum(torch.square(points_cart), axis=1))
        phi = torch.asin(z_arr / rho)  # Map elevation to [-pi/2, pi/2]
        return torch.stack((theta, phi, rho), dim=-1).reshape(output_shape)

    @classmethod
    def sphere_to_pixel(cls, points_sph: torch.Tensor, width: int) -> torch.Tensor:
        """
        Convert spherical coordinates to pixel coordinates inside a 360 pano image with a given width.
        """
        output_shape = (points_sph.shape[0], 2)  # type: ignore

        num_points = points_sph.shape[0]
        assert num_points > 0

        num_coords = points_sph.shape[1]
        assert num_coords == 2 or num_coords == 3

        height = width / 2
        assert width > 1 and height > 1

        # We only consider the azimuth and elevation angles.
        theta = points_sph[:, 0]
        assert torch.all(torch.greater_equal(theta, -math.pi - cls.EPS)), theta.min()
        assert torch.all(torch.less_equal(theta, math.pi + cls.EPS))

        phi = points_sph[:, 1]
        assert torch.all(torch.greater_equal(phi, -math.pi / 2.0 - cls.EPS))
        assert torch.all(torch.less_equal(phi, math.pi / 2.0 + cls.EPS))

        # Convert the azimuth to x-coordinates in the pano image, where
        # theta = 0 maps to the horizontal center.
        x_arr = theta + math.pi  # Map to [0, 2*pi]
        x_arr /= 2.0 * math.pi  # Map to [0, 1]
        x_arr *= width - 1  # Map to [0, width)

        # Convert the elevation to y-coordinates in the pano image, where
        # phi = 0 maps to the vertical center.
        y_arr = phi + math.pi / 2.0  # Map to [0, pi]
        y_arr /= math.pi  # Map to [0, 1]
        y_arr = 1.0 - y_arr  # Flip so that y goes up.
        y_arr *= height - 1  # Map to [0, height)

        return torch.stack((x_arr, y_arr), dim=-1).reshape(output_shape)

    @classmethod
    def cartesian_to_pixel(cls, points_cart: torch.Tensor, width: int):
        return cls.sphere_to_pixel(cls.cartesian_to_sphere(points_cart), width)


class Transformation3D:
    """
    Class to handle transformation from the 2D top-down floor map coordinates to 3D cartesian coordinates
    """

    def __init__(self, ceiling_height: float, camera_height: float):
        """
        :param ceiling_height: The height of the ceiling
        :param camera_height: The height of the camera
        """
        self._ceiling_height = ceiling_height
        self._camera_height = camera_height

    def to_3d(self, room_vertices: List[Point2D]):
        """
        Transform 2D room vertices to 3D cartesian points.

        :param room_vertices: The top-down 2D projected vertices

        :return: Both the floor as well as the ceiling vertices in 3D cartesian coordinates
        """
        # Extract and format room shape coordinates
        num_vertices = room_vertices.shape[0]
        floor_z = torch.Tensor(np.repeat([-self._camera_height], num_vertices).reshape(
            num_vertices, 1
        ))
        ceiling_z = torch.Tensor(np.repeat(
            [self._ceiling_height - self._camera_height], num_vertices
        ).reshape(num_vertices, 1))

        # Create floor and ceiling coordinates
        floor_coordinates = torch.hstack((room_vertices, floor_z))
        ceiling_coordinates = torch.hstack((room_vertices, ceiling_z))

        return floor_coordinates, ceiling_coordinates
