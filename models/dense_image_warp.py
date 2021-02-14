# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

"""Image warping using per-pixel flow vectors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def dense_image_warp3d(image, flow, indexing='ij', name=None):
    
    with tf.name_scope(name or "dense_image_warp"):
        image = tf.convert_to_tensor(image)
        flow  = tf.convert_to_tensor(flow)

        batch_size, nx, ny, nz, channels = image.shape

        # material coordinates Xi,Xj,Xk
        grid_x, grid_y, grid_z = tf.meshgrid(tf.range(nx), tf.range(ny),  tf.range(nz), indexing=indexing)
        stacked_grid = tf.cast(tf.stack([grid_x, grid_y, grid_z], axis=3), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        # deformed coordinates x(X,t)=X+u(X,t), where X=batched_grid and u(X,t)=flow
        query_points_on_grid   = batched_grid + flow
        query_points_flattened = tf.reshape(query_points_on_grid, [tf.shape(image)[0], nx * ny * nz, 3])

        interpolated = interpolate_bilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [tf.shape(image)[0], nx, ny, nz, channels])
        
        return interpolated

def interpolate_bilinear(grid, query_points, indexing="ijk", name=None):

    with tf.name_scope(name or "interpolate_bilinear"):
        grid_shape  = grid.shape
        query_shape = query_points.shape

        batch_size, height, width, depth, channels = (tf.shape(grid)[0], 
                                                      grid_shape[1], grid_shape[2], grid_shape[3], grid_shape[4])

        shape = [batch_size, height, width, depth, channels]

        # pylint: disable=bad-continuation
        with tf.control_dependencies([
                tf.debugging.assert_equal(
                    query_shape[2],
                    3,
                    message="Query points must be size 3 in dim 3.")
        ]):
            num_queries = query_shape[1]
        # pylint: enable=bad-continuation

        query_type = query_points.dtype
        grid_type = grid.dtype

        # pylint: disable=bad-continuation
        with tf.control_dependencies([
                tf.debugging.assert_greater_equal(
                    height, 2, message="Grid height must be at least 2."),
                tf.debugging.assert_greater_equal(
                    width, 2, message="Grid width must be at least 2."),
                tf.debugging.assert_greater_equal(
                    depth, 2, message="Grid depth must be at least 2."),
        ]):
            alphas = []
            floors = []
            ceils  = []
            index_order = [0, 1, 2] if indexing == "ijk" else [2, 1, 0]
            unstacked_query_points = tf.unstack(query_points, axis=2)
        # pylint: enable=bad-continuation

        for dim in index_order:
                with tf.name_scope("dim-" + str(dim)):
                    queries = unstacked_query_points[dim]

                    size_in_indexing_dimension = shape[dim + 1]

                    # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                    # is still a valid index into the grid.
                    max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                    min_floor = tf.constant(0.0, dtype=query_type)
                    floor = tf.math.minimum(
                        tf.math.maximum(min_floor, tf.math.floor(queries)),
                        max_floor)
                    int_floor = tf.cast(floor, tf.dtypes.int32)
                    floors.append(int_floor)
                    ceil = int_floor + 1
                    ceils.append(ceil)

                    # alpha has the same type as the grid, as we will directly use alpha
                    # when taking linear combinations of pixel values from the image.
                    alpha = tf.cast(queries - floor, grid_type)
                    min_alpha = tf.constant(0.0, dtype=grid_type)
                    max_alpha = tf.constant(1.0, dtype=grid_type)
                    alpha = tf.math.minimum(
                        tf.math.maximum(min_alpha, alpha), max_alpha)

                    # Expand alpha to [b, n, 1] so we can use broadcasting
                    # (since the alpha values don't depend on the channel).
                    alpha = tf.expand_dims(alpha, 2)
                    alphas.append(alpha)

         # pylint: disable=bad-continuation

        flattened_grid = tf.reshape(
            grid, [batch_size * height * width * depth, channels])
        batch_offsets = tf.reshape(
            tf.range(batch_size) * height * width * depth, [batch_size, 1])
        # pylint: enable=bad-continuation




        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, z_coords, name):
            with tf.name_scope("gather-" + name):
                # 2D: y_coords * width + x_coords
                # 3D: y_coords * width * depth + x_coords * depth + z_coords
                linear_coordinates = (batch_offsets + y_coords * width * depth + x_coords * depth + z_coords)
                gathered_values = tf.gather(flattened_grid, linear_coordinates)

            return tf.reshape(gathered_values,  [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        I000 = gather(floors[0], floors[1], floors[2], "top_left")   #00
        I010 = gather(floors[0], ceils[1], floors[2], "top_right")
        I100 = gather(ceils[0], floors[1], floors[2], "bottom_left")
        I110 = gather(ceils[0], ceils[1], floors[2], "bottom_right")

        I001 = gather(floors[0], floors[1], ceils[2], "top_left")
        I011 = gather(floors[0], ceils[1], ceils[2], "top_right")
        I101 = gather(ceils[0], floors[1], ceils[2], "bottom_left")
        I111 = gather(ceils[0], ceils[1], ceils[2], "bottom_right")


        # now, do the actual interpolation
        with tf.name_scope("interpolate"):    
            dx, dy, dz = alphas
            w000 = (1. - dx) * (1. - dy) * (1. - dz)
            w001 = (1. - dx) * (1. - dy) * dz
            w010 = (1. - dx) * dy * (1. - dz)
            w011 = (1. - dx) * dy * dz
            w100 = dx * (1. - dy) * (1. - dz)
            w101 = dx * (1. - dy) * dz
            w110 = dx * dy * (1. - dz)
            w111 = dx * dy * dz
            interp = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                               w100 * I100, w101 * I101, w110 * I110, w111 * I111])
        
    return interp 