from gamera.core import init_gamera, Image, load_image, RGBPixel
from gamera import gamera_xml

import numpy as np
import copy
import matplotlib.pyplot as plt

from SliceFinding import SliceFinder

import sys
import datetime
import os
import glob

init_gamera()


class ProjectionSplitter (object):
    # Given an image of a neume, should return
    # a list of separated neume components

    def __init__(self, **kwargs):

        self.kwargs = kwargs

        # projection smoothing
        self.smoothing = kwargs['smoothing']                    # (1, inf) how much convolution to apply
        self.extrema_threshold = kwargs['extrema_threshold']    # ignore extrema points < # pixels from last point

        self.min_glyph_size = kwargs['min_glyph_size']          # min number of x or y pixels for a glyph
        self.max_recursive_cuts = kwargs['max_recursive_cuts']  # max number of sub-cuts

        self.min_proj_seg_length = kwargs['min_projection_segments']    # should be relational

        self.low_projection_threshold = kwargs['low_projection_threshold']
        self.min_slice_spread = kwargs['min_slice_spread']

        self.min_slice_spread_rel = kwargs['min_slice_spread_rel']

        self.rotation = kwargs['rotation']

        self.kfill_amount = 3

        self.prefer_multi_cuts = True   # if a projection dimension has more slices than the other, cut that one first
        self.multi_cut_min = 0

        self.cut_number = 0
        self.recursion_number = 0

    ##########
    # Public
    ##########

    def recursive_run(self, image, rec):
        # process image
        images = self._run(image)

        # repeat
        if len(images) > 1 and rec < self.max_recursive_cuts:
            new_images = []
            for i, im in enumerate(images):
                # print 'recurse:', i
                new_images += self.recursive_run(im, rec + 1)

            return new_images

        else:
            # print 'end recursion'
            return images

    def _run(self, image):
        image = self._preprocess_image(image)
        analysis_image = self._preprocess_analysis_image(image)

        sf = SliceFinder(**self.kwargs)
        x_slices, y_slices = sf.get_slices(analysis_image)
        best_slice = self._get_best_slice(x_slices, y_slices)

        # col_arrays, row_arrays = self._get_diagonal_projection_arrays(analysis_image)
        # x_slice, y_slice = self._get_slices(col_arrays, row_arrays)

        images = self._split_image(image, best_slice)

        images = self._postprocess_images(images)
        # if images:
        #     print images

        return images

    ###############
    # Best Slices
    ###############

    def _get_best_slice(self, x_slices, y_slices):
        best_x_slice = self._best_slice(x_slices)
        best_y_slice = self._best_slice(y_slices)

        if self.prefer_multi_cuts:
            if len(x_slices) > len(y_slices) + self.multi_cut_min:
                return (best_x_slice, 'x')

            elif len(y_slices) > len(x_slices) + self.multi_cut_min:
                return (best_y_slice, 'y')

        if best_y_slice is None or best_x_slice[1] > best_y_slice[1]:
            return (best_x_slice, 'x')
        else:
            return (best_y_slice, 'y')

    def _best_slice(self, slices):
        best_slice = None
        for s in slices:
            if not best_slice or s[1] > best_slice[1]:
                best_slice = s

        return best_slice

    ####################
    # Image Processing
    ####################

    def _preprocess_image(self, image):
        image = self._to_onebit(image)

        return image

    def _preprocess_analysis_image(self, image):
        # image = image.to_rgb().simple_sharpen(1.0).to_onebit()
        # image = image.kfill_modified(self.kfill_amount)
        # image = image.convex_hull_as_image(True)
        # image = image.medial_axis_transform_hs()

        return image

    def _split_image(self, image, (best_slice, dim)):

        # if no slice, don't split image
        if not best_slice:
            splits = [image]
            fix_bb = False

        else:
            splits = self._split(image, best_slice[0], dim)

        return splits

    def _split(self, image, pos, dim):
        theta = self.rotation

        # image properties
        cols, rows = image.ncols, image.nrows
        i_ul = image.ul_x, image.ul_y
        cp = self._get_center_of_image(image)           # center point

        # rotated image properties
        r_image = self._rotate_image(image, theta)
        r_cols, r_rows = r_image.ncols, r_image.nrows
        rcp = self._get_center_of_image(r_image)        # rotated center point

        # rotated image points
        r_p1 = (pos, r_rows) if dim is 'x' else (0, pos)  # left / bottom
        r_p2 = (pos, 0) if dim is 'x' else (r_cols, pos)  # top / right
        # print r_p1, r_p2

        # # show rotated cuts
        # r_image.draw_line(r_p1, r_p2, RGBPixel(255, 255, 255), 2.0)
        # r_image.save_PNG('./output/rcut_' + str(datetime.datetime.now().time()).replace(':', '.') + '.png')

        # relate points from ul to center of rotated image
        r_p1 = self._translate_point(r_p1, (0, 0), rcp)
        r_p2 = self._translate_point(r_p2, (0, 0), rcp)
        # print r_p1, r_p2

        # rotate points around origin
        p1 = self._rotate_point_around_origin(r_p1, -theta)
        p2 = self._rotate_point_around_origin(r_p2, -theta)
        # print p1, p2

        # relate points from center of image to ul point
        p1 = self._translate_point(p1, cp, (0, 0))
        p2 = self._translate_point(p2, cp, (0, 0))
        # print p1, p2

        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        # print m

        b1 = p1[1] - (m * p1[0])
        b2 = p2[1] - (m * p2[0])
        # print('b1', b1), ('b2', b2)

        def f_x(x): return m * x + b1

        # DRAW on normal image
        draw_p1, draw_p2 = (0, f_x(0)), (image.ncols, f_x(image.ncols))
        draw_p1 = self._translate_point(draw_p1, i_ul, (0, 0))
        draw_p2 = self._translate_point(draw_p2, i_ul, (0, 0))

        cut_image = image.image_copy()
        cut_image.draw_line(draw_p1, draw_p2, RGBPixel(0, 0, 0), 3)     # cut glyph with white line

        # show cuts
        rgb = image.to_rgb()
        rgb.draw_line(draw_p1, draw_p2, RGBPixel(255, 0, 0), 1)
        rgb.save_PNG('./output/cut_' + str(self.cut_number + 1) + '.png')

        # return [cut_image]

        splits = [x.image_copy() for x in cut_image.cc_analysis()]

        # for s in splits:
        #     s = self._trim_image(s)

        self.cut_number += 1

        return splits

    def _to_onebit(self, image):
        return image.to_onebit()

    def _rotate_image(self, image, theta):
        return image.rotate(theta, None, 1)

    def _trim_image(self, image):
        return image.trim_image(None)

    def _maybe_invert_image(self, image):
        # check the amount of blackness of the image. If it's inverted,
        # the black area will vastly outweigh the white area.
        area = image.area().tolist()[0]
        black_area = image.black_area()[0]

        if area == 0:
            raise AomrError("Cannot divide by a zero area. Something is wrong.")

        # if greater than 60% black, invert the image.
        if (black_area / area) > 0.6:
            image.invert()

        return image

    #####################
    # Images Processing
    #####################

    def _postprocess_images(self, images):
        images = self._filter_tiny_images(images)

        processed_images = []
        for i, im in enumerate(images):

            im = self._trim_image(im)
            processed_images.append(im)

        return processed_images

    def _filter_tiny_images(self, images):
        filtered_images = []    # contains glyphs deemed large enough
        for image in images:
            if image.ncols > self.min_glyph_size or image.nrows > self.min_glyph_size:
                filtered_images.append(image)

        return filtered_images

    ########
    # Math
    ########

    def _rotate_point_around_origin(self, (x1, y1), degrees):
        rads = np.radians(degrees)

        x2 = -1 * y1 * np.sin(rads) + x1 * np.cos(rads)
        y2 = y1 * np.cos(rads) + x1 * np.sin(rads)

        return x2, y2

    def _translate_point(self, point, old_origin, new_origin):
        neutral_p = point[0] + old_origin[0], point[1] + old_origin[1]
        relative_p = neutral_p[0] - new_origin[0], neutral_p[1] - new_origin[1]
        return relative_p

    def _get_center_of_image(self, image):
        l, h = image.ncols, image.nrows
        p = int(0.5 + l / 2.0), int(0.5 + h / 2.0)

        return p


if __name__ == "__main__":
    inImage, inXML = None, None

    (in0) = sys.argv[1]
    if '.png' in in0:
        inImage = in0
        image = load_image(inImage)
    elif '.xml' in in0:
        inXML = in0
        glyphs = gamera_xml.glyphs_from_xml(inXML)

    # remove files already there so they dont get stacked up
    filesPNG = glob.glob('./output/*.png')
    filesXML = glob.glob('./output/*.xml')
    for f in filesPNG + filesXML:
        os.remove(f)

    kwargs = {
        'smoothing': 6,
        'extrema_threshold': 0,
        'min_glyph_size': 20,
        'max_recursive_cuts': 50,
        'rotation': 45,

        'min_projection_segments': 15,       # ++ less likely to cut ligs
        'low_projection_threshold': 0,     # FORCE a cut if valley under a certain value
        'min_slice_spread': 50,             # minimum spread for a cut
        'min_slice_spread_rel': 0.5,

        # Debug Options
        'print_projection_array': True,
        'plot_projection_array': False,  # script only
    }

    ps = ProjectionSplitter(**kwargs)

    if inImage:

        image = image.to_onebit()
        image = ps._maybe_invert_image(image)
        cc_images = image.cc_analysis()
        cc_images = ps._filter_tiny_images(cc_images)

        # output_glyphs = []
        # for g in cc_images:
        #     output_glyphs += ps.recursive_run(g, 0)

        output_glyphs = ps.recursive_run(image, 0)

        # save all as images
<<<<<<< HEAD
        for g in results:
            g.save_PNG('./output/' + str(datetime.datetime.now().date()) + '_' +
                       str(datetime.datetime.now().time()).replace(':', '.') + '.png')
=======
        for i, g in enumerate(output_glyphs):
            g.save_PNG('./output/piece' + str(i + 1) + '.png')
>>>>>>> rotating-line-splitting

    elif inXML:

        output_glyphs = []
        for g in glyphs:
            output_glyphs += ps.recursive_run(g, 0)

        gamera_xml.glyphs_to_xml('./output/output.xml', output_glyphs)

    print('do stuff')
