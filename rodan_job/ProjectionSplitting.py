from gamera.core import init_gamera, Image, load_image, RGBPixel
from gamera import gamera_xml

import numpy as np
import copy

import sys
import datetime
import os
import glob

init_gamera()


class ProjectionSplitter (object):
    # Given an image of a neume, should return
    # a list of separated neume components

    def __init__(self, **kwargs):

        # projection smoothing
        self.smoothing = kwargs['smoothing']                    # (1, inf) how much convolution to apply
        self.extrema_threshold = kwargs['extrema_threshold']    # ignore extrema points < # pixels from last point

        self.min_glyph_size = kwargs['min_glyph_size']          # min number of x or y pixels for a glyph
        self.max_recursive_cuts = kwargs['max_recursive_cuts']  # max number of sub-cuts

        self.rotation = 45

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

        # get max and min points
        col_extrema, row_extrema = self._get_diagonal_extrema(image)
        slice_x, slice_y = self._get_diagonal_slices(col_extrema, row_extrema)

        images = self._split_image(image, slice_x, slice_y)

        images = self._postprocess_images(images)
        if images:
            print images

        return images

    ###############
    # Projections
    ###############

    def _get_diagonal_extrema(self, image):
        # returns an array of positions representing
        # pixel projection minima and maxima

        col_projs, row_projs = self._get_diagonal_projections(image)

        # print col_projs, row_projs

        col_projs = self._smooth_projection(col_projs)
        row_projs = self._smooth_projection(row_projs)

        col_extrema = self._find_extrema(col_projs)
        row_extrema = self._find_extrema(row_projs)

        smoothed_col_extrema = self._smooth_extrema(col_extrema)
        smoothed_row_extrema = self._smooth_extrema(row_extrema)

        return smoothed_col_extrema, smoothed_row_extrema

    def _smooth_projection(self, projection):
        val = np.convolve(projection, self.smoothing * [1])
        # print val
        return val

    def _find_extrema(self, projection):

        extrema = []
        direction = 'flat'

        # find first and last non_0 value
        not_0 = list(i for i, x in enumerate(projection) if x != 0)
        if not_0:
            start_pos, end_pos = not_0[0], not_0[-1]
        else:   # all 0s, so no extrema
            return extrema

        # get maxima/minima
        for i, pxl in enumerate(projection[start_pos:]):
            pos = i + start_pos

            pxl_prev = projection[pos - 1]
            dif = abs(pxl - pxl_prev)

            if direction is 'flat':

                if pxl > pxl_prev:
                    # found an edge before ascending
                    extrema.append((pos - 1, pxl_prev))
                    direction = 'up'

                elif pxl < pxl_prev:
                    # found an edge before descending
                    extrema.append((pos - 1, pxl_prev))
                    direction = 'up'

            elif direction is 'up':

                if pxl < pxl_prev:
                    # found max
                    extrema.append((pos - 1, pxl_prev))
                    direction = 'down'

            elif direction is 'down':
                if pxl > pxl_prev:
                    # found min
                    extrema.append((pos - 1, pxl_prev))
                    direction = 'up'

                if pos > end_pos:
                    extrema.append((pos, pxl))
                    break

            # BE CAREFUL
            if i == end_pos:
                extrema.append((pos, pxl))

        # print 'extrema\t', extrema
        return extrema

    def _smooth_extrema(self, extrema):
        nudge = 0
        smoothed_extrema = extrema[:1]

        for i, e in enumerate(extrema[1:]):
            # print e, smoothed_extrema, i, nudge
            dif = abs(e[1] - smoothed_extrema[i - nudge][1])
            if not dif < self.extrema_threshold:
                smoothed_extrema.append(e)
            else:
                nudge += 1

        return smoothed_extrema

    ##########
    # Slices
    ##########

    def _get_diagonal_slices(self, col_extrema, row_extrema):
        # find best slice
        col_slices = self._find_slices(col_extrema)
        row_slices = self._find_slices(row_extrema)
        return self._find_best_slice(col_slices, row_slices)

    def _find_slices(self, extrema):
        slices = []

        for i in range(int(len(extrema) / 2.0 + 0.5) - 2):
            pos = 2 * i + 2

            val_prev, val_this, val_next = extrema[pos - 1], extrema[pos], extrema[pos + 1]
            spread = min(val_prev[1], val_next[1]) - val_this[1]

            slices.append((val_this[0], spread))

        return slices

    def _find_best_slice(self, col_slices, row_slices):
        slice_x, slice_y = None, None

        for i, (index, spread) in enumerate(col_slices):
            if not slice_x or spread > slice_x[1]:
                slice_x = (index, spread)

        for i, (index, spread) in enumerate(row_slices):
            if not slice_y or spread > slice_y[1]:
                slice_y = (index, spread)

        return (slice_x, slice_y)

    ####################
    # Image Processing
    ####################

    def _preprocess_image(self, image):
        # image = self._slant_image(image, self.rotation)
        image = self._to_onebit(image)

        return image

    def _split_image(self, image, slice_x, slice_y):

        # print slice_x, slice_y

        # if no slices, don't split image
        if not slice_x and not slice_y:
            splits = [image]
            fix_bb = False

        # if missing slice in one dimension, split other
        elif not slice_x:
            splits = self._split(image, slice_y[0], 'y')
        elif not slice_y:
            splits = self._split(image, slice_x[0], 'x')

        # if both slices, pick the best one
        elif slice_x[1] > slice_y[1]:
            splits = self._split(image, slice_x[0], 'x')
        else:
            splits = self._split(image, slice_y[0], 'y')

        fixed_outputs = []

        # fix bounding_box
        # print '\n'
        for i in range(len(splits)):
            pass

        return splits

    def _split(self, image, pos, dim):
        theta = self.rotation

        # image properties
        cols, rows = image.ncols, image.nrows
        i_ul = image.ul_x, image.ul_y
        cp = self._get_center_of_image(image)           # center point

        # rotated image properties
        r_image = self._slant_image(image, theta)
        r_cols, r_rows = r_image.ncols, r_image.nrows
        rcp = self._get_center_of_image(r_image)        # rotated center point

        # rotated image points
        r_p1 = (pos, r_rows) if dim is 'x' else (0, pos)  # left / bottom
        r_p2 = (pos, 0) if dim is 'x' else (r_cols, pos)  # top / right
        # print r_p1, r_p2

        # # DRAW on rotated image
        # r_image.draw_line(r_p1, r_p2, RGBPixel(255, 255, 255), 2.0)
        # r_image.save_PNG('./output/rotated' + str(datetime.datetime.now().date()) + '.png')

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
        drawn_image = image.image_copy()
        # drawn_image.draw_line(draw_p1, draw_p2, RGBPixel(255, 255, 255), 1)

        image.draw_line(draw_p1, draw_p2, RGBPixel(0, 0, 0), 3)     # cut glyph with white line

        # image.save_PNG('./output/not_drawn' + str(datetime.datetime.now().time()).replace(':', '.') + '.png')
        # drawn_image.save_PNG('./output/drawn' + str(datetime.datetime.now().time()).replace(':', '.') + '.png')

        #

        splits = [x.image_copy() for x in image.cc_analysis()]

        # for s in splits:
        #     s = self._trim_image(s)

        return splits

    def _to_onebit(self, image):
        return image.to_onebit()

    def _slant_image(self, image, theta):
        return image.rotate(theta, None, 1)

    def _trim_image(self, image):
        return image.trim_image(None)

    def _unslant_image(self, image, theta):
        return image.rotate(- theta, None, 1)

    #####################
    # Images Processing
    #####################

    def _postprocess_images(self, images):
        images = self._filter_images(images)

        processed_images = []
        for i, im in enumerate(images):

            # im = self._unslant_image(im, self.rotation)

            im = self._trim_image(im)
            processed_images.append(im)

        return processed_images

    def _filter_images(self, images):
        # removes glyphs deemed too small
        filtered_images = []
        for image in images:
            if image.ncols > self.min_glyph_size or image.nrows > self.min_glyph_size:
                filtered_images.append(image)

        return filtered_images

    ##################
    # Image Analysis
    ##################

    def _get_diagonal_projections(self, image):
        d_image = self._slant_image(image, self.rotation)
        # d_image = self._trim_image(d_image)
        projections = d_image.projection_cols(), d_image.projection_rows()

        # print projections[0], '\n\n', projections[1], '\n'
        return projections

    ########
    # Math
    ########

    def _rotate_point_around_origin(self, (x1, y1), degrees):
        rads = np.radians(degrees)

        x2 = -1 * y1 * np.sin(rads) + x1 * np.cos(rads)
        y2 = y1 * np.cos(rads) + x1 * np.sin(rads)

        return x2, y2

    def _translate_point(self, point, old_origin, new_origin):
        # point relative to 0, 0
        neutral_p = point[0] + old_origin[0], point[1] + old_origin[1]
        # point relative to new_origin
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
    elif '.xml' in in0:
        inXML = in0

    # remove files already there so they dont get stacked up
    filesPNG = glob.glob('./output/*.png')
    filesXML = glob.glob('./output/*.xml')
    for f in filesPNG + filesXML:
        os.remove(f)

    kwargs = {
        'smoothing': 5,
        'extrema_threshold': 0,
        'min_glyph_size': 40,
        'max_recursive_cuts': 100,
    }

    ps = ProjectionSplitter(**kwargs)

    if inImage:

        image = load_image(inImage)

        # run job
        results = ps.recursive_run(image, 0)
        # save all as images
        for g in results:
            g.save_PNG('./output/' + str(datetime.datetime.now().date()) + '_' +
                       str(datetime.datetime.now().time()).replace(':', '.') + '.png')

    elif inXML:
        glyphs = gamera_xml.glyphs_from_xml(inXML)

        output_glyphs = []
        for g in glyphs:
            output_glyphs += ps.recursive_run(g, 0)

        gamera_xml.glyphs_to_xml('./output/output.xml', output_glyphs)

    print('do stuff')
