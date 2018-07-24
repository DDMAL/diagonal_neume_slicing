from gamera.core import init_gamera, Image, load_image

import numpy

import sys
import datetime
import os
import glob

init_gamera()


class ProjectionSplitter (object):
    # Given an image of a neume, should return
    # a list of separated neume components

    def __init__(self):

        # projection smoothing
        self.smoothing = 5              # (1, inf) how much convolution to apply
        self.extrema_threshold = 0      # ignore extrema points < # pixels from last point

        self.min_glyph_size = 40        # min number of x or y pixels for a glyph
        self.max_recursive_cuts = 10    # max number of sub-cuts

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
                print 'recurse:', i
                new_images += self.recursive_run(im, rec + 1)

            return new_images

        else:
            print 'end recursion'
            return images

    def _run(self, image):
        image = self._preprocess_image(image)

        # get max and min points
        col_extrema, row_extrema = self._get_extrema(image)
        slice_x, slice_y = self._get_slices(col_extrema, row_extrema)

        images = self._split_image(image, slice_x, slice_y)
        images = self._postprocess_images(images)

        return images

    ###############
    # Projections
    ###############

    def _get_extrema(self, image):
        # returns an array of positions representing
        # pixel projection minima and maxima

        col_projs, row_projs = self._get_xy_projections(image)

        col_projs = self._smooth_projection(col_projs)
        row_projs = self._smooth_projection(row_projs)

        col_extrema = self._find_extrema(col_projs)
        row_extrema = self._find_extrema(row_projs)

        smoothed_col_extrema = self._smooth_extrema(col_extrema)
        smoothed_row_extrema = self._smooth_extrema(row_extrema)

        return smoothed_col_extrema, smoothed_row_extrema

    def _smooth_projection(self, projection):
        val = numpy.convolve(projection, self.smoothing * [1])
        # print val
        return val

    def _find_extrema(self, projection):

        extrema = []
        direction = 'flat'

        # find first and last non_0 value
        not_0 = list(i for i, x in enumerate(projection) if x != 0)
        start_pos, end_pos = not_0[0], not_0[-1]

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

        # print 'extrema\t', extrema
        return extrema

    def _smooth_extrema(self, extrema):
        nudge = 0
        smoothed_extrema = [extrema[0]]

        for i, e in enumerate(extrema[1:]):
            dif = abs(e[1] - smoothed_extrema[i - nudge][1])
            if not dif < self.extrema_threshold:
                smoothed_extrema.append(e)
            else:
                nudge += 1

        return smoothed_extrema

    ##########
    # Slices
    ##########

    def _get_slices(self, col_extrema, row_extrema):
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
        image = self._slant_image(image)
        image = self._to_onebit(image)

        return image

    def _split_image(self, image, slice_x, slice_y):
        # if no slices, don't split image
        if not slice_x and not slice_y:
            print 'single\n'
            return [image]

        # if missing slice in one dimension, split other
        elif not slice_x:
            return image.splity([slice_y[0] / float(image.nrows)])
        elif not slice_y:
            return image.splitx([slice_x[0] / float(image.ncols)])

        # if both slices, pick the best one
        elif slice_x[1] > slice_y[1]:
            return image.splitx([slice_x[0] / float(image.ncols)])
        else:
            return image.splity([slice_y[0] / float(image.nrows)])

    def _to_onebit(self, image):
        return image.to_onebit()

    def _slant_image(self, image):
        return image.rotate(45, None, 1)

    def _unslant_image(self, image):
        return image.rotate(-45, None, 1)

    #####################
    # Images Processing
    #####################

    def _postprocess_images(self, images):
        images = self._filter_images(images)

        processed_images = []
        for i, im in enumerate(images):

            im = self._unslant_image(im)
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

    def _get_xy_projections(self, image):
        return image.projection_cols(), image.projection_rows()


if __name__ == "__main__":

    (inImage) = sys.argv[1]
    image = load_image(inImage)

    # remove files already there so they dont get stacked up
    files = glob.glob('./*.png')
    for f in files:
        os.remove(f)

    # init job
    splitter = ProjectionSplitter()
    # run job
    results = splitter.recursive_run(image, 0)

    # save all as images
    for g in results:
        g.save_PNG(str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + '.png')

    print('do stuff')
