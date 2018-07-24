from rodan.jobs.base import RodanTask

from gamera.core import init_gamera, Image, load_image
from gamera import gamera_xml
import numpy

from ProjectionSplitting import ProjectionSplitter

init_gamera()


class DiagonalNeumeSlicing(RodanTask):
    name = 'Diagonal Neume Slicing'
    author = 'Noah Baxter'
    description = 'A tool for splitting neumes into neume components based on diagonal projection.'
    enabled = True
    category = 'Image Processing'
    interactive = False

    settings = {
        'title': 'Settings',
        'type': 'object',
        'required': ['Smoothing', 'Minimum Glyph Size', 'Maximum Recursive Cuts'],
        'properties': {
            'Smoothing': {
                'type': 'integer',
                'default': 5,
                'minimum': 1,
                'maximum': 20,
                'description': 'How much convolution to apply to to projections. More smoothing results in softer cuts.'
            },
            'Minimum Glyph Size': {
                'type': 'integer',
                'default': 40,
                'minimum': 5,
                'maximum': 1000,
                'description': 'Discard post-splitting glyphs with an x or y dimension less than the Minimum Glyph Size.'
            },
            'Maximum Recursive Cuts': {
                'type': 'integer',
                'default': 10,
                'minimum': 1,
                'maximum': 100,
                'description': 'How many subcuts are allowed on a glyph. Note that this does not equate to the number of cuts, as if no ideal cuts can be found, the image in returned unprocessed.'
            }
        }
    }

    input_port_types = [{
        'name': 'GameraXML - Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    output_port_types = [{
        'name': 'GameraXML - Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):

        glyphs = gamera_xml.glyphs_from_xml(inputs['GameraXML - Connected Components'][0]['resource_path'])

        kwargs = {
            'smoothing': settings['Smoothing'],
            'extrema_threshold': 0,
            'min_glyph_size': settings['Minimum Glyph Size'],
            'max_recursive_cuts': settings['Maximum Recursive Cuts'],
        }

        ps = ProjectionSplitter(**kwargs)

        output_glyphs = []
        for g in glyphs:
            output_glyphs += ps.recursive_run(g, 0)

        outfile_path = outputs['GameraXML - Connected Components'][0]['resource_path']
        with open(outfile_path, "w") as outfile:
            gamera_xml.glyphs_to_xml(outfile, output_glyphs)

        return True
