import xobjects as xo
from cgeom.structures import ApertureModel, CrossSections


def build_aperture_kernels(context):
    source = '#include "cgeom/headers/polygons.h"'

    kernels = {
        "build_profile_polygons": xo.Kernel(
            c_name="build_profile_polygons",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(CrossSections, name="cross_sections"),
            ],
        ),
    }

    context.add_kernels(
        sources=[source],
        kernels=kernels,
    )
