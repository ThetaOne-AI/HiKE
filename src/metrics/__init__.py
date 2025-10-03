try:
    from measures import pier
except ModuleNotFoundError:
    raise ModuleNotFoundError("submodule pier is not found, please run `git submodule update --init --recursive`")

from .mer import mer


def pier_fixed(ref, pred):
    # If there's no "rest" in reference, PIER always returns 0.
    # To prevent this, add dummy token out of poi to reference
    # This does not affect the calculation of PIER SCORE on poi region.
    _ref = ref + " 뷁"
    _pred = pred + " 뷁"
    # {'poi': {'PIER': 100.0, 'insertions': 1, 'deletions': 0, 'substitutions': 4, 'hits': 1, 'poiWords': 5}, 'rest': {'PIER': 100.0, 'insertions': 1, 'deletions': 0, 'substitutions': 1, 'hits': 1, 'otherWords': 2}}
    pier_poi = pier(_ref, _pred)["poi"]
    pier_score = pier_poi["PIER"]
    return pier_score
