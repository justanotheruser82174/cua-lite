from __future__ import annotations

import importlib
import sys


def test_lite_utils_namespace_does_not_eagerly_reexport_or_import_image():
    sys.modules.pop("lite.utils", None)
    sys.modules.pop("lite.utils.image", None)
    importlib.invalidate_caches()

    utils = importlib.import_module("lite.utils")

    assert not hasattr(utils, "__all__")
    assert not hasattr(utils, "BaseRegistry")
    assert not hasattr(utils, "smart_resize")
    assert "lite.utils.image" not in sys.modules
    assert "pure utilities" in (utils.__doc__ or "")
    assert "documented cross-layer contracts" in (utils.__doc__ or "")
