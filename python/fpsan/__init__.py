# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Interactive Python bindings for hip-fpsan."""

from ._fpsan import *  # noqa: F401,F403

for _name, _obj in list(globals().items()):
    if (
        isinstance(_obj, type)
        and getattr(_obj, "__module__", None) == __name__ + "._fpsan"
    ):
        _obj.__module__ = __name__

del _name, _obj
