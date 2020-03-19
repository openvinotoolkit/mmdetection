import importlib

from torch.onnx.symbolic_registry import _symbolic_versions, register_version, _registry
from torch.onnx.symbolic_helper import _onnx_stable_opsets


_extra_onnx_opsets = (11, )


def init_extra_opsets(extra_onnx_opsets=None):
    print(list(_registry.keys()))
    if extra_onnx_opsets is None:
        extra_onnx_opsets = _extra_onnx_opsets
    for opset_version in extra_onnx_opsets:
        module = importlib.import_module('mmdet.utils.export.symbolic_opset{}'.format(opset_version))
        _symbolic_versions[opset_version] = module
        if opset_version not in _onnx_stable_opsets:
            _onnx_stable_opsets.append(opset_version)
        register_version('', opset_version)
    print(list(_registry.keys()))
    # print(_onnx_stable_opsets)
