import pytest

from onnx2torch2.node_converters.registry import add_converter, get_converter

from onnx2torch2.node_converters.nms import OnnxNonMaxSuppression
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node


def _register(override_registry: bool = False) -> None:
    @add_converter(operation_type='NonMaxSuppression', version=10, override_registry=override_registry)
    @add_converter(operation_type='NonMaxSuppression', version=11, override_registry=override_registry)
    def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
        center_point_box = node.attributes.get('center_point_box', 0)
        return OperationConverterResult(
            torch_module=OnnxNonMaxSuppression(center_point_box=center_point_box),
            onnx_mapping=onnx_mapping_from_node(node),
        )

def test_registry_override() -> None:
    # prove that converter is already registered
    assert get_converter("NonMaxSuppression", version=10) is not None
    assert get_converter("NonMaxSuppression", version=11) is not None

    # prove that we cannot override register without passing flag
    with pytest.raises(ValueError, match="already registered"):
        _register()

    # can only override with flag set
    _register(override_registry=True)
