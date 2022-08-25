import tempfile
from pathlib import Path
from typing import Union

import onnx
from onnx.onnx_ml_pb2 import ModelProto
from onnx.shape_inference import infer_shapes
from onnx.shape_inference import infer_shapes_path


def _is_big_model(model: ModelProto) -> bool:
    return model.ByteSize() / (1024 * 1024 * 1024) > 2.0


def _shape_inference_by_model_path(
    model_path: Union[Path, str],
    output_path: Union[Path, str],
    **kwargs,
) -> ModelProto:
    model_path = str(Path(model_path).resolve())
    output_path = str(Path(output_path).resolve())
    infer_shapes_path(model_path, output_path=output_path, **kwargs)

    return onnx.load(output_path)


def safe_shape_inference(  # pylint: disable=missing-function-docstring
    onnx_model_or_path: Union[ModelProto, Path, str],
    **kwargs,
) -> ModelProto:
    if isinstance(onnx_model_or_path, ModelProto):
        if not _is_big_model(onnx_model_or_path):
            return infer_shapes(onnx_model_or_path, **kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / 'model.onnx'
            onnx.save_model(
                proto=onnx_model_or_path,
                f=str(tmp_model_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
            )
            return _shape_inference_by_model_path(tmp_model_path, output_path=tmp_model_path, **kwargs)

    with tempfile.NamedTemporaryFile(dir=Path(onnx_model_or_path).parent) as tmp_model_file:
        return _shape_inference_by_model_path(onnx_model_or_path, output_path=tmp_model_file.name, **kwargs)
