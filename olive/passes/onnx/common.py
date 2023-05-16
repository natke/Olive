# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional, Union

import onnx

from olive.constants import ModelFileFormat
from olive.model import ModelStorageKind, ONNXModel
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


def get_external_data_config():
    return {
        "save_as_external_data": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Serializes tensor data to separate files instead of directly in the ONNX file. Large models (>2GB)"
                " may be forced to save external data regardless of the value of this parameter."
            ),
        ),
        "all_tensors_to_one_file": PassConfigParam(
            type_=bool,
            default_value=True,
            description=(
                "Effective only if save_as_external_data is True. If true, save all tensors to one external file"
                " specified by 'external_data_name'. If false, save each tensor to a file named with the tensor name."
            ),
        ),
        "external_data_name": PassConfigParam(
            type_=str,
            default_value=None,
            description=(
                "Effective only if all_tensors_to_one_file is True and save_as_external_data is True. If not specified,"
                " the external data file will be named with <model_path_name>.data"
            ),
        ),
    }


def model_proto_to_file(
    model: onnx.ModelProto,
    output_path: Union[str, Path],
    save_as_external_data: Optional[bool] = False,
    all_tensors_to_one_file: Optional[bool] = True,
    external_data_name: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Save the ONNX model to the specified path.

    :param model: The ONNX model to save.
    :param output_path: The path to save the ONNX model to.
    :param save_as_external_data: If True, save tensor data to separate files instead of directly in the ONNX file.
        Large models (>2GB) may be forced to save external data regardless of the value of this parameter.
    :param all_tensors_to_one_file: Effective only if save_as_external_data is True. If True, save all tensors to one
        external file specified by 'external_data_name'. If False, save each tensor to a file named with the tensor
        name.
    :param external_data_name: Effective only if all_tensors_to_one_file is True and save_as_external_data is True.
        If not specified, the external data file will be named with <model_path_name>.data
    :param is_mlflow_format: If True, save the model in MLflow format.

    :return: True if the model has external data, False otherwise.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logger.info(f"Deleting existing onnx file: {output_path}")
        output_path.unlink()

    # parent directory of .onnx file
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not save_as_external_data:
        try:
            # save model
            onnx.save_model(model, str(output_path))
        except ValueError as e:
            # there are different types of error message for large model (>2GB) based on onnx version
            # just try to save as external data
            # if it fails, raise the original error
            try:
                logger.debug(f"Model save failed with error: {e}. Trying to save as external data.")
                model_proto_to_file(model, output_path, True, all_tensors_to_one_file, external_data_name)
                logger.warning(
                    "Model is too large to save as a single file but 'save_as_external_data' is False. Saved tensors"
                    " as external data regardless."
                )
                return True
            except Exception:
                raise e
        return False

    # location for external data
    external_data_path = output_dir / (external_data_name if external_data_name else f"{output_path.name}.data")
    location = external_data_path.name if all_tensors_to_one_file else None

    if all_tensors_to_one_file:
        if external_data_path.exists():
            # Delete the external data file. Otherwise, data will be appended to existing file.
            logger.info(f"Deleting existing external data file: {external_data_path}")
            external_data_path.unlink()
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

    # save model
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location,
    )
    return True


def model_proto_to_olive_model(
    model_proto: onnx.ModelProto,
    output_model_path: Union[str, Path],
    external_data_config: dict,
    name: Optional[str] = None,
    model_file_format: Optional[ModelFileFormat] = None,
) -> ONNXModel:
    """
    Save the ONNX model to the specified path and return the ONNXModel.

    :param model_proto: The ONNX model to save.
    :param output_model_path: The path to save the ONNX model to.
    :param external_data_config: The external data configuration. Must be a dictionary with keys
        "save_as_external_data", "all_tensors_to_one_file", and "external_data_name".
    :param name: The name of the model.
    :param model_file_format: Olive model file format.

    :return: The ONNXModel.
    """
    has_external_data = model_proto_to_file(
        model_proto,
        output_model_path,
        save_as_external_data=external_data_config["save_as_external_data"],
        all_tensors_to_one_file=external_data_config["all_tensors_to_one_file"],
        external_data_name=external_data_config["external_data_name"],
    )

    is_mlflow_format = True if model_file_format and "mlflow" in model_file_format.lower() else False

    return ONNXModel(
        model_path=output_model_path,
        name=name,
        model_storage_kind=ModelStorageKind.LocalFile if not has_external_data else ModelStorageKind.LocalFolder,
        model_file_format=ModelFileFormat.ONNX_MLFLOW_MODEL if is_mlflow_format else ModelFileFormat.ONNX,
    )
