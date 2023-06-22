# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from itertools import chain
import logging
from pathlib import Path
from typing import Any, Dict, Union

import onnx
import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeOnnxModel, ONNXModel, PyTorchModel
from olive.model.hf_utils import get_hf_model_io_config, get_onnx_config
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class EnDeConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            )
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: PyTorchModel, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        
        from optimum.exporters import TasksManager
        load_model = model.load_model()

        onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            model=load_model, exporter="onnx", task="feature-extraction"
        )
        onnx_config = onnx_config_constructor(load_model.config)
        encoder = load_model.get_encoder()
        onnx_models = []
        component_names = []
        onnx_config_encoder = onnx_config.with_behavior("encoder")
        onnx_config_decoder = onnx_config.with_behavior("decoder", use_past=False)
        
        
        en_component_output_path = Path(output_model_path).with_suffix("") / "encoder"
        en_output_model_path = ONNXModel.resolve_path(en_component_output_path)
        self.export_pytorch(encoder, onnx_config_encoder, en_output_model_path, config["target_opset"])
        onnx_model = onnx.load(en_output_model_path)
        onnx_models.append(model_proto_to_olive_model(onnx_model, en_output_model_path, config))
        component_names.append("encoder")
        
        de_component_output_path = Path(output_model_path).with_suffix("") / "decoder"
        de_output_model_path = ONNXModel.resolve_path(de_component_output_path)
        self.export_pytorch(load_model, onnx_config_decoder, de_output_model_path, config["target_opset"])
        onnx_model = onnx.load(de_output_model_path)
        onnx_models.append(model_proto_to_olive_model(onnx_model, de_output_model_path, config))
        component_names.append("decoder")
        
        # from optimum.onnxruntime.trainer_seq2seq import exportt
        
        # exportt(model.load_model(), output_model_path, config["target_opset"])
        
        
        # io_config = get_hf_model_io_config(
        #     "openai/whisper-tiny.en", "automatic-speech-recognition", "default"
        # )
        # model = model.load_model()
        # onnx_models = []
        # component_names = []
        # encoder_model = model.get_encoder()
        # component_output_path = Path(output_model_path).with_suffix("") / "encoder"
        # print(f"dummy_inputs: {encoder_model.dummy_inputs}")
        # torch.onnx.export(
        #     encoder_model,
        #     encoder_model.dummy_inputs,
        #     component_output_path,
        #     export_params=True,
        #     opset_version=config["target_opset"],
        #     input_names=io_config["input_names"],
        #     output_names=io_config["output_names"],
        #     dynamic_axes=io_config["dynamic_axes"],
        # )
        # onnx_model = onnx.load(component_output_path)
        # onnx_models.append(model_proto_to_olive_model(onnx_model, output_model_path, config))
        # component_names.append("encoder")

        # decoder_model = model.get_decoder()
        # component_output_path = Path(output_model_path).with_suffix("") / "decoder"
        # torch.onnx.export(
        #     decoder_model,
        #     decoder_model.dummy_inputs,
        #     component_output_path,
        #     export_params=True,
        #     opset_version=config["target_opset"],
        #     input_names=io_config["input_names"],
        #     output_names=io_config["output_names"],
        #     dynamic_axes=io_config["dynamic_axes"],
        # )
        # onnx_model = onnx.load(component_output_path)
        # onnx_models.append(model_proto_to_olive_model(onnx_model, output_model_path, config))
        # component_names.append("decoder")
        
        return CompositeOnnxModel(onnx_models, component_names, hf_config=model.hf_config)
    

    def export_pytorch(self, model, config, output, opset):
        dummy_inputs = config.generate_dummy_inputs(framework="pt")
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())
        torch.onnx.export(
            model,
            (dummy_inputs,),
            output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dict(chain(inputs.items(), config.outputs.items())),
            do_constant_folding=True,
            opset_version=opset,
        )
        