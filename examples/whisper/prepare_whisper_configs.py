# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from copy import deepcopy
from pathlib import Path
from urllib import request

from transformers import WhisperConfig

SUPPORTED_WORKFLOWS = {
    ("cpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("cpu", "int8"): ["conversion", "onnx_dynamic_quantization", "insert_beam_search", "prepost"],
    ("gpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("gpu", "fp16"): ["conversion", "transformers_optimization", "mixed_precision", "insert_beam_search", "prepost"],
    ("gpu", "int8"): ["conversion", "onnx_dynamic_quantization", "insert_beam_search", "prepost"],
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Prepare config file for Whisper")
    parser.add_argument(
        "--no_audio_decoder",
        action="store_true",
        help="Don't use audio decoder in the model. Default: False",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # load template
    template_json = json.load(open("whisper_template.json", "r"))

    whisper_config = WhisperConfig(template_json["input_model"]["config"]["hf_config"]["model_name"])

    # set dataloader
    template_json["evaluators"]["common_evaluator"]["metrics"][0]["user_config"]["dataloader_func"] = (
        "whisper_audio_decoder_dataloader" if not args.no_audio_decoder else "whisper_no_audio_decoder_dataloader"
    )

    # update model specific values for transformer optimization pass
    template_json["passes"]["transformers_optimization"]["config"]["num_heads"] = whisper_config.encoder_attention_heads
    template_json["passes"]["transformers_optimization"]["config"]["hidden_size"] = whisper_config.d_model

    # download audio test data
    test_audio_path = download_audio_test_data()
    template_json["passes"]["prepost"]["config"]["testdata_filepath"] = str(test_audio_path)

    for device, precision in SUPPORTED_WORKFLOWS:
        workflow = SUPPORTED_WORKFLOWS[(device, precision)]
        config = deepcopy(template_json)

        # set output name
        config["engine"]["output_name"] = f"whisper_{device}_{precision}"
        config["engine"]["packaging_config"]["name"] = f"whisper_{device}_{precision}"

        # set device for system
        config["systems"]["local_system"]["config"]["accelerators"] = [device]

        # add passes
        config["passes"] = {}
        for pass_name in workflow:
            pass_config = deepcopy(template_json["passes"][pass_name])
            if pass_name == "transformers_optimization":
                pass_config["config"]["use_gpu"] = device == "gpu"
            if pass_name == "prepost":
                pass_config["config"]["tool_command_args"]["use_audio_decoder"] = not args.no_audio_decoder
            config["passes"][pass_name] = pass_config

        # dump config
        json.dump(config, open(f"whisper_{device}_{precision}.json", "w"), indent=4)


def download_audio_test_data():
    cur_dir = Path(__file__).parent
    data_dir = cur_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    test_audio_name = "1272-141231-0002.mp3"
    test_audio_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/" + test_audio_name
    )
    test_audio_path = data_dir / test_audio_name
    request.urlretrieve(test_audio_url, test_audio_path)

    return test_audio_path.relative_to(cur_dir)


if __name__ == "__main__":
    main()
