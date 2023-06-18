# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import sys
from pathlib import Path

import pytest
from utils import check_no_eval_output, check_no_search_output


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = str(cur_dir / "whisper")
    os.chdir(example_dir)
    sys.path.append(example_dir)

    # prepare configs
    from prepare_whisper_configs import main as prepare_whisper_configs

    prepare_whisper_configs(["--no_audio_decoder"])

    yield
    os.chdir(cur_dir)
    sys.path.remove(example_dir)


@pytest.mark.parametrize("custom_device_precision", [(False, None, None)])
def test_whisper(custom_device_precision):
    from olive.workflows import run as olive_run

    is_custom, device, precision = custom_device_precision
    config_file = "whisper.json"
    
    if is_custom:
        config_file = f"whisper_{device}_{precision}.json"
    
    olive_config = json.load(open(config_file, "r"))

    # test workflow
    result = olive_run(olive_config)
    if is_custom:
        check_no_search_output(result)
    else:
        check_no_eval_output(result)
        
    # test transcription
    from test_transcription import main as test_transcription

    transcription = test_transcription(["--config", config_file])
    print(transcription)
    assert len(transcription) > 0

