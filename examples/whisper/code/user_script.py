# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import io
import onnx
import torch
from examples.whisper.code.process_data import WhisperPrePipeline
from olive.passes.utils.whisper_prepost import _to_onnx_stft
from past_helper import PastKeyValuesHelper
from transformers import WhisperForConditionalGeneration
from whisper_dataset import WhisperDataset
from whisper_decoder import WhisperDecoder, WhisperDecoderInputs
from whisper_encoder_decoder_init import WhisperEncoderDecoderInit, WhisperEncoderDecoderInitInputs


def get_encoder_decoder_init():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return WhisperEncoderDecoderInit(
        model,
        model,
        None,
        model.config,
        decoder_start_token_id=None,
    )


def get_decoder():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return WhisperDecoder(model, None, model.config)


def get_encdec_io_config():
    model = get_encoder_decoder_init()
    use_decoder_input_ids = True

    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=use_decoder_input_ids,
        device="cpu",
        use_int32_inputs=True,
    )

    out = model(inputs.encoder_input_ids, inputs.decoder_input_ids)
    present = out[2]
    present_names = PastKeyValuesHelper.get_input_names(present, encoder=True)

    output_names = ["logits", "encoder_hidden_states", *present_names]

    input_names = ["encoder_input_ids"]

    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference.
    # We use a workaround here: first use dim_param "1" for sequence_length, and later change to dim_value.
    sequence_length = "1"
    num_heads = str(model.config.encoder_attention_heads)
    hidden_size = str(model.config.d_model)
    head_size = str(model.config.d_model // model.config.encoder_attention_heads)
    dynamic_axes = {
        "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {
            0: "batch_size",
            1: "encode_sequence_length",
            2: hidden_size,
        },
        "logits": {
            0: "batch_size",
            1: sequence_length,
        },
    }

    if use_decoder_input_ids:
        input_names.append("decoder_input_ids")
        dynamic_axes["decoder_input_ids"] = {
            0: "batch_size",
            1: sequence_length,
        }

    for name in present_names:
        if "cross" in name:
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: "encode_sequence_length",
                3: head_size,
            }

        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: sequence_length,
                3: head_size,
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
        "string_to_int_dim_params": [sequence_length, num_heads, hidden_size, head_size],
    }


def get_dec_io_config():
    # Fix past disappearing bug - duplicate first past entry
    # input_list.insert(2, input_list[2])
    model = get_decoder()
    past_names = PastKeyValuesHelper.get_past_names(model.config.decoder_layers, present=False)
    present_names = PastKeyValuesHelper.get_past_names(model.config.decoder_layers, present=True)
    present_self_names = present_names[: 2 * model.config.decoder_layers]

    input_past_names = past_names
    output_present_names = present_self_names
    output_names = ["logits", *output_present_names]

    input_names = ["input_ids"]
    input_names.extend(input_past_names)

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "encode_sequence_length / 2"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    for name in input_past_names:
        dynamic_axes[name] = {
            0: "batch_size",
            2: "past_decode_sequence_length" if "self" in name else "encode_sequence_length",
        }

    for name in output_present_names:
        if "cross" in name:
            dynamic_axes[name] = {0: "batch_size", 2: "encode_sequence_length"}
        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                2: "past_decode_sequence_length + 1",
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


def encoder_decoder_init_dummy_inputs(model):
    model = model.load_model()
    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=True,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def decoder_dummy_inputs(model):
    model = model.load_model()
    inputs = WhisperDecoderInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        past_decode_sequence_length=5,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def whisper_audio_decoder_dataloader(data_dir, batch_size=None):
    return WhisperDataset(data_dir=data_dir, use_audio_decoder=True)


def whisper_no_audio_decoder_dataloader(data_dir, batch_size=None):
    return WhisperDataset(data_dir=data_dir, use_audio_decoder=False)
    
def preprocess(data_filepath, output_filepath):
    import librosa
    import numpy as np
    
    audio_blob, _ = librosa.load(data_filepath)
    audio_blob = np.expand_dims(audio_blob, axis=0)
    audio_pcm = torch.from_numpy(audio_blob)
    
    whisper_processing = WhisperPrePipeline()
    model_args = (audio_pcm,)
    
    with io.BytesIO() as strm:
        torch.onnx.export(
            whisper_processing,
            model_args,
            strm,
            input_names=["audio_pcm"],
            output_names=["log_mel"],
            do_constant_folding=True,
            export_params=True,
            opset_version=17,
            dynamic_axes={
                "audio_pcm": {1: "sample_len"},
            },
        )
        model = onnx.load_from_string(strm.getvalue())
    model = _to_onnx_stft(model)
    onnx.save_model(model, output_filepath)
    return model

def postprocess():
    from transformers import WhisperProcessor
    from onnxruntime_extensions import PyOrtFunction
    from onnxruntime_extensions.cvt import HFTokenizerConverter

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder", cvt=HFTokenizerConverter(processor.tokenizer).bpe_decoder, skip_special_tokens=True, cpu_only=True
    )
    
    return fn_decoder.onnx_model
    
def eval(model, data_dir, batch_size, device, execution_providers):
    from transformers import AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny.en")
    _pipeline = pipeline("automatic-speech-recognition", model=model.load_model(), tokenizer=tokenizer)
    dataloader = whisper_no_audio_decoder_dataloader(data_dir, batch_size)
    input_data, _ = next(iter(dataloader))
    print(f"input_data: {input_data}")
    result = _pipeline(input_data)
    print(f"result: {result}")
    
def t(model, data_dir, batch_size, device, execution_providers):
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperTokenizerFast,
        WhisperFeatureExtractor,
    )
    from datasets import load_dataset
    feature_extractor = WhisperFeatureExtractor()
    audio = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[3]["audio"]["array"]
    input_features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    session = model.prepare_session(
        inference_settings=None,
        device=device,
        execution_providers=execution_providers,
    )
    PREV_TOKEN = 50360 # <|startofprev|>
    prompt_tokens = [PREV_TOKEN, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284, 7062, 465, 21443, 13, 5414, 318, 1770, 13, 2264, 346, 353, 338, 5642, 1342, 3499, 621, 465, 2300, 13, 679, 4952, 514, 326, 379, 428, 43856, 1622, 286, 262, 614, 11, 351, 6786, 290, 32595, 12023, 28236, 878, 514, 11, 985, 2915, 7428, 422, 6600, 290, 663, 2482, 3051, 749, 14704, 284, 262, 2000, 13]

    SOT_TOKEN = 50257 # <|startoftranscript|>
    NO_TIMESTAMPS_TOKEN = 50362 # <|notimestamps|>
    decoder_input_ids = torch.LongTensor([prompt_tokens + [SOT_TOKEN, NO_TIMESTAMPS_TOKEN]])
    print(f"input_features: {input_features}")
    
    input_feed={"input_features": input_features, "decoder_input_ids": decoder_input_ids[0].numpy()}
    res = session.run(input_feed=input_feed, output_names=None)
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny.en", language="english")
    print(tokenizer.decode(res, decode_with_timestamps=False))
    
    
