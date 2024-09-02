import os
import argparse
import evaluate
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import whisper_timestamped as whisper
from transformers import pipeline

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""

def get_text(sample):
    for key in ['text', 'sentence', 'normalized_text', 'transcript', 'transcription']:
        if key in sample:
            return sample[key]
    raise ValueError(f"Expected transcript column not found. Got sample keys: {', '.join(sample.keys())}")

def get_text_column_names(column_names):
    return next((col for col in ['text', 'sentence', 'normalized_text', 'transcript', 'transcription'] if col in column_names), None)

whisper_norm = BasicTextNormalizer()

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

def transcribe_pipeline(audio, asr):
    result = asr(audio)
    return result["text"]

def transcribe_whisper(audio, model, language, use_vad, use_hyperparameter_tuning):
    if use_hyperparameter_tuning:
        result = whisper.transcribe(model, audio, language=language, best_of=5, beam_size=5, temperature=0.0, vad=use_vad, hyperparameter_tuning=True)
    else:
        result = whisper.transcribe(model, audio, language=language, vad=use_vad)
    return " ".join([segment['text'] for segment in result['segments']])

def main(args):
    if args.use_pipeline:
        print("Using Transformers Pipeline")
        model = pipeline("automatic-speech-recognition", model=args.model_name, device=args.device)
        model.model.config.forced_decoder_ids = model.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")
        transcribe_fn = lambda audio: transcribe_pipeline(audio, model)
    else:
        print("Using Whisper Timestamped")
        model = whisper.load_model(args.model_name, device=args.device)
        transcribe_fn = lambda audio: transcribe_whisper(audio, model, args.language, args.use_vad, args.use_hyperparameter_tuning)

    dataset = load_dataset(args.dataset, args.config, split=args.split, use_auth_token=True).select(range(10))

    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, num_proc=2)
    dataset = dataset.filter(
        is_target_text_in_range, input_columns=[text_column_name], num_proc=2
    )

    predictions = []
    references = []
    norm_predictions = []
    norm_references = []

    for item in tqdm(dataset, desc="Transcribing"):
        audio = item['audio']['array'] if args.use_pipeline else whisper.load_audio(item['audio']['path'])
        prediction = transcribe_fn(audio)
        reference = get_text(item)
        
        predictions.append(prediction)
        references.append(reference)
        norm_predictions.append(whisper_norm(prediction))
        norm_references.append(item["norm_text"])
        print(norm_predictions[-1])
        print(norm_references[-1])
    wer = wer_metric.compute(references=references, predictions=predictions)
    cer = cer_metric.compute(references=references, predictions=predictions)
    norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
    norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)

    metrics = {
        "WER": wer,
        "CER": cer,
        "NORMALIZED WER": norm_wer,
        "NORMALIZED CER": norm_cer
    }

    for name, value in metrics.items():
        print(f"{name}: {round(100 * value, 2)}")

    os.makedirs(args.output_dir, exist_ok=True)
    dset = f"{args.dataset.replace('/', '_')}_{args.config}_{args.split}"
    method = "pipeline" if args.use_pipeline else "whisper"
    op_folder = f"{args.output_dir}/{dset}_{args.model_name}_{method}_vad" if args.use_vad else f"{args.output_dir}/{dset}_{args.model_name}_{method}"
    op_folder += "_tuned" if args.use_hyperparameter_tuning else ""
    if not os.path.exists(op_folder):
        os.makedirs(op_folder)
    op_file = os.path.join(op_folder, "results.txt")
    print(f"Writing results to {op_file}")
    with open(op_file, "w") as result_file:
        for name, value in metrics.items():
            result_file.write(f"{name}: {round(100 * value, 2)}\n")
        result_file.write("\n\n")
        for ref, hyp in zip(references, predictions):
            result_file.write(f"REF: {ref}\n")
            result_file.write(f"HYP: {hyp}\n")
            result_file.write("-" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pipeline", action="store_true", help="Use Transformers pipeline instead of Whisper timestamped")
    parser.add_argument("--model_name", type=str, required=False, default="openai/whisper-tiny",
                        help="Model name. For pipeline, use full HF model name. For Whisper, use 'tiny', 'base', etc.")
    parser.add_argument("--language", type=str, required=False, default="vi",
                        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi.")
    parser.add_argument("--dataset", type=str, required=False, default="mozilla-foundation/common_voice_11_0",
                        help="Dataset from huggingface to evaluate the model on.")
    parser.add_argument("--config", type=str, required=False, default="vi",
                        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice")
    parser.add_argument("--split", type=str, required=False, default="test",
                        help="Split of the dataset. Eg. 'test'")
    parser.add_argument("--device", type=str, required=False, default="cpu",
                        help="The device to run the model on. 'cpu' or 'cuda'.")
    parser.add_argument("--output_dir", type=str, required=False, default="predictions_dir",
                        help="Output directory for the predictions and hypotheses generated.")
    parser.add_argument("--use_vad", action="store_true", help="Use VAD for Whisper timestamped")
    parser.add_argument("--use_hyperparameter_tuning", action="store_true", help="Use hyperparameter tuning for Whisper timestamped")
    args = parser.parse_args()
    main(args)