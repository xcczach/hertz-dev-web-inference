from ml_web_inference import expose, Request, StreamingResponse, get_proper_device, get_model_size_mb
import torch
import io
import argparse
import torchaudio
from tokenizer import make_tokenizer
from model import get_hertz_dev_config

audio_tokenizer = None
generator = None
device = None

async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]

    
    gen_len = 20 * 8
    audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(device)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
    max_samples = 16000 * 60 * 5
    if audio_tensor.shape[1] > max_samples:
        audio_tensor = audio_tensor[:, :max_samples]
    audio_tensor = audio_tensor.unsqueeze(0)
    prompt_len_seconds = audio_tensor.shape[-1] / 16000
    prompt_len = prompt_len_seconds * 8
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        encoded_prompt_audio = audio_tokenizer.latent_from_data(audio_tensor.to(device))
        encoded_prompt_audio = encoded_prompt_audio[:, :prompt_len]
        completed_audio = generator.completion(
            encoded_prompt_audio, 
            temps=(.8, (0.5, 0.1)), # (token_temp, (categorical_temp, gaussian_temp))
            use_cache=True,
            gen_len=gen_len)
        decoded_completion = audio_tokenizer.data_from_latent(completed_audio.bfloat16())
    audio_tensor = decoded_completion.cpu().squeeze()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.float()

    if audio_tensor.abs().max() > 1:
        audio_tensor = audio_tensor / audio_tensor.abs().max()

    result_arr = audio_tensor[:, max(prompt_len*2000 - 16000, 0):]

    result = io.BytesIO()
    torchaudio.save(result, result_arr, 24000, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global audio_tokenizer, generator, device
    device = 2
    audio_tokenizer = make_tokenizer(device=f"cuda:{device}")
    generator_config = get_hertz_dev_config(is_split=False, use_pure_audio_ablation=False)
    generator = generator_config()
    generator = generator.eval().to(torch.bfloat16).to(f"cuda:{device}")
    print(f"Model size: {get_model_size_mb(generator):.2f} MB")

def hangup():
    global audio_tokenizer, generator
    del audio_tokenizer
    del generator
    torch.cuda.empty_cache()


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="miniomni")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )