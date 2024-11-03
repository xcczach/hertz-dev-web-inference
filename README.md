# hertz-dev

Hertz-dev is the first conversational audio base model.

See our blog post for more details: https://si.inc/hertz-dev/

We recommend starting by using `inference.ipynb` to generate one- or two-channel completions from a prompt.

Then, you can use `inference_client.py` and `inference_server.py` to talk to the model live through your microphone.

All three scripts will automatically download the models to the `./ckpt` directory.