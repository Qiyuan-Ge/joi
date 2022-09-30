import transformers
from transformers import T5Tokenizer, T5EncoderModel

transformers.logging.set_verbosity_error()

PAD_id = 0
EOS_id = 1
MAX_LENGTH = 256
DEFAULT_T5_NAME = "t5-base"

def create_tokenizer(name=DEFAULT_T5_NAME):
    return T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)


def create_encoder(name=DEFAULT_T5_NAME):
    return T5EncoderModel.from_pretrained(name)


def create_mask(txt, pad=PAD_id):
    src_pad = txt != pad
    src_pad_mask = src_pad.unsqueeze(1).expand(-1, txt.shape[1], -1)

    return src_pad_mask


def tokenize(texts, name=DEFAULT_T5_NAME, max_len=MAX_LENGTH, device='cuda', return_attn_mask=False):
    if not torch.cuda.is_available():
        device = 'cpu'
    tokenizer = create_tokenizer(name)
    encoded = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding='longest', max_length=max_len, truncation=True)
    input_ids = encoded.input_ids.to(device)
    if return_attn_mask:
        attn_mask = create_mask(encoded.attention_mask.to(device))
        return input_ids, attn_mask
    else:
        return input_ids


