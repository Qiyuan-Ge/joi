from transformers import T5Tokenizer, T5EncoderModel

PAD_id = 0
EOS_id = 1

def create_tokenizer(name="t5-small"):
    return T5Tokenizer.from_pretrained(name)


def create_encoder(name="t5-small"):
    return T5EncoderModel.from_pretrained(name)


def create_mask(src, pad=0): # src (b, l1)
    src_pad = src != pad
    src_pad_mask = src_pad.unsqueeze(1).expand(-1, src.shape[1], -1)

    return src_pad_mask


def tokenize(texts, name="t5-small", max_len=256, device='cuda', return_attn_mask=False):
    tokenizer = create_tokenizer(name)
    encoded = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding='longest', max_length=max_len, truncation=True)
    input_ids = encoded.input_ids.to(device)
    if return_attn_mask:
        attn_mask = create_mask(encoded.attention_mask.to(device))
        return input_ids, attn_mask
    else:
        return input_ids


    