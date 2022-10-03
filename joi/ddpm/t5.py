import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

transformers.logging.set_verbosity_error()

PAD_id = 0
EOS_id = 1
MAX_LENGTH = 256
DEFAULT_T5_NAME = "t5-base"

def create_tokenizer(name=DEFAULT_T5_NAME):
    return T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)


def create_encoder(name=DEFAULT_T5_NAME, pretrained=True):
    if pretrained:
        return T5EncoderModel.from_pretrained(name)
    else:
        return T5EncoderModel(T5Config.from_pretrained(name))
        

def create_mask(txt, pad=PAD_id):
    src_pad = txt != pad
    src_pad_mask = src_pad.unsqueeze(1).expand(-1, txt.shape[1], -1)

    return src_pad_mask


def tokenize(texts, name=DEFAULT_T5_NAME, max_len=MAX_LENGTH):
    tokenizer = create_tokenizer(name)
    encoded = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding='longest', max_length=max_len, truncation=True)
    input_ids = encoded.input_ids

    return input_ids


def encode_text(texts, name=DEFAULT_T5_NAME, max_len=MAX_LENGTH, pad=PAD_id, eos=EOS_id, cuda=False):
    token_ids = tokenize(texts, name, max_len)
    attn_mask = create_mask(token_ids, pad)
    t5 = create_encoder(name=name)
    if cuda:
        t5.cuda()
        token_ids = token_ids.cuda()
        attn_mask = attn_mask.cuda()
    t5.eval()
    with torch.no_grad():
        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
        encoded_text = encoded_text[token_ids==eos]
        
    return encoded_text
