import bpe.tokenizer as tokenizer
from pathlib import Path

text = "../data/test_bpe"
encode_file = "test_bpe_encode"
decode_file = "test_bpe_decode"

bpe = tokenizer.load_file(text, 100)
bpe.encode_file(text, encode_file)
tokens = tokenizer.load_tokens(encode_file)
tokenizer.render_tokens(tokens)
print(bpe.decode(tokens))