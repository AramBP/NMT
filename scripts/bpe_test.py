import bpe.tokenizer as tokenizer
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
text = project_root / "data" / "test_bpe"
encode_file = project_root / "data" / "test_bpe_encode"
decode_file = project_root / "scripts" / "test_bpe_decode"
bpe_file = project_root / "scripts" / "test.bpe"

bpe_tokenizer = tokenizer.load_file(text, 10, bpe_file)
tokenizer.encode_file(text, encode_file, bpe_file)
print(tokenizer.decode_file(encode_file, decode_file, bpe_file))