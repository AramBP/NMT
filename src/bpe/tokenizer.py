from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from os import linesep

TOKEN_FORMAT = "=I"   # native byte order, standard-size unsigned 32-bit
INT64_FORMAT = "=q"   # native byte order, standard-size signed 64-bit
TOKEN_SIZE = struct.calcsize(TOKEN_FORMAT)
INT64_SIZE = struct.calcsize(INT64_FORMAT)

TokenPair = tuple[int, int]
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>"]
BOS = 256
EOS = 257

def pack_token(token: int) -> bytes:
    return struct.pack(TOKEN_FORMAT, token)


def read_token(file) -> int:
    data = file.read(TOKEN_SIZE)
    if len(data) != TOKEN_SIZE:
        raise EOFError("Unexpected end of file while reading token")
    return struct.unpack(TOKEN_FORMAT, data)[0]


def read_int64(file) -> int:
    data = file.read(INT64_SIZE)
    if len(data) != INT64_SIZE:
        raise EOFError("Unexpected end of file while reading Int64")
    return struct.unpack(INT64_FORMAT, data)[0]


def is_byte(token: int) -> bool:
    return token < 256


def is_special(token: int) -> bool:
    return token in (BOS, EOS)


def load_tokens(filename: str | Path) -> list[int]:
    path = Path(filename)
    size = path.stat().st_size

    if size % TOKEN_SIZE != 0:
        raise ValueError("Token file size is not a multiple of 4 bytes")

    with path.open("rb") as file:
        data = file.read()

    return [
        struct.unpack_from(TOKEN_FORMAT, data, offset)[0]
        for offset in range(0, len(data), TOKEN_SIZE)
    ]


def render_tokens(tokens: list[int]) -> None:
    for token in tokens:
        if is_byte(token):
            print(chr(token), end="")
        elif token == BOS:
            print("<|bos|>", end="")
        elif token == EOS:
            print("<|eos|>", end="")
        else:
            print(f"[{token}]", end="")


def preprocess_tokens(tokens: list[int]) -> list[int]:
    new_tokens: list[int] = []

    if tokens:
        new_tokens.append(BOS)

    for i, token in enumerate(tokens):
        new_tokens.append(token)

        # 46 is the ASCII code for "."
        if token == 46 or i == len(tokens) - 1:
            new_tokens.append(EOS)

            if i != len(tokens) - 1:
                new_tokens.append(BOS)

    return new_tokens


class BPE:

  def __init__(self):
    self.pairs : list[TokenPair] = [(i, 0) for i in range(258)]
    self.merge_seq : dict[TokenPair, int] = {} 


  def to_file(self, filename: str) -> None:
    with Path(filename).open("wb") as file:
        file.write(struct.pack(INT64_FORMAT, len(self.pairs)))
        file.write(struct.pack(INT64_FORMAT, len(self.merge_seq)))

        for left, right in self.pairs:
            file.write(pack_token(left))
            file.write(pack_token(right))

        for (left, right), rank in self.merge_seq.items():
            file.write(pack_token(left))
            file.write(pack_token(right))
            file.write(struct.pack(INT64_FORMAT, rank))


  def from_file(self, filename: str | Path) -> None:
      path = Path(filename)

      if path.is_file() or path.stat().st_size == 0:
        pairs: list[TokenPair] = []
        merge_seq: dict[TokenPair, int] = {}

        with path.open("rb") as file:
          n_pairs = read_int64(file)
          n_merge_seq = read_int64(file)

          if n_pairs < 0 or n_merge_seq < 0:
              raise ValueError("Invalid BPE file counts")

          for _ in range(n_pairs):
              pairs.append((read_token(file), read_token(file)))

          for _ in range(n_merge_seq):
              key = (read_token(file), read_token(file))
              value = read_int64(file)
              merge_seq[key] = value

          self.pairs = pairs
          self.merge_seq = merge_seq


  def encode(self, text: str) -> list[int]:
      tokens = preprocess_tokens(list(text.encode("utf-8")))

      while len(tokens) >= 2:
          candidates = [
              (self.merge_seq[(tokens[i], tokens[i + 1])], (tokens[i], tokens[i + 1]))
              for i in range(len(tokens) - 1)
              if (tokens[i], tokens[i + 1]) in self.merge_seq
          ]

          if not candidates:
              break

          best_rank, best_pair = min(candidates, key=lambda item: item[0])

          new_tokens: list[int] = []
          i = 0
          while i < len(tokens):
              if (i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair):
                  new_tokens.append(best_rank)
                  i += 2
              else:
                  new_tokens.append(tokens[i])
                  i += 1

          tokens = new_tokens

      return tokens


  def encode_file(self, input_file: str | Path, output_file : str | Path):
    with Path(output_file).open("wb") as output:
        with Path(input_file).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue

                for token in self.encode(line.rstrip("\n")):
                    output.write(pack_token(token))


  def decode_token(self, token: int) -> str:
      output = bytearray()

      def expand(current: int) -> None:
          if is_special(current):
              if current == EOS:
                output.extend(linesep.encode("utf-8"))
              return
          if is_byte(current):
              output.append(current)
              return

          try:
              left, right = self.pairs[current]
          except IndexError as exc:
              raise ValueError(f"Token {current} has no corresponding pair") from exc

          expand(left)
          expand(right)

      expand(token)
      return output.decode("utf-8", errors="replace")


  def decode(self, tokens: list[int]) -> str:
      return "".join(self.decode_token(token) for token in tokens)


  def decode_file(self, input_file: str | Path, output_file: str | Path) -> None:
      tokens = load_tokens(input_file)
      text = self.decode(tokens)
      Path(output_file).write_text(text, encoding="utf-8")


  def bpe_train(self, tokens: list[int], max_it: int) -> None:
      tokens_in = list(tokens)

      n = 0
      while n < max_it and len(tokens_in) >= 2:
          freqs: dict[TokenPair, int] = {}

          for i in range(len(tokens_in) - 1):
              pair = (tokens_in[i], tokens_in[i + 1])
              freqs[pair] = freqs.get(pair, 0) + 1

          if not freqs:
              break

          max_pair, max_freq = max(freqs.items(), key=lambda item: item[1])

          if max_freq < 2:
              print(f"BPE terminated after {n} iterations.")
              break

          self.pairs.append(max_pair)
          new_token = len(self.pairs) - 1
          self.merge_seq[max_pair] = new_token

          tokens_out: list[int] = []
          i = 0
          while i < len(tokens_in):
              if (i + 1 < len(tokens_in) and (tokens_in[i], tokens_in[i + 1]) == max_pair):
                  tokens_out.append(new_token)
                  i += 2
              else:
                  tokens_out.append(tokens_in[i])
                  i += 1

          tokens_in = tokens_out
          n += 1


def load_file(filename: str | Path, max_it: int) -> BPE:
    tokens: list[int] = []

    with Path(filename).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            tokens.extend(preprocess_tokens(list(line.rstrip("\n").encode("utf-8"))))

    bpe = BPE()
    bpe.bpe_train(tokens, max_it)
    return bpe
