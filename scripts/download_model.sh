#!/bin/sh
# Download the embedding model BAAI/bge-m3 into models/bge-m3 (about 2.3 GB).
# curl is used instead of the Python client because it trusts the system certificate
# store, which matters behind a corporate TLS proxy.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/bge-m3"
mkdir -p "$DIR/1_Pooling"
for f in 1_Pooling/config.json config.json config_sentence_transformers.json modules.json \
         sentence_bert_config.json sentencepiece.bpe.model special_tokens_map.json \
         tokenizer.json tokenizer_config.json pytorch_model.bin; do
  echo "downloading $f"
  curl -sSL --fail --retry 3 -C - -o "$DIR/$f" "https://huggingface.co/BAAI/bge-m3/resolve/main/$f"
done
