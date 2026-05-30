# song-gpt

A character-level GPT trained on Taylor Swift lyrics, built from scratch to understand how transformers work.

## What I built

A mini transformer language model that generates song lyrics character by character. Trained on a dataset of Taylor Swift albums.

## Architecture

- **Tokenizer** — character-level encoder/decoder (vocab size ~202 chars)
- **Token + positional embeddings** — each character gets an embedding, plus a learned position vector
- **Multi-head self-attention** — 4 heads, causal masking so tokens can only attend to past positions
- **Feed-forward layers** — 2-layer MLP with 4x expansion after each attention block
- **Residual connections + layer norm** — pre-norm style, applied before each sublayer
- **Stacked transformer blocks** — 4 blocks deep
- **Dropout** — 0.2 rate for regularization

## What I learned

- How attention works: Q, K, V projections, scaled dot-product, causal masking
- Why residual connections matter for training deep networks
- How cross entropy loss drives next-token prediction
- The full forward pass from raw text → tokens → embeddings → logits → loss