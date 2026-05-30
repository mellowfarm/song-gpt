# song-gpt

a character-level GPT trained on Taylor Swift lyrics, built from scratch to understand how transformers work

## what i built

a mini transformer language model that generates song lyrics character by character. Trained on a dataset of Taylor Swift albums

## architecture

- **tokenizer** — character-level encoder/decoder (vocab size ~202 chars)
- **token + positional embeddings** — each character gets an embedding, plus a learned position vector
- **multi-head self-attention** — 4 heads, causal masking so tokens can only attend to past positions
- **feed-forward layers** — 2-layer MLP with 4x expansion after each attention block
- **residual connections + layer norm** — pre-norm style, applied before each sublayer
- **stacked transformer blocks** — 4 blocks deep
- **dropout** — 0.2 rate for regularization

## what i learnt

- how attention works: Q, K, V projections, scaled dot-product, causal masking
- why residual connections matter for training deep networks
- how cross entropy loss drives next-token prediction
- the full forward pass from raw text → tokens → embeddings → logits → loss
