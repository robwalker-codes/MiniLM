# MiniLM

MiniLM is a tiny, fully managed language model implemented in C# 12 on .NET 8. It demonstrates every stage of a character-level language model pipeline: fetching and cleaning HTML content, normalising and tokenising text, batching and training a compact neural network, and sampling continuations from a saved checkpoint.

## Projects

- **MiniLM.Common** – shared infrastructure covering text ingestion, tokenisation, batching, tensor utilities, optimisers, and checkpoint IO.
- **MiniLM.Train** – CLI and training loop for fitting a multilayer perceptron language model. Supports deterministic seeding, Adam/SGD optimisers, layer normalisation, and checkpoint export.
- **MiniLM.Infer** – CLI for loading checkpoints and generating continuations with configurable temperature, top-*k*, and deterministic sampling.
- **MiniLM.Tests** – xUnit behavioural test suite exercising the tokenizer, data loader, model forward/backward passes, loss, checkpoint IO, sampler logic, and CLI option parsing.

## Model summary

The default model is an embedding-based MLP:

1. Character embeddings are looked up for a fixed context window.
2. Embeddings are flattened and processed by two dense layers with ReLU activations.
3. An optional layer norm stabilises activations before a final projection to the vocabulary.
4. Cross-entropy loss and token-level accuracy are reported during training.

All parameters are tracked in custom `Tensor` objects with manual gradient computation. Optimisers (SGD and Adam) operate on these tensors directly, making backpropagation easy to inspect.

## Preparing data

Provide a text file listing URLs (one per line). The trainer fetches each URL with `HttpClient`, strips scripts/styles via AngleSharp, normalises whitespace and Unicode, and inserts a separator between pages. Alternatively, pass `--corpus` to supply raw text directly (useful for experiments and tests).

Example URL list (`urls.txt`):

```
https://www.example.com
https://www.iana.org/domains/example
```

## Training

```
dotnet run --project src/MiniLM.Train -- \
  --urls urls.txt \
  --output ckpt/minilm-charmlp.json \
  --epochs 3 --batch-size 64 --context-length 128 \
  --lr 0.0003 --seed 42 --embedding-dim 64 \
  --hidden-dim 128 --hidden-dim2 128 --model mlp
```

Useful flags:

- `--corpus` – bypass URL fetching with inline text.
- `--optimiser` – choose `adam` (default) or `sgd`.
- `--disable-layer-norm` – remove the optional layer norm.
- `--no-shuffle` – iterate batches deterministically.
- `--verbose` – enable additional training logs.

Checkpoints serialise weights, shapes, vocabulary, and training metadata to a single JSON file.

## Inference

```
dotnet run --project src/MiniLM.Infer -- \
  --checkpoint ckpt/minilm-charmlp.json \
  --prompt "In the beginning " \
  --temperature 0.9 --top-k 40 --max-tokens 120 --seed 7
```

Set `--temperature 0` to force argmax decoding. `--no-sampler-determinism` switches to stochastic seeding for varied runs.

## Testing

Run the full test suite with:

```
dotnet test
```

Tests cover round-trip tokenisation, batching and padding rules, loss numerics, gradient flow, overfitting a toy corpus, checkpoint integrity, sampler behaviour, and CLI validation.

## Limitations

- The model trains on CPU only and is intended for small corpora.
- Character-level modelling keeps the implementation compact but limits fluency.
- HTML fetching is sequential and does not retry failed requests.
- Only the MLP architecture is implemented; extending `IModel` enables experimentation with attention blocks.

## Reproducibility

All major components accept seeds to ensure deterministic data ordering, parameter initialisation, and sampling. Checkpoints store all tensor weights and shapes, allowing deterministic reloads across processes.
