# My First GPT Model 🧠

A fully custom, **51-million parameter** GPT-style autoregressive language model built from scratch in PyTorch. Trained natively on the 10-billion-token Hugging Face *FineWeb-Edu* dataset!

## 🚀 Features

- **Custom Architecture**: 
  - 6 Transformer Blocks featuring Multi-Head Causal Attention.
  - 5x Feed-Forward Network expansion factors for massive structural "thinking" depth.
  - Native Sinusoidal Positional Embeddings.
  - Smart Weight-tying implemented between the vocabulary embedding matrix and the final LM projection head to save ~38.5 Million parameters!
- **Data Streaming Ecosystem**: 
  - Custom PyTorch `IterableDataset` pipeline directly mapping Hugging Face's `fineweb-edu` stream.
  - Dynamically downloads and slices tokens lazily to drastically eliminate RAM / VRAM overhead.
  - Defeats local SSD storage limits to handle a massive 10B subset without skipping a beat.
- **Hugging Face Hub Native**: 
  - Built-in `push_to_hf.py` script that handles authentication, local token processing (`.env`), write permission tests, and repository pushes securely.
- **Interactive Eval Mode Console**: 
  - Speak with the model interactively using the autoregressive generation pipeline utilizing `torch.multinomial` sampling. 

## 📂 Project Structure

```text
├── src/
│   ├── config.py         # Unified hyperparameter configuration
│   ├── data.py           # Streaming Iterable DataLoader built for huggingface datasets
│   ├── model.py          # The core GPT Architecture (Attention, MLPs, Embs)
│   ├── train.py          # Master Training Loop, saving weights automatically to /checkpoints
│   └── push_to_hf.py     # Secure, dependency-free script to push models to the HF hub
├── checkpoints/          # Where gpt_model_5B_tokens.pt weights are dropped
├── chat.py               # The Interactive GPT chatting console!
├── train.sh              # bash execution script pointing to 'uv run' virtual environments
└── .env                  # Environment Variables securely defining the Write access Token
```

## 🛠️ Usage

### 1. Training the Model
To start training exactly 5 Billion tokens logic (defined in `config.py` as 1,220,703 max steps on an A40 GPU partition):
```bash
./train.sh
```
*Note: We highly recommend using `uv` (e.g. `uv run`) to natively wrap your commands lightning-fast.*

### 2. Chatting with the Model
Once `train.sh` outputs the `.pt` file inside the `checkpoints/` folder, spin up the interactive terminal console! The script handles evaluating sequentially:
```bash
uv run python chat.py
```

### 3. Pushing your custom GPT to the World
Ensure your `.env` file explicitly contains:
```env
HF_TOKEN=your_token_here
```
Then deploy the generated GPT safely to your Hugging Face Account:
```bash
uv run python src/push_to_hf.py
```

## 📈 Curriculum Learning Training (Dynamic Context)

The model features an advanced **Curriculum Learning** pipeline designed to expand the context window to **1000 tokens** safely without running out of memory. 

- **`max_seq_len` (1000)**: A robust absolute positional embedding matrix capped at 1000 tokens is allocated before training begins.
- **Dynamic Growth**: During the execution of `train.py`, the training loop executes in progressive stages according to the `curriculum_schedule` (defined in `config.py`), automatically recreating the DataLoaders to continuously widen the `block_size` up to 1000 while progressively halving the `batch_size`.
- **Stream State Preservation**: Shifting dimensions dynamically on a HuggingFace `IterableDataset` is handled flawlessly. The raw Python iterator is passed directly from phase to phase so the dataset stream continues right where it left off!
