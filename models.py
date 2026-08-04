"""
Custom classification heads / architectures for the Subtask 1 experiment battery.

All of these wrap a plain `AutoModel` encoder (not `AutoModelForSequenceClassification`,
which bakes in CLS-token pooling) so the pooling strategy is under our control. They
return a `transformers.modeling_outputs.SequenceClassifierOutput` so they remain
drop-in-compatible with `transformers.Trainer` / our `WeightedTrainer`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput

# T5-family checkpoints (e.g. byt5) need T5EncoderModel, not AutoModel -- AutoModel on a
# T5 repo returns the full encoder-decoder T5Model, whose forward() expects
# decoder_input_ids and isn't a drop-in (input_ids, attention_mask) -> hidden_states call
# the way a plain BERT-family AutoModel is.
_T5_FAMILY_PREFIXES = ("google/byt5", "google/mt5", "google/t5", "google-t5/")


class _Base(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        if model_name.startswith(_T5_FAMILY_PREFIXES) or getattr(self.config, "model_type", "") == "t5":
            self.encoder = T5EncoderModel.from_pretrained(model_name, use_safetensors=False)
            self.hidden_size = self.config.d_model
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.config.hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        return self.encoder(**kwargs).last_hidden_state  # (B, T, H)

    @staticmethod
    def masked_mean(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class PooledClassifier(_Base):
    """pooling in {'mean', 'cls', 'cls_mean', 'attention'}"""

    def __init__(self, model_name, num_labels, pooling="mean", dropout=0.1):
        super().__init__(model_name, num_labels, dropout)
        self.pooling = pooling
        in_dim = self.hidden_size * 2 if pooling == "cls_mean" else self.hidden_size
        if pooling == "attention":
            self.attn_proj = nn.Linear(self.hidden_size, 1)
        self.classifier = nn.Linear(in_dim, num_labels)

    def pool(self, hidden_states, attention_mask):
        if self.pooling == "mean":
            return self.masked_mean(hidden_states, attention_mask)
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if self.pooling == "cls_mean":
            return torch.cat([hidden_states[:, 0], self.masked_mean(hidden_states, attention_mask)], dim=-1)
        if self.pooling == "attention":
            scores = self.attn_proj(hidden_states).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (hidden_states * weights).sum(1)
        raise ValueError(f"Unknown pooling '{self.pooling}'")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                sample_weight=None, return_hidden=False, **kwargs):
        hidden_states = self.encode(input_ids, attention_mask, token_type_ids)
        pooled = self.dropout(self.pool(hidden_states, attention_mask))
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        out = SequenceClassifierOutput(loss=loss, logits=logits)
        if return_hidden:
            out.pooled_hidden = pooled
        return out


class DialectAwarePooledClassifier(_Base):
    """Mean-pools the text as usual, but ALSO concatenates a learned dialect embedding
    before the classifier head. Dialect is a labeled property available at BOTH train and
    test time (unlike source-polarity in Subtask 2, which had to be detected) but was
    never fed to the model as an input feature anywhere in this project -- every technique
    so far only used it for CV stratification / IRM environments. Real risk: Lebanese has
    zero gold rows, so its embedding only gets gradient signal from self-training
    pseudo-labels, not gold data -- weaker-calibrated than the other 4 dialects' slots,
    but not a pure never-seen case either."""

    def __init__(self, model_name, num_labels, num_dialects, dialect_emb_dim=16, dropout=0.1):
        super().__init__(model_name, num_labels, dropout)
        self.dialect_embed = nn.Embedding(num_dialects, dialect_emb_dim)
        self.classifier = nn.Linear(self.hidden_size + dialect_emb_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                dialect_ids=None, sample_weight=None, return_hidden=False, **kwargs):
        hidden_states = self.encode(input_ids, attention_mask, token_type_ids)
        pooled_text = self.masked_mean(hidden_states, attention_mask)
        dial_emb = self.dialect_embed(dialect_ids)
        pooled = self.dropout(torch.cat([pooled_text, dial_emb], dim=-1))
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        out = SequenceClassifierOutput(loss=loss, logits=logits)
        if return_hidden:
            out.pooled_hidden = pooled
        return out


class CharHybridClassifier(_Base):
    """Subword encoder (mean-pooled) concatenated with a from-scratch character-level
    encoder (CNN or BiLSTM) over the raw sentence text, to add robustness to unseen
    dialectal spellings that fragment badly under the subword tokenizer's vocabulary."""

    def __init__(self, model_name, num_labels, char_vocab, mode="cnn", char_emb_dim=64,
                 char_hidden=128, dropout=0.1, max_char_len=256):
        super().__init__(model_name, num_labels, dropout)
        self.mode = mode
        self.char_vocab = char_vocab  # dict: char -> id (0 reserved for PAD, 1 for UNK)
        self.max_char_len = max_char_len
        self.char_embed = nn.Embedding(len(char_vocab), char_emb_dim, padding_idx=0)
        if mode == "cnn":
            self.char_convs = nn.ModuleList([
                nn.Conv1d(char_emb_dim, char_hidden, kernel_size=k, padding=k // 2) for k in (3, 5, 7)
            ])
            char_out_dim = char_hidden * len(self.char_convs)
        elif mode == "bilstm":
            self.char_lstm = nn.LSTM(char_emb_dim, char_hidden, batch_first=True, bidirectional=True)
            char_out_dim = char_hidden * 2
        else:
            raise ValueError(mode)
        self.classifier = nn.Linear(self.hidden_size + char_out_dim, num_labels)

    def encode_chars(self, char_ids):
        # char_ids: (B, L)
        emb = self.char_embed(char_ids)  # (B, L, E)
        mask = (char_ids != 0).float()
        if self.mode == "cnn":
            x = emb.transpose(1, 2)  # (B, E, L)
            feats = []
            for conv in self.char_convs:
                c = F.relu(conv(x))  # (B, H, L)
                c = c.masked_fill((mask == 0).unsqueeze(1), float("-inf"))
                pooled, _ = c.max(dim=2)
                pooled = torch.nan_to_num(pooled, neginf=0.0)
                feats.append(pooled)
            return torch.cat(feats, dim=-1)
        else:
            lengths = mask.sum(1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.char_lstm(packed)
            return torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2*H)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, char_ids=None,
                labels=None, sample_weight=None, **kwargs):
        hidden_states = self.encode(input_ids, attention_mask, token_type_ids)
        subword_repr = self.masked_mean(hidden_states, attention_mask)
        char_repr = self.encode_chars(char_ids)
        combined = self.dropout(torch.cat([subword_repr, char_repr], dim=-1))
        logits = self.classifier(combined)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def build_char_ids(self, texts, device):
        ids = torch.zeros(len(texts), self.max_char_len, dtype=torch.long)
        for i, t in enumerate(texts):
            for j, ch in enumerate(str(t)[: self.max_char_len]):
                ids[i, j] = self.char_vocab.get(ch, 1)
        return ids.to(device)


def build_char_vocab(texts):
    chars = sorted({ch for t in texts for ch in str(t)})
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        vocab[ch] = len(vocab)
    return vocab
