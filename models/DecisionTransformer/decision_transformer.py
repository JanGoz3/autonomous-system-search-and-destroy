"""
WaypointTransformer - sekwencyjny model behavior cloning dla waypointow.

Rozne wzgledem DecisionTransformer:
  * BRAK embed_return - nie ma warunkowania return-to-go
  * BRAK embed_timestep - pozycja jest liczona WZGLEDEM OKNA kontekstu (0..K-1),
    nie wzgledem numeru kroku w epizodzie. Dzieki temu trening i inferencja
    widza dokladnie te same indeksy pozycyjne.
  * BRAK tokenow akcji - jeden token na krok, same stany. Nie ma exposure bias
    (w treningu kontekst mial akcje eksperta, w inferencji wlasne predykcje).
  * Maska uwagi z ODBLOKOWANA DIAGONALA - zaden wiersz softmaxu nie jest w pelni
    zamaskowany, wiec padding nie generuje NaN.

Glowa akcji: kierunek jako klasyfikacja na n_dir_bins kubelkow + osobna regresja
dlugosci. Kat liczony jako atan2(x, z), zgodnie z ukladem auta w Unity
(x = w prawo, z = do przodu).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mha.set_fastpath_enabled(False)


NEG = -1e9   # zamiast -inf: przenosne miedzy ONNX Runtime i Unity Inference Engine


class WaypointTransformer(nn.Module):
    def __init__(self, state_dim, context_length=20, hidden_size=128,
                 n_layer=3, n_head=4, dropout=0.1,
                 n_dir_bins=36, mag_weight=1.0, label_smooth=0.15):
        super().__init__()
        self.state_dim = state_dim
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_dir_bins = n_dir_bins
        self.mag_weight = mag_weight
        self.label_smooth = label_smooth

        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_pos = nn.Embedding(context_length, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_head, dim_feedforward=4 * hidden_size,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer,
                                                 enable_nested_tensor=False)

        self.predict_dir = nn.Linear(hidden_size, n_dir_bins)
        self.predict_mag = nn.Linear(hidden_size, 1)

        centers = (torch.arange(n_dir_bins, dtype=torch.float32) + 0.5) \
                  * (2 * np.pi / n_dir_bins) - np.pi
        self.register_buffer("bin_centers", centers)

        K = context_length
        # Maski liczone RAZ, jako bufory. Liczenie ich w forward przez torch.eye
        # eksportuje sie do wezla ONNX EyeLike, ktorego ani ONNX Runtime, ani
        # Unity Inference Engine nie implementuja.
        # persistent=False: to stale, nie parametry. Trzymanie ich w state_dict
        # zepsulo by wczytywanie checkpointow przy kazdej zmianie K.
        self.register_buffer("causal_add",
                             torch.triu(torch.full((K, K), NEG), diagonal=1),
                             persistent=False)
        self.register_buffer("keep_diag",
                             1.0 - torch.eye(K), persistent=False)   # 0 na diagonali
        self.register_buffer("positions",
                             torch.arange(K, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------

    def _attn_mask(self, attention_mask):
        """Addytywna maska (B*n_head, K, K).

        Sklada sie z przyczynowosci i paddingu, ale diagonala pozostaje
        odblokowana: wiersz softmaxu zamaskowany w CALOSCI daje NaN, ktory w
        kolejnej warstwie mnozy sie przez wage 0 (0 * NaN = NaN) i rozlewa sie
        na wszystkie pozycje. Na tym polegal blad w poprzedniej wersji.
        """
        B, K = attention_mask.shape
        pad_add = (1.0 - attention_mask).unsqueeze(1) * NEG      # (B, 1, K)
        pad_add = pad_add * self.keep_diag.unsqueeze(0)          # diagonala wolna
        m = self.causal_add.unsqueeze(0) + pad_add               # (B, K, K)
        return m.unsqueeze(1).expand(B, self.n_head, K, K).reshape(B * self.n_head, K, K)

    def _hidden(self, states, attention_mask):
        B, T, _ = states.shape
        pos = self.positions[:T].unsqueeze(0).expand(B, T)

        x = self.embed_ln(self.embed_state(states) + self.embed_pos(pos))
        return self.transformer(x, mask=self._attn_mask(attention_mask))

    def heads(self, states, attention_mask):
        """Surowe wyjscia glow: (logits kierunku, dlugosc w jednostkach akcji)."""
        h = self._hidden(states, attention_mask)
        logits = self.predict_dir(h)                       # (B, T, n_bins)
        mag = F.softplus(self.predict_mag(h)).squeeze(-1)   # (B, T)
        return logits, mag

    def forward(self, states, attention_mask):
        """Wektory (B, T, 2) w jednostkach akcji (znormalizowanych)."""
        logits, mag = self.heads(states, attention_mask)
        ang = self.bin_centers[logits.argmax(dim=-1)]       # (B, T)
        return torch.stack([mag * torch.sin(ang), mag * torch.cos(ang)], dim=-1)

    # ------------------------------------------------------------------

    def compute_loss(self, states, attention_mask, target_actions, loss_mask):
        """target_actions: (B, T, 2) w jednostkach akcji. loss_mask: (B, T)."""
        logits, mag = self.heads(states, attention_mask)
        B, T, n = logits.shape

        tgt_ang = torch.atan2(target_actions[..., 0], target_actions[..., 1])
        tgt_bin = torch.floor((tgt_ang + np.pi) / (2 * np.pi / n)).long().clamp(0, n - 1)

        s = self.label_smooth
        soft = torch.zeros(B, T, n, device=logits.device)
        soft.scatter_(2, tgt_bin.unsqueeze(-1), 1.0 - 2 * s)
        one = torch.full((B, T, 1), s, device=logits.device)
        soft.scatter_add_(2, ((tgt_bin - 1) % n).unsqueeze(-1), one)
        soft.scatter_add_(2, ((tgt_bin + 1) % n).unsqueeze(-1), one)

        ce = -(soft * F.log_softmax(logits, dim=-1)).sum(-1)          # (B, T)
        mag_err = (mag - target_actions.norm(dim=-1)) ** 2            # (B, T)

        denom = loss_mask.sum().clamp(min=1.0)
        total = ((ce + self.mag_weight * mag_err) * loss_mask).sum() / denom
        return total, (ce * loss_mask).sum() / denom, (mag_err * loss_mask).sum() / denom

    @torch.no_grad()
    def direction_accuracy(self, states, attention_mask, target_actions, loss_mask,
                           tol_bins=1):
        """Udzial krokow, w ktorych argmax kierunku mie sci sie w tol_bins od celu."""
        logits, _ = self.heads(states, attention_mask)
        n = self.n_dir_bins
        tgt_ang = torch.atan2(target_actions[..., 0], target_actions[..., 1])
        tgt_bin = torch.floor((tgt_ang + np.pi) / (2 * np.pi / n)).long().clamp(0, n - 1)
        pred_bin = logits.argmax(dim=-1)

        d = (pred_bin - tgt_bin).abs()
        d = torch.minimum(d, n - d)                        # kubelki sa cykliczne
        hit = (d <= tol_bins).float()
        return (hit * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)