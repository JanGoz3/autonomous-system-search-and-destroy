import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DecisionTransformer.decision_transformer import WaypointTransformer

CHECKPOINT_FILE = "bc_ckpt_v2.pt"
ONNX_OUTPUT_FILE = "../Search and destroy/Assets/neural_nets/WaypointTransformer_yolov3.onnx"
TOLERANCE = 1e-3


class ExportWrapper(nn.Module):
    def __init__(self, model, state_mean, state_std, action_scale,
                 use_yaw_sincos, yaw_index, state_clip):
        super().__init__()
        self.model = model
        self.state_clip = float(state_clip)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.action_scale = float(action_scale)
        self.use_yaw_sincos = bool(use_yaw_sincos)
        self.yaw_index = int(yaw_index)

    def _expand_yaw(self, raw):
        i = self.yaw_index
        rad = raw[..., i:i + 1] * (3.141592653589793 / 180.0)
        return torch.cat([raw[..., :i], torch.sin(rad), torch.cos(rad),
                          raw[..., i + 1:]], dim=-1)

    def forward(self, states, attention_mask):
        x = self._expand_yaw(states) if self.use_yaw_sincos else states
        x = (x - self.state_mean) / self.state_std
        x = torch.clamp(x, -self.state_clip, self.state_clip)

        logits, mag = self.model.heads(x, attention_mask)
        logits = logits[:, -1, :]                       # (B, n_bins)
        mag_m = mag[:, -1] * self.action_scale          # (B,) w METRACH

        probs = F.softmax(logits, dim=-1)
        ang = self.model.bin_centers[logits.argmax(dim=-1)]        # (B,)
        action = torch.stack([mag_m * torch.sin(ang),
                              mag_m * torch.cos(ang)], dim=-1)     # (B, 2)
        return action, probs, mag_m.unsqueeze(-1)


def main():
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print("Konfiguracja checkpointu:")
    for k in ("state_dim", "act_dim", "context_length", "hidden_size", "n_layer",
              "n_head", "n_dir_bins", "action_scale", "use_yaw_sincos", "yaw_index"):
        print(f"  {k} = {cfg[k]}")
    print(f"  state_clip = {cfg.get('state_clip', 5.0)}")

    model = WaypointTransformer(
        state_dim=cfg["state_dim"], context_length=cfg["context_length"],
        hidden_size=cfg["hidden_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], dropout=0.0,
        n_dir_bins=cfg["n_dir_bins"], mag_weight=cfg["mag_weight"],
        label_smooth=cfg["label_smooth"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ExportWrapper(model, ckpt["state_mean"], ckpt["state_std"],
                            cfg["action_scale"], cfg["use_yaw_sincos"],
                            cfg["yaw_index"], cfg.get("state_clip", 5.0)).eval()

    K = cfg["context_length"]
    sd_raw = cfg["state_dim"] - 1 if cfg["use_yaw_sincos"] else cfg["state_dim"]

    dummy = (torch.randn(1, K, sd_raw), torch.ones(1, K))
    torch.onnx.export(
        wrapper, dummy, ONNX_OUTPUT_FILE, export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["states", "attention_mask"],
        output_names=["predicted_action", "dir_probs", "mag_m"],
        dynamo=False)
    print(f"\nZapisano {ONNX_OUTPUT_FILE}")

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_OUTPUT_FILE, providers=["CPUExecutionProvider"])

    def check(name, n_pad):
        s = torch.randn(1, K, sd_raw)
        s[..., cfg["yaw_index"]] = torch.rand(1, K) * 360.0
        m = torch.ones(1, K)
        if n_pad:
            m[0, :n_pad] = 0
            s[0, :n_pad] = 0
        with torch.no_grad():
            ref = wrapper(s, m)
        got = sess.run(None, {"states": s.numpy(), "attention_mask": m.numpy()})

        d_act = float(np.abs(ref[0].numpy() - got[0]).max())
        d_prob = float(np.abs(ref[1].numpy() - got[1]).max())
        finite = bool(np.isfinite(got[0]).all() and np.isfinite(got[1]).all())
        print(f"\n--- {name} (padding: {n_pad}/{K}) ---")
        print(f"  akcja PyTorch: {ref[0].numpy().ravel()}")
        print(f"  akcja ONNX:    {got[0].ravel()}   dlugosc={np.linalg.norm(got[0]):.3f} m")
        print(f"  pewnosc kierunku: {got[1].max():.3f}  max roznica: akcja={d_act:.2e} "
              f"rozklad={d_prob:.2e}  skonczone={finite}")

        assert finite, f"{name}: ONNX zwrocil NaN/Inf"
        assert d_act < TOLERANCE, f"{name}: rozbieznosc akcji {d_act:.2e} > {TOLERANCE}"
        assert d_prob < TOLERANCE, f"{name}: rozbieznosc rozkladu {d_prob:.2e}"

    check("Pelny bufor (przypadek docelowy)", 0)
    check("Krotka historia", K - 3)
    check("Skrajnie krotka historia", K - 1)

    print("\nOK - eksport zgodny z PyTorch we wszystkich trzech przypadkach.")
    print("\n=== Do wpisania w Inspectorze BCInference ===")
    print(f"  contextLength      = {K}")
    print(f"  stateDim           = {sd_raw}    (surowy wektor, yaw w stopniach)")
    print(f"  nDirBins           = {cfg['n_dir_bins']}")
    print(f"  yaw w kolumnie     = {cfg['yaw_index']}")
    if cfg.get("state_columns"):
        cols = cfg["state_columns"]
        print(f"  kolejnosc stanu    = {len(cols)} kolumn, pierwsze 5: {cols[:5]}")
    print(f"  action_scale       = {cfg['action_scale']:.4f} m (wbudowane w graf)")


if __name__ == "__main__":
    main()