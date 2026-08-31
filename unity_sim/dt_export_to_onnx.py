import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from decision_transformer import DecisionTransformer

torch.backends.mha.set_fastpath_enabled(False)

CHECKPOINT_FILE = "dt_checkpoint.pt"
ONNX_OUTPUT_FILE = "DecisionTransformer_new.onnx"


class DTExportWrapper(nn.Module):
    def __init__(self, model, state_mean, state_std, return_scale, action_scale):
        super().__init__()
        self.model = model
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.return_scale = float(return_scale)
        self.action_scale = float(action_scale)

    def forward(self, raw_states, actions, raw_returns_to_go, timesteps, attention_mask):
        states_norm = (raw_states - self.state_mean) / self.state_std
        rtg_scaled = raw_returns_to_go / self.return_scale
        preds = self.model(states_norm, actions, rtg_scaled, timesteps, attention_mask)
        last = torch.nan_to_num(preds[:, -1, :], nan=0.0, posinf=0.0, neginf=0.0)
        return last * self.action_scale


def build_model(cfg):
    return DecisionTransformer(
        state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], max_ep_len=cfg["max_ep_len"],
        action_head=cfg.get("action_head", "continuous"),
        n_dir_bins=cfg.get("n_dir_bins", 36),
    )


def main():
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print("Konfiguracja:", cfg)

    action_scale = cfg.get("action_scale")
    if action_scale is None:
        raise RuntimeError("Brak action_scale w checkpoincie - przetrenuj nowym train_dt.py")

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = DTExportWrapper(model, ckpt["state_mean"], ckpt["state_std"],
                              cfg["return_scale"], action_scale).eval()

    K, sd, ad = cfg["context_length"], cfg["state_dim"], cfg["act_dim"]
    max_ep_len = cfg["max_ep_len"]

    dummy = (torch.randn(1, K, sd), torch.zeros(1, K, ad),
             torch.full((1, K, 1), -20.0),
             torch.arange(K, dtype=torch.long).unsqueeze(0), torch.ones(1, K))

    print(f"\nEksport do {ONNX_OUTPUT_FILE} "
          f"(K={K}, state_dim={sd}, act_dim={ad}, head={cfg.get('action_head')}, "
          f"action_scale={action_scale:.4f} m, return_scale={cfg['return_scale']:.3f})")

    torch.onnx.export(
        wrapper, dummy, ONNX_OUTPUT_FILE, export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["states", "actions", "returns_to_go", "timesteps", "attention_mask"],
        output_names=["predicted_action"], dynamo=False,
    )

    sess = ort.InferenceSession(ONNX_OUTPUT_FILE)

    def compare(name, mask_pad):
        s = torch.randn(1, K, sd)
        a = torch.zeros(1, K, ad)          # w treningu akcje byly zerowane
        r = torch.full((1, K, 1), -20.0)
        t = torch.arange(K, dtype=torch.long).unsqueeze(0)
        m = torch.ones(1, K)
        if mask_pad:
            m[0, :mask_pad] = 0
            t[0, :mask_pad] = 0
        with torch.no_grad():
            ref = wrapper(s, a, r, t, m).numpy()
        got = sess.run(None, {"states": s.numpy(), "actions": a.numpy(),
                              "returns_to_go": r.numpy(), "timesteps": t.numpy(),
                              "attention_mask": m.numpy()})[0]
        d = np.abs(ref - got).max()
        print(f"\n--- {name} ---")
        print(f"  PyTorch: {ref}")
        print(f"  ONNX:    {got}")
        print(f"  max roznica: {d:.8f}   dlugosc ONNX: {np.linalg.norm(got):.3f} m")
        return d

    compare("Test 1: krotka historia (padding)", mask_pad=K - 3)
    d2 = compare("Test 2: pelny bufor (przypadek docelowy)", mask_pad=0)

    print()
    if d2 < 1e-3:
        print("OK - eksport zgodny z PyTorch dla pelnego bufora.")
    else:
        print("UWAGA - rozbieznosc w pelnym buforze, sprawdz eksport.")

    print(f"\nDo wpisania w Inspectorze DTInference:")
    print(f"  contextLength   = {K}")
    print(f"  stateDim        = {sd}")
    print(f"  maxEpLen        = {max_ep_len}   (do przycinania timesteps)")
    print(f"  zeroActionsInContext = {bool(cfg.get('zero_actions', False))}")
    print(f"\nPlik: {ONNX_OUTPUT_FILE}")


if __name__ == "__main__":
    main()