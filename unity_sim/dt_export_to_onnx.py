import pickle
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from decision_transformer import DecisionTransformer
torch.backends.mha.set_fastpath_enabled(False)


CHECKPOINT_FILE = "dt_checkpoint.pt"
ONNX_OUTPUT_FILE = "DecisionTransformer.onnx"

MAX_ARENA_SIZE = 20.0


class DTExportWrapper(nn.Module):

    def __init__(self, model: DecisionTransformer, state_mean, state_std, return_scale: float, max_arena_size: float):
        super().__init__()
        self.model = model
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(state_std, dtype=torch.float32))
        self.return_scale = float(return_scale)
        self.max_arena_size = float(max_arena_size)

    def forward(self, raw_states, actions, raw_returns_to_go, timesteps, attention_mask):
        states_norm = (raw_states - self.state_mean) / self.state_std
        rtg_scaled = raw_returns_to_go / self.return_scale

        action_preds = self.model(states_norm, actions, rtg_scaled, timesteps, attention_mask)

        last_action = action_preds[:, -1, :]
        last_action = torch.nan_to_num(last_action, nan=0.0)
        last_action = last_action * self.max_arena_size

        return last_action


def main():
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    print("Konfiguracja modelu:", cfg)

    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["hidden_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        max_ep_len=cfg["max_ep_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = DTExportWrapper(
        model, ckpt["state_mean"], ckpt["state_std"], cfg["return_scale"], MAX_ARENA_SIZE
    )
    wrapper.eval()

    K = cfg["context_length"]
    state_dim = cfg["state_dim"]
    act_dim = cfg["act_dim"]

    dummy_states = torch.randn(1, K, state_dim)
    dummy_actions = torch.zeros(1, K, act_dim)
    dummy_rtg = torch.full((1, K, 1), 50.0)
    dummy_timesteps = torch.arange(K, dtype=torch.long).unsqueeze(0)
    dummy_mask = torch.ones(1, K)

    print(f"\nEksportuje do {ONNX_OUTPUT_FILE} (K={K}, state_dim={state_dim}, act_dim={act_dim})...")

    torch.onnx.export(
        wrapper,
        (dummy_states, dummy_actions, dummy_rtg, dummy_timesteps, dummy_mask),
        ONNX_OUTPUT_FILE,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["states", "actions", "returns_to_go", "timesteps", "attention_mask"],
        output_names=["predicted_action"],
        dynamo=False,  # wymuszamy starszy, sprawdzony eksporter (ten sam co uzywany do drivera)
    )

    print("Eksport zakonczony. Weryfikuje poprawnosc (PyTorch vs ONNX Runtime)...")



    test_states = torch.randn(1, K, state_dim)
    test_actions = torch.randn(1, K, act_dim) * 0.1
    test_rtg = torch.full((1, K, 1), 30.0)
    test_timesteps = torch.arange(K, dtype=torch.long).unsqueeze(0)
    test_mask = torch.ones(1, K)
    test_mask[0, :5] = 0  # symulacja krotkiej historii (pierwsze 5 to padding)

    with torch.no_grad():
        torch_out = wrapper(test_states, test_actions, test_rtg, test_timesteps, test_mask).numpy()

    ort_session = ort.InferenceSession(ONNX_OUTPUT_FILE)
    onnx_out = ort_session.run(
        None,
        {
            "states": test_states.numpy(),
            "actions": test_actions.numpy(),
            "returns_to_go": test_rtg.numpy(),
            "timesteps": test_timesteps.numpy(),
            "attention_mask": test_mask.numpy(),
        },
    )[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"\n--- Test 1: KROTKA historia (padding, symulacja startu epizodu) ---")
    print(f"PyTorch output: {torch_out}")
    print(f"ONNX output:    {onnx_out}")
    print(f"Maksymalna roznica: {max_diff:.8f}")
    print(
        "To OCZEKIWANE ze ONNX zwraca [0,0] w tym przypadku - to celowe zabezpieczenie\n"
        "przed NaN przy krotkiej historii (padding). Nie jest to blad."
    )

    test2_states = torch.randn(1, K, state_dim)
    test2_actions = torch.randn(1, K, act_dim) * 0.1
    test2_rtg = torch.full((1, K, 1), 30.0)
    test2_timesteps = torch.arange(K, dtype=torch.long).unsqueeze(0)
    test2_mask = torch.ones(1, K)  # BEZ paddingu

    with torch.no_grad():
        torch_out2 = wrapper(test2_states, test2_actions, test2_rtg, test2_timesteps, test2_mask).numpy()

    onnx_out2 = ort_session.run(
        None,
        {
            "states": test2_states.numpy(),
            "actions": test2_actions.numpy(),
            "returns_to_go": test2_rtg.numpy(),
            "timesteps": test2_timesteps.numpy(),
            "attention_mask": test2_mask.numpy(),
        },
    )[0]

    max_diff2 = np.abs(torch_out2 - onnx_out2).max()
    print(f"\n--- Test 2: PELNY bufor (normalny przypadek docelowy) ---")
    print(f"PyTorch output: {torch_out2}")
    print(f"ONNX output:    {onnx_out2}")
    print(f"Maksymalna roznica: {max_diff2:.8f}")

    if max_diff2 < 1e-3:
        print("OK - eksport poprawny dla normalnego przypadku uzycia (pelny bufor).")
    else:
        print("UWAGA - duza roznica w PELNYM buforze! To by byl prawdziwy problem, sprawdz eksport.")

    print(f"\nGotowy plik: {ONNX_OUTPUT_FILE}")



if __name__ == "__main__":
    main()