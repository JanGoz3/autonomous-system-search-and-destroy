import argparse
import pickle
import random

import numpy as np
import torch

from models.DecisionTransformer.decision_transformer import DecisionTransformer, discount_cumsum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dt_dataset.pkl",
                         help="sciezka do .pkl (np. dt_sanity_dataset.pkl albo dt_dataset.pkl). "
                              "Domyslnie: dt_dataset.pkl")
    parser.add_argument("--K", type=int, default=20, help="dlugosc kontekstu (okno sekwencji)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=200)
    parser.add_argument("--scale", type=float, default=None,
                         help="normalizacja returns/rewards. Jesli None, liczona automatycznie z danych.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="car_dt_model.pt")
    args = parser.parse_args()

    device = args.device
    print(f"Uzywam urzadzenia: {device}")

    with open(args.dataset, "rb") as f:
        trajectories = pickle.load(f)

    state_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    max_ep_len = max(len(t["rewards"]) for t in trajectories) + 1

    for traj in trajectories:
        if "returns_to_go" not in traj:
            traj["returns_to_go"] = discount_cumsum(traj["rewards"], gamma=1.0)

    traj_lens = np.array([len(t["rewards"]) for t in trajectories])
    returns = np.array([t["rewards"].sum() for t in trajectories])

    states_concat = np.concatenate([t["observations"] for t in trajectories], axis=0)
    state_mean = np.mean(states_concat, axis=0)
    state_std = np.std(states_concat, axis=0) + 1e-6

    scale = args.scale if args.scale is not None else max(abs(returns).max(), 1.0)

    print("=" * 50)
    print(f"{len(trajectories)} epizodow, {traj_lens.sum()} krokow lacznie")
    print(f"state_dim={state_dim}, act_dim={act_dim}, max_ep_len={max_ep_len}")
    print(f"Srednia suma nagrod: {returns.mean():.2f}, std: {returns.std():.2f}, "
          f"min: {returns.min():.2f}, max: {returns.max():.2f}")
    print(f"Uzywana skala normalizacji return/reward: {scale:.2f}")
    print("=" * 50)

    num_trajectories = len(trajectories)
    p_sample = traj_lens / traj_lens.sum()
    K = args.K

    def get_batch(batch_size):
        batch_inds = np.random.choice(
            np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample
        )

        s, a, rtg, timesteps, mask = [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, len(traj["rewards"]) - 1)

            s.append(traj["observations"][si:si + K].reshape(1, -1, state_dim))
            a.append(traj["actions"][si:si + K].reshape(1, -1, act_dim))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1

            rtg.append(traj["returns_to_go"][si:si + K].reshape(1, -1, 1))

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, K - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, K - tlen, act_dim)), a[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, K - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, K - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, rtg, timesteps, mask

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_ep_len=max_ep_len,
        dropout=args.dropout,
    ).to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )

    for it in range(args.max_iters):
        model.train()
        losses = []
        for _ in range(args.num_steps_per_iter):
            s, a, rtg, timesteps, mask = get_batch(args.batch_size)
            action_target = a.clone()

            action_preds = model(s, a, rtg, timesteps, attention_mask=mask)

            loss = torch.mean(((action_preds - action_target) * mask.unsqueeze(-1)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"Iteracja {it + 1}/{args.max_iters} - "
              f"train_loss_mean: {np.mean(losses):.5f}, train_loss_std: {np.std(losses):.5f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "scale": scale,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "K": K,
            "max_ep_len": max_ep_len,
            "embed_dim": args.embed_dim,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
        },
        args.save_path,
    )
    print(f"\nZapisano model do: {args.save_path}")


if __name__ == "__main__":
    main()