"""Train briefly and compare layerwise gradient norms on a fixed probe batch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["plain_56", "resnet_56"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="experiments/05_gradient_flow")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader, Subset
    from resnet_research.analysis.gradient_flow import probe_gradient_flow
    from resnet_research.data.dataset import build_dataloaders, load_cifar10
    from resnet_research.models import build_model
    from resnet_research.training.losses import build_loss
    from resnet_research.training.schedulers import build_optimizer
    from resnet_research.training.runner import resolve_device
    from resnet_research.utils.config import load_config
    from resnet_research.utils.seed import set_seed

    device = resolve_device(args.device)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    results = {"device": str(device), "epochs": args.epochs, "models": {}}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in args.models:
        overrides = ["training.amp=false"]
        if args.data_root:
            overrides.append("data.root=" + args.data_root)
        cfg = load_config([f"configs/models/{name}.yaml"], overrides=overrides)
        set_seed(cfg["seed"])
        loader, _, _ = build_dataloaders(cfg, num_workers=args.num_workers)
        if args.subset_size is not None:
            if args.subset_size < 1:
                parser.error("subset-size must be positive")
            loader = DataLoader(Subset(loader.dataset, range(min(args.subset_size, len(loader.dataset)))),
                                batch_size=cfg["training"]["batch_size"], shuffle=True,
                                num_workers=args.num_workers)
        clean_train, _, _ = load_cifar10(
            cfg["data"]["root"], split_seed=cfg["split_seed"], val_size=cfg["data"]["val_size"],
            augment="minimal",
        )
        probe = next(iter(DataLoader(clean_train, batch_size=32, shuffle=False)))
        model = build_model(cfg["model"]).to(device)
        loss_fn = build_loss(cfg)
        optimizer = build_optimizer(model, cfg)
        snapshots = {}
        for epoch in range(args.epochs + 1):
            if epoch in (0, 1, args.epochs):
                snapshots[str(epoch)] = probe_gradient_flow(model, probe, device=device, loss_fn=loss_fn)
            if epoch == args.epochs:
                break
            model.train()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(images), labels).backward()
                optimizer.step()
        results["models"][name] = snapshots
        final = snapshots[str(args.epochs)]
        ax.plot(range(len(final)), list(final.values()), label=name)
    ax.set(xlabel="Convolutional and linear layer order", ylabel="Weight gradient L2 norm",
           title="Gradient norms on a fixed CIFAR-10 batch")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "gradient_norms.png", dpi=160)
    plt.close(fig)
    (folder / "gradient_norms.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
