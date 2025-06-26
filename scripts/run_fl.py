#!/usr/bin/env python
import argparse, os, sys, yaml

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your modular imports (must exist in pneumonia_detector/)
from pneumonia_detector.data_loader import get_data_loaders
from pneumonia_detector.models      import build_model
from pneumonia_detector.trainer     import train_model, evaluate_model
from pneumonia_detector.fl_core     import run_federated_simulation

def main(cfg_path):
    if not os.path.exists(cfg_path):
        sys.exit(f"[Error] Config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Unpack sections
    m_cfg = cfg['model']
    t_cfg = cfg['training']
    f_cfg = cfg['federated']
    d_cfg = cfg['data']

    # Resolve absolute manifest path
    manifest = os.path.abspath(d_cfg['manifest_path'])

    print(f"→ Model: {m_cfg['name']} (pretrained={m_cfg['pretrained']})")
    print(f"→ FL: {f_cfg['num_clients']} clients × {f_cfg['num_rounds']} rounds ({f_cfg['strategy']})")
    print(f"→ Locally: {t_cfg['epochs']} epochs, BS={t_cfg['batch_size']}, LR={t_cfg['learning_rate']}")
    print(f"→ Data manifest at: {manifest}")

    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        manifest_path=manifest,
        img_size=d_cfg['img_size'],
        batch_size=t_cfg['batch_size']
    )

    # Model
    model = build_model(m_cfg['name'], m_cfg['pretrained'])

    # Run FL simulation
    run_federated_simulation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_clients=f_cfg['num_clients'],
        num_rounds=f_cfg['num_rounds'],
        strategy=f_cfg['strategy'],
        epochs_per_round=t_cfg['epochs'],
        learning_rate=t_cfg['learning_rate']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run config-driven federated experiment")
    parser.add_argument(
        "--config", "-c",
        default="configs/densenet_fl.yaml",
        help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
