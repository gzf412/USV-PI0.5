"""Serve a checkpoint trained with a USV 33- or 39-state configuration."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint directory containing params/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", choices=['pi05_usv', 'pi05_usv_gps'], default='pi05_usv',
                        help='Must match the configuration used to train the checkpoint')
    args = parser.parse_args()
    if not (args.checkpoint / "params").is_dir():
        parser.error("--checkpoint must contain a trained params/ directory")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config
    config = get_config(args.config)
    # The USV adapter owns fixed engineering scales; no arm normalization stats.
    policy = create_trained_policy(config, args.checkpoint, norm_stats={})
    metadata = dict(config.policy_metadata, policy_kind="usv_pi05_checkpoint",
                    checkpoint_source=str(args.checkpoint.resolve()))
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
