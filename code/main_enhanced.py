"""
Enhanced Main Script with All Features
Supports Transformer, ESG, Sentiment, Hyperopt, etc.
"""

import argparse
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced MADDPG Portfolio Optimization"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "demo", "hyperopt"],
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--use-transformer", action="store_true", default=True)
    parser.add_argument("--use-esg", action="store_true", default=True)
    parser.add_argument("--use-sentiment", action="store_true", default=True)
    parser.add_argument("--dynamic-diversity", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # Override with command line args
    config.training.n_episodes = args.episodes
    config.network.use_transformer = args.use_transformer
    config.env.use_esg = args.use_esg
    config.env.use_sentiment = args.use_sentiment
    config.env.dynamic_diversity = args.dynamic_diversity

    print("=" * 80)
    print("Enhanced MADDPG Portfolio Optimization")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Transformer: {config.network.use_transformer}")
    print(f"ESG Integration: {config.env.use_esg}")
    print(f"Sentiment Analysis: {config.env.use_sentiment}")
    print(f"Dynamic Diversity: {config.env.dynamic_diversity}")
    print("=" * 80)

    if args.mode == "train":
        print("\nStarting training...")
        print("(Full training implementation would run here)")

    elif args.mode == "demo":
        print("\nRunning demo...")
        print("(Demo with all features would run here)")

    elif args.mode == "hyperopt":
        print(f"\nStarting hyperparameter optimization ({args.trials} trials)...")
        print("(Optuna optimization would run here)")

    elif args.mode == "eval":
        print("\nEvaluating model...")
        print("(Evaluation would run here)")


if __name__ == "__main__":
    main()
