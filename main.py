# main.py — Main CLI for AutoGuard-RL
import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def collect_data(args):
    logging.info("Collecting CARLA data...")

def train_worldmodel(args):
    logging.info("Training world model (Dreamer-style placeholder)...")

def train_policy(args):
    logging.info("Training safe reinforcement learning policy...")

def evaluate(args):
    logging.info("Evaluating agent performance...")

def run_dashboard(args):
    logging.info("Launching dashboard (Streamlit).")
    os.system(f"streamlit run {ROOT / 'dashboard' / 'app.py'}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['collect','train_world','train_policy','eval','dashboard'], required=True)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == 'collect': collect_data(args)
    elif args.mode == 'train_world': train_worldmodel(args)
    elif args.mode == 'train_policy': train_policy(args)
    elif args.mode == 'eval': evaluate(args)
    elif args.mode == 'dashboard': run_dashboard(args)

if __name__ == '__main__':
    main()
