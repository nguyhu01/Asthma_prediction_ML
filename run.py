import argparse

from train import SeverityClassificationPipeline

file_path = "data/processed-data.csv"

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_size', dest='test_size', help='Test Size (default 0.2)', type=float, default=0.2)
    parser.add_argument('--random_state', dest='random_state', help='Random State', type=int, default=42)
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=None)
    parser.add_argument('--max_depth', dest='max_depth', help='Max Depth', type=int, default=None)

    return parser.parse_args()

def main(args):
    train_model = SeverityClassificationPipeline(file_path, args.test_size, args.random_state, args.lr, args.max_depth)
    train_model.run_augmented_data()

if __name__ == '__main__':
    main(parse_args())