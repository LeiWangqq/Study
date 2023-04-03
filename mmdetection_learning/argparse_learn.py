"""
@FileName：argparse_learn.py\n
@Description：\n
@Author：Wang.Lei\n
@Time：2022/11/30 14:54\n
@Department：Postgrate\n
"""
import argparse
parser = argparse.ArgumentParser(description= 'test')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
args = parser.parse_args()
print(args.sparse)
print(args.seed)
print(args.epochs)


