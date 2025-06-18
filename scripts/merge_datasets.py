#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent))

from data_processor import DatasetProcessor
from metadata_utils import save_metadata_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='+', required=True)
    parser.add_argument('--output-dir', '-o', default='data/processed')
    parser.add_argument('--train-ratio', '-tr', type=float, default=0.7)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--report', '-r', action='store_true')
    
    args = parser.parse_args()
    
    logger.info("Начинаю обработку датасетов")
    
    processor = DatasetProcessor(args.input, args.output_dir)
    results = processor.process_all(args.train_ratio, args.seed)
    
    if args.report:
        output_dir = Path(args.output_dir)
        save_metadata_report(results['merged'], str(output_dir / "report.txt"))
    
    logger.info("Готово!")


if __name__ == "__main__":
    main() 