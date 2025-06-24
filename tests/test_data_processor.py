import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DatasetProcessor

# Создание процессора с файлами для объединения
processor = DatasetProcessor(
    input_files=[
      "data/generated/deepseek_v3_dataset_manual_final.json",
      "data/generated/llama-4-maverick-1-manual_final.json",
      "data/generated/llama-4-maverick-2_final_annotated.json",
      "data/generated/llama-4-maverick-3_complete_annotated.json",
      "data/generated/llama-4-maverick-4.1_super_annotated.json",
      "data/generated/llama-4-maverick-4.2_super_annotated.json",
      "data/generated/llama-4-maverick-4.3_super_annotated.json",
      "data/generated/llama-4-maverick-4.4_super_annotated.json",
      "data/generated/llama-4-maverick-4.5_super_annotated.json",
      "data/generated/llama-4-maverick-4.6_super_annotated.json",
      "data/generated/llama-4-maverick-4.7_super_annotated.json",
      "data/generated/llama-4-maverick-4.8_super_annotated.json",
      "data/generated/llama-4-maverick-4.9_super_annotated.json",
      "data/generated/llama-4-maverick-4.10_super_annotated.json",
    ],
    output_dir="data/processed"
)

# Объединение датасетов (автоматически загружает все input_files)
merged_data = processor.merge_datasets()

# Анализ объединенного датасета
stats = processor.calculate_stats(merged_data.get('dialogs', []))
print(f"Всего диалогов: {stats.total_dialogs}")
print(f"Диалогов с ПД: {stats.pd_dialogs} ({stats.pd_percentage:.1f}%)")

# Разделение на train/val
train_data, val_data = processor.split_dataset(merged_data, train_ratio=0.7)

# Сохранение результатов
processor.save_dataset(merged_data, "merged_dataset.json")
processor.save_dataset(train_data, "train_dataset.json")
processor.save_dataset(val_data, "val_dataset.json")

# Загрузка ранее сохраненного файла
saved_data = processor.load_dataset("data/processed/merged_dataset.json")

# import pandas as pd

# df = pd.read_csv("data/reports/merged_dataset_combined_metrics.csv")

# print(df.head(), len(df), len(df['method'].unique()))

