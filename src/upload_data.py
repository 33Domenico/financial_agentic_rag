import pandas as pd

df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)

df.to_json("../data/raw/data.json",orient="records",lines=True,force_ascii=False)#salva le righe del df in file json,lines scirve il json in formatio lines ce su una linea per valore

