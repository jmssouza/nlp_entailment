import pandas as pd 
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset
import warnings
import torch 

warnings.filterwarnings("ignore")


def main():
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')

    # carrega dados
    df_dev = pd.read_xml("assin2-dev.xml")

    # processa dados
    df_dev.drop(columns=['id', 'similarity'], axis=1, inplace=True)
    df_dev["t"] = df_dev["t"].astype(str)
    df_dev["h"] = df_dev["h"].astype(str)
    df_dev["text"] = df_dev["t"] +  " [SEP] " + df_dev["h"]
    df_dev["label"] = df_dev["entailment"].apply(lambda x: 1 if x=="Entailment" else 0)
    df_dev["text"] = df_dev["text"].apply(lambda x: x.lower())
    df_dev.drop(columns=["entailment", "t", "h"], axis=1, inplace=True)
    dataset_dev = Dataset.from_pandas(df_dev)

    # tokeniza dados
    tokenized_valid = tokenizer(df_dev['text'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')

    # carrega modelo 
    loaded_model = AutoModelForSequenceClassification.from_pretrained('bertimbau-large-fine-tune')
    print(loaded_model)

    # divide em lotes
    batch_size = 5
    predictions = []
    for i in range(0, len(dataset_dev), batch_size):
        tokenized_batch = {k: v[i:i+batch_size] for k, v in tokenized_valid.items()}
        batch_outputs = loaded_model(**tokenized_batch)
        batch_predictions = batch_outputs.logits.argmax(dim=1)
        predictions.append(batch_predictions)

    # concatena previsões e labels
    predictions = torch.cat(predictions)
    labels = torch.tensor(dataset_dev["label"], dtype=torch.long)

    # exibe relatório de classificação
    report = classification_report(labels, predictions)
    print(report)

    # salvar o classification report em um arquivo de texto
    with open('classification_report_dev.txt', 'w') as f:
        print(report, file=f)

if __name__ == "__main__":
    main()
