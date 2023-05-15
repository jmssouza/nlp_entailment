import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import warnings
import torch 
import evaluate

warnings.filterwarnings("ignore")


def data_processing(path_train="assin2-train.xml", path_test="assin2-test.xml"):

    # carregar datasets 
    df_train = pd.read_xml(path_train)
    df_test = pd.read_xml(path_test)

    # deletar colunas
    df_test.drop(columns=['id', 'similarity'], axis=1, inplace=True)
    df_train.drop(columns=['id', 'similarity'], axis=1, inplace=True)


    # concaternar T e H 
    df_train["t"] = df_train["t"].astype(str)
    df_train["h"] = df_train["h"].astype(str)

    df_test["t"] = df_test["t"].astype(str)
    df_test["h"] = df_test["h"].astype(str)

    df_train["text"] = df_train["t"] +  " [SEP] " + df_train["h"]
    df_test["text"] = df_test["t"] +  " [SEP] " + df_test["h"]


    # transformar label (entailment = 1 | none = 0)
    df_train["label"] = df_train["entailment"].apply(lambda x: 1 if x=="Entailment" else 0)
    df_test["label"] = df_test["entailment"].apply(lambda x: 1 if x=="Entailment" else 0)


    # transformar frase em lower case 
    df_train["text"] = df_train["text"].apply(lambda x: x.lower())
    df_test["text"] = df_test["text"].apply(lambda x: x.lower())

    # deletar colunas 
    df_train.drop(columns=["entailment", "t", "h"], axis=1, inplace=True)
    df_test.drop(columns=["entailment", "t", "h"], axis=1, inplace=True)

    return df_train, df_test


def tokenize_function(data):
     # carrregar tokenizador BERTimbau 
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased') 
    return tokenizer(data['text'], truncation=True, padding='max_length', max_length=128)

def preprocessing(dataframe):

    # transformar dataframe pandas ---> dataset object 
    dataset = Dataset.from_pandas(dataframe)

    # aplicando tokenizacao
    tokenized_data = dataset.map(tokenize_function, batched=True)

    return tokenized_data




def bertimbau_model(training_config, train_data, test_data):

    # carregar modelo (base)
    global model
    model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-large-portuguese-cased", num_labels=2) #  BERTimbau-base 

    # metrica 
    global metric
    metric = evaluate.load("accuracy")

    # métricas a serem monitoradas
    metrics = [
        {'name': 'accuracy', 'function': accuracy_score},
        {'name': 'f1', 'function': lambda p, l: f1_score(p, l, average='weighted')}
    ]

    # argumentos de treinamento (hiperparametros)
    training_args = TrainingArguments(training_config)

    # definir trainer 
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=lambda eval_pred: {metric['name']: metric['function'](eval_pred.predictions.argmax(axis=-1), eval_pred.label_ids) for metric in metrics}
)

    return trainer



def report_num_parameters():
    num_params = model.num_parameters()
    print(f'O modelo tem {num_params} parâmetros.')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(): 

    print("GPU: ", torch.cuda.is_available())

    # carregar dados 
    df_train, df_test = data_processing(path_train="assin2-train.xml", path_test="assin2-test.xml")

    # tokenization 
    train_data = preprocessing(df_train)
    test_data = preprocessing(df_test)

    #print(df_train.head())

    # Define os argumentos do treinamento
    training_args = TrainingArguments(
        output_dir='./results_bert_large',
        num_train_epochs=10,
        per_device_train_batch_size=16, # default 16 
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
        save_total_limit=3,
        save_steps=500,
        metric_for_best_model='f1'
    )


    # BERT 
    model = bertimbau_model(training_config=training_args.output_dir, train_data=train_data, test_data=test_data)
    report_num_parameters()
    
    # train 
    model.train()

    # Imprime as métricas obtidas no final do treinamento
    metrics = model.evaluate()
    print("\n\n\n metricas")
    print(metrics)

    # Salvar o modelo
    model.save_model('bertimbau-large-fine-tune')



if __name__ == "__main__":
    main()
