import sys
import time
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
from google.colab import drive
import wandb

drive.mount('/content/gdrive')
OUTPUT_DIR = '/content/gdrive/MyDrive/anlp_ex1'
MODEL_NAMES = ["bert-base-uncased", "roberta-base",
               "google/electra-base-generator"]
SA_DATASET = "sst2"
RESULTS_FILE_NAME = "res.txt"
PREDICTIONS_FILE_NAME = "prediction.txt"


class SingleFineTuneExperiment:
    def __init__(self, model, tokenizer, metric, seed, train_dataset,
                 eval_dataset, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.metric = metric
        self.seed = seed
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.test_dataset = test_dataset

    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = self.metric.compute(predictions=preds, references=p.label_ids)
        return result

    def train(self):
        training_args = TrainingArguments(output_dir=OUTPUT_DIR,
                                          evaluation_strategy="epoch",
                                          seed=self.seed,
                                          save_total_limit=1)
        self.trainer = Trainer(model=self.model,
                               args=training_args,
                               train_dataset=self.train_dataset,
                               eval_dataset=self.eval_dataset,
                               compute_metrics=self.compute_metrics,
                               data_collator=DataCollatorWithPadding(self.tokenizer,
                                                                     padding=True))
        train_start = time.time()
        self.trainer.train()
        train_end = time.time()
        return train_end - train_start

    def evaluate(self):
        return self.trainer.evaluate(eval_dataset=self.eval_dataset)["eval_accuracy"]

    def predict(self):
        self.trainer.args.set_testing(batch_size=1)
        self.trainer.data_collator = None
        self.trainer.model.eval()
        pred_start = time.time()
        preds, _, _ = self.trainer.predict(self.test_dataset)
        pred_end = time.time()
        pred_time = pred_end - pred_start
        return preds, pred_time


class FineTuneExperimentManager:
    def __init__(self, model_name, dataset_name, num_train_samples,
                 num_valid_samples, num_predict_samples):
        self.raw_dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.metric = evaluate.load("accuracy")
        self.tokenize_dataset()
        self.init_datasets(num_train_samples, num_valid_samples, num_predict_samples)

    def tokenize_dataset(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"], truncation=True)

        self.tokenized_dataset = self.raw_dataset.map(tokenize_function,
                                                      batched=True)

    def init_datasets(self, num_train_samples, num_valid_samples, num_predict_samples):
        self.train_dataset = self.tokenized_dataset["train"]
        if num_train_samples > 0:
            self.train_dataset = self.train_dataset.select(range(num_train_samples))

        self.eval_dataset = self.tokenized_dataset["validation"]
        if num_valid_samples > 0:
            self.eval_dataset = self.eval_dataset.select(range(num_valid_samples))

        self.test_dataset = self.tokenized_dataset["test"].remove_columns(['label'])
        self.test_dataset.set_format("pt", output_all_columns=True)
        if num_predict_samples > 0:
            self.test_dataset = self.test_dataset.select(range(num_predict_samples))

    def get_results_dict(self):
        return {"mean": round(np.mean(self.accuracy), 3),
                "std": round(np.std(self.accuracy), 3),
                "best_accuracy": round(max(self.accuracy), 3),
                "best_exp": self.exps[np.argmax(self.accuracy)],
                "train_time": self.total_train_time}

    def run_exps(self, num_seeds):
        self.accuracy, self.exps, self.total_train_time = [], [], 0
        for seed in range(num_seeds):
            wandb.init(project='anlp_ex1_19.05', name=f'{self.model_name}_seed_{seed}')
            cur_exp = SingleFineTuneExperiment(self.model, self.tokenizer,
                                               self.metric, seed,
                                               self.train_dataset, self.eval_dataset,
                                               self.test_dataset)
            self.total_train_time += cur_exp.train()
            self.accuracy.append(cur_exp.evaluate())
            self.exps.append(cur_exp)

        wandb.finish()
        exp_results = self.get_results_dict()
        return exp_results


def find_best_exp_across_models(exp_results):
    highest_acc = 0
    best_experiment = None
    total_train_time = 0
    for model_name in exp_results:
        total_train_time += exp_results[model_name]["train_time"]
        if exp_results[model_name]["best_accuracy"] > highest_acc:
            best_experiment = exp_results[model_name]["best_exp"]
    return best_experiment, total_train_time


def write_out_results(exp_results, train_time, pred_time, test_dataset, pred_results):
    with open(f'{OUTPUT_DIR}/{RESULTS_FILE_NAME}', 'w+') as f:
        for model_name in exp_results:
            f.write(f"{model_name},{exp_results[model_name]['mean']} +- {exp_results[model_name]['std']}\n")
        f.write("----\n")
        f.write(f"train time,{round(train_time, 3)}\n")
        f.write(f"predict time,{round(pred_time, 3)}")

    with open(f'{OUTPUT_DIR}/{PREDICTIONS_FILE_NAME}', 'w+') as f:
        for ind, result in enumerate(list(pred_results)):
            f.write(f"{test_dataset[ind]['sentence']}###{np.argmax(result)}\n")


def main(num_seeds, num_train_samples, num_valid_samples, num_predict_samples):
    exp_results_across_models = {}
    for model_name in MODEL_NAMES:
        exp_manager = FineTuneExperimentManager(model_name,
                                                SA_DATASET,
                                                num_train_samples,
                                                num_valid_samples,
                                                num_predict_samples)
        exp_results_across_models[model_name] = exp_manager.run_exps(num_seeds)
    best_experiment, total_train_time = find_best_exp_across_models(exp_results_across_models)
    pred_results, pred_time = best_experiment.predict()
    write_out_results(exp_results_across_models, total_train_time, pred_time,
                      best_experiment.test_dataset, pred_results)


if __name__ == "__main__":
    assert len(sys.argv) == 5
    main(num_seeds=int(sys.argv[1]), num_train_samples=int(sys.argv[2]),
         num_valid_samples=int(sys.argv[3]), num_predict_samples=int(sys.argv[4]))
    wandb.login()