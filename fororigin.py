import transformers as trs
import torch as th
import peft as p
from torch.utils.data import Dataset as d
import json as j
class transformer:
    def __init__(self, data_path, model_name, output_dir):
        self.data_path = data_path
        self.dataset = None
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq  = 1
        self.batch_size = 1
        self.epoch = 1
        self.lr = 1e-3
        self.train_data = None
        self.test_data = None
        self.model = None
        self.tokenizer = None
        self.training_args = None
        self.trainer = None
    def config(self, max_seq, batch_size, epoch, lr):
        self.max_seq  = max_seq
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = j.load(f)
            
    def data_split(self):
        x = int(len(self.dataset)*0.80)
        self.train_data = self.dataset[:x]
        self.test_data = self.dataset[x:]
    
    def load_tokens(self):
        self.tokenizer = trs.AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_model(self):
        self.model = trs.AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=th.float16)
    
    def merge_model(self, lora_config):
        if lora_config is None:
            raise Exception("Lora Config is not generated. Run lora_config = lora(param)")
        else:
            self.model = p.get_peft_model(self.model, lora_config)
    
    def training_args(self, gradient_accumulation, log_dir, log_steps, save_steps, save_t_limit, load_best_model, report=None):
        self.training_args = trs.TrainingArguments(
            output_dir= self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=self.epoch,
            learning_rate=self.lr,
            logging_dir=log_dir,
            logging_steps=log_steps,
            save_steps=save_steps,
            save_total_limit=save_t_limit,
            load_best_model_at_end=load_best_model,
            report_to=report, # Disable reporting to services like W&B
        )

    def init_trainer(self):
        self.trainer = trs.Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset=self.training_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
                    
        )
    
    def train(self):
        print("Starting training...")
        self.trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Training complete. Model saved to {self.output_dir}")