from peft import LoraConfig
class lora:
    def __init__(self, r:int, lora_alpha:int, target_modules:list, lora_dropout: float, bias:str, task_type:str) ->None:
        self.lora_config = LoraConfig(
            r, 
            lora_alpha, 
            target_modules, 
            lora_dropout, 
            bias, 
            task_type
        )
    
    def __repr__(self):
        return self.lora_config