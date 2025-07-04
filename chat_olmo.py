import torch
import pathlib
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse 

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    return model, tokenizer

def generate_reply(model, tokenizer, user_input,**kwargs):
    if kwargs['sft']:
        messages =['<|user|>\n' + user_input +'<|assistant|>\n']
    else:
        messages=[user_input]
    model_inputs = tokenizer(messages, return_tensors="pt").to("cuda")
    input_len = model_inputs['input_ids'].shape[1]
    generated_ids = model.generate(**model_inputs,do_sample=kwargs['do_sample'],
                                   max_length=kwargs['max_length'],num_beams=kwargs['num_beams'],
                                   temperature=kwargs['temperature'])
    if kwargs['sft']:
        output_ids = generated_ids[..., input_len:]
    else:
        output_ids = generated_ids
    outputs=tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return outputs 

def interactive_loop(model, tokenizer,**kwargs):
    print("进入OLMo2 chat模式")
    while True:
        user_input=input("\n 🧑•💻你:")
        if user_input.lower() in {'q','quit','exit'}:
            print("👋再见")
            break
        reply = generate_reply(model,tokenizer,user_input,**kwargs)
        print(f"\n 🤖OLMo2: {reply}") 
def main():
    parser = argparse.ArgumentParser(description='Training')
    # system and basic setting
    parser.add_argument('--path',type=str,required=True)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--do_sample',type=bool,default=False)
    parser.add_argument('--sft',type=int, default=0, help='模型是否经过sft')
    args = parser.parse_args()
    model,tokenizer=load_model(args.path)
    args_dict = vars(args)
    interactive_loop(model, tokenizer,**args_dict)

if __name__=="__main__":
    print("🚀正在加载模型:")
    main()