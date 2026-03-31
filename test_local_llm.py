import json
import torch
import warnings
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

from server.sdsmp_environment import SdsmpEnvironment, TASKS
from server.baseline import SYSTEM_PROMPT, _parse_action

def run_local_llm():
    print("Initializing fully local LLM inference...")
    print("Downloading 'Qwen/Qwen2.5-0.5B-Instruct' (Alibaba's 0.5 Billion Parameter Model)")
    print("This will take 1-2 minutes depending on connection, but keeps all data 100% offline.")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware utilized: {device.upper()}")
    
    # Load fast inference tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n--- Testing '{task_id}' task with Local AI Model ---")
        env = SdsmpEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        task = TASKS[task_id]
        
        system_msg = SYSTEM_PROMPT.format(task_description=task["description"])
        
        for step in range(25):
            if obs.done:
                break
                
            obs_dict = obs.model_dump()
            user_msg = (
                f"Pending Jobs:\n{json.dumps(obs_dict['pending_jobs'])}\n\n"
                f"VMs:\n{json.dumps(obs_dict['smp_vms'])}\n\n"
                f"Metrics: Cost=${obs_dict['current_cost']:.2f}, Load Balancing={obs_dict['load_balancing_rate']:.2f}, QoS={obs_dict['qos_satisfaction_rate']:.2f}\n\n"
                f"Pick next action. Respond JSON."
            )
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Format using standard chat templates
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            
            # Predict
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # Environment acts on prediction
            action = _parse_action(response_text)
            obs = env.step(action)
            
            # Summarize result
            log_snippet = obs.execution_log.split("(")[0].strip() if "(" in obs.execution_log else obs.execution_log
            print(f"[{step+1:02d}] AI Action: {action.get('command')} -> {log_snippet[:40]} | QoS:{obs.qos_satisfaction_rate*100:3.0f}% Cost:${obs.current_cost:.3f}")
            
        grade = env.get_grade()
        print(f">>> Final Grade for {task_id}: {grade:.4f} <<<")

if __name__ == "__main__":
    run_local_llm()
