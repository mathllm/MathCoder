import re
import json
import os
import sys
from io import StringIO
import threading

from tqdm import tqdm
from multiprocessing import Pool,RLock
from huggingface_hub import InferenceClient
from fire import Fire
from jupyter_client.manager import start_new_kernel
import zmq
import time
from argparse import ArgumentParser



def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def truncate_decimals(s, decimal_places=6, role='unknown'):
    logs = []  # 用于记录修改的日志

    # 定义一个辅助函数，将匹配到的小数截短到指定位数或保持不变
    def repl(match):
        value = match.group(0)
        integer_part, decimal_part = value.split('.')
        if len(decimal_part) <= decimal_places:
            return value
        
        format_string = '{:.' + str(decimal_places) + 'f}'
        truncated_value = format_string.format(float(value))
        
        # 记录修改
        logs.append(f'Modified {role}: {value} => {truncated_value}')
        
        return truncated_value

    # 使用正则表达式找到所有的小数并用上面的函数替换
    modified_s = re.sub(r"\d+\.\d+", repl, s)

    # 打印日志
    for log in logs:
        print(log)
    
    return modified_s


class JupyterNotebookKernel(object):

    lock = RLock()

    def __init__(self, retries=5, delay=3):
        JupyterNotebookKernel.lock.acquire()
        for _ in range(retries):
            try:
                self.manager, self.client = start_new_kernel(kernel_name='python')
                break
            except zmq.ZMQError as e:
                if 'Address already in use' in str(e) and _ < retries - 1:  # check if the error is because the address is in use
                    print(f'Address already in use. Retrying in {delay} seconds...')
                    time.sleep(delay)
                else:
                    raise
        else:
            raise Exception('Failed to start kernel after multiple retries.')
        JupyterNotebookKernel.lock.release()
                
    def __del__(self):
        try:
            if hasattr(self, 'manager'):
                self.manager.shutdown_kernel()
            if hasattr(self, 'client'):
                self.client.stop_channels()
        except:
            pass
    
    def shutdown(self):
        if self.manager:
            self.manager.shutdown_kernel()
            self.client.stop_channels()


    def handle_iopub_msg(self):
        result = ''

        while msg := self.client.get_iopub_msg(timeout=10):
            
            if msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                break

            if msg['msg_type'] == 'stream':
                result += msg['content']['text']
            
            if msg['msg_type'] == 'execute_result':
                result += msg['content']['data']['text/plain']
            
            if msg['msg_type'] == 'error':
                if isinstance(msg['content']['traceback'], list):
                    msg['content']['traceback'] = ' '.join(msg['content']['traceback'])

                error = re.sub(
                    '\x1B\\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]',
                    '',
                    msg['content']['traceback'],
                )

                result += error
        
        if len(result) == 0:
            result = '<empty_execution>'

        return result.strip()

    def run_code(self, code):
        try:
            self.client.execute(code, allow_stdin=False, reply=True, timeout=6)
            return self.handle_iopub_msg()
        except zmq.ZMQError as e:
            if 'Address already in use' in str(e):
                print('Address already in use. Restarting kernel...')
                raise
        except Exception as e:
            return f'{"-"*75} {str(e)}{" "*32}Traceback (most recent call last) '

    def monitor_errors(self):
        old_stderr = sys.stderr
        sys.stderr = captured_stderr = StringIO()
        while True:
            # Check the error stream every second (adjust as needed)
            time.sleep(1)
            error_output = captured_stderr.getvalue()
            if "[IPKernelApp] WARNING | Parent appears to have exited, shutting down." in error_output:
                # Do your restart logic here
                os.execl(sys.executable, sys.executable, *sys.argv)

    def start_monitoring(self):
        # This starts the error monitor in a separate thread
        error_monitor_thread = threading.Thread(target=self.monitor_errors)
        error_monitor_thread.daemon = True  # So the thread will exit when the main program exits
        error_monitor_thread.start()


class API:

    def __init__(self, ip='101.230.144.194', port='8000'):
        self.client = InferenceClient(model=f'http://{ip}:{port}')

    def get_result(self, inputs, parameters=None):

        local_parameters = dict(max_new_tokens=512, details=True, decoder_input_details=True)

        if parameters is not None:
            local_parameters.update(parameters)
        
        try:
            result = self.client.text_generation(prompt=inputs, **local_parameters)

            text = result.generated_text
            # print(type(result.details))
            if result.details.tokens[0].special and not text.startswith(result.details.tokens[0].text.strip()):
                text = result.details.tokens[0].text.strip() + text

            if result.details.tokens[-1].special and not text.endswith(result.details.tokens[-1].text.strip()):
                text = text + result.details.tokens[-1].text.strip()

            return text
        except:
            import traceback
            traceback.print_exc()
            print(inputs) 
            return None
        

def code_generation(query):
    raw_prompt = f'<|system|><|text|>{system}<|endofblock|><|endofmessage|><|user|><|text|>{query}<|endofblock|><|endofmessage|><|assistant|>'
    prompt = raw_prompt

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': query}
    ]

    jupyter = JupyterNotebookKernel()
    jupyter.start_monitoring()

    parameters=dict(
        do_sample=False,
        max_new_tokens=512,
        stop_sequences=['<|endofmessage|>', '<|endofblock|>'], 
        truncate=1536,
        details=True, 
        decoder_input_details=True
    )
    code = ''

    for _ in range(16):            
        result = api.get_result(prompt, parameters=parameters)

        if result is None:
            print('WARNING: exceed_max_length/return_first_c!')
            break
        result = truncate_decimals(result,role=result.split('|>')[0]+'|>') 

        prompt += result
        
        if result.startswith('<|code|>'):
            code = result.replace('<|code|>', '').replace('<|endofblock|>', '')
            messages.append({'type': 'code', 'content': code})
            
            execution = jupyter.run_code(code)

            prompt += f"<|execution|>{execution}<|endofblock|>"
            messages.append({'type': 'execution', 'content': execution})
        elif not result.endswith('<|endofmessage|>'):
            messages.append({'type': 'text', 'content': result.replace('<|text|>', '').replace('<|endofblock|>', '')})
        else:
            break
    
    jupyter.shutdown()
    return messages

def process_full(data):
    global cot_prompt
    
    if 'messages' in data:
        query = data['messages'][1]['content'][0]['content']
    else:
        query = data['question']

    debug_result = code_generation(query+cot_prompt)

    data['debug_result'] = debug_result

    return data


def main(dataset, pnum, outdir, ip, port='8000', type='test', syspt=False, cot=False, slice=None):
    global api, system, cot_prompt
    if ':' in ip:
        port = ip.split(':')[-1]
        ip = ip.split(':')[0]
    api = API(ip=ip, port=port) 

    if syspt:
        system = 'Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}".'
    else:
        system = ""

    if cot:
        cot_prompt = "Let's think step by step."
    else:
        cot_prompt = ""
    
 
    
    if '/' not in dataset:
        if slice:
            input_path = f'./data/{dataset}_{slice}_{type}_post.jsonl'
        else:
            input_path = f'./data/{dataset}_{type}_post.jsonl'
    else:
        input_path = dataset
        dataset = dataset.split('/')[-1].split('.')[0]

    outdir = outdir.split('outs/')[-1]
    output_path = os.path.join('./outs/', outdir, dataset)
  
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if slice:
        output_path = os.path.join(output_path, f'{dataset}_{slice}_{type}_result.jsonl')
    else:
        output_path = os.path.join(output_path, f'{dataset}_{type}_result.jsonl')
    
    
    print("\n======================================================== INFO ========================================================")
    print('system:', f'"{system}"')
    print('CoT   :', f'"{cot_prompt}"')
    cot_prompt = '\n' + cot_prompt
    print('input :', input_path)
    print('output:', output_path)
    print('====================================================== END INFO ======================================================\n')

            

    try:
        all = load_jsonl(output_path)
    except FileNotFoundError:
        all = []

    BEGIN = len(all)

    OVER_WRITE = True
    humaneval = load_jsonl(input_path)
    END = len(humaneval)
    outs = []

    counter = BEGIN
    pool = Pool(pnum)
    try:
        results = pool.imap(process_full, humaneval[BEGIN:END])
        for d in tqdm(results, total=len(humaneval[BEGIN:END])):
            d['completion'] = d['debug_result'][-1]['content']
            outs.append(d)
            all.append(d)
            counter += 1
            if counter % 10 == 0 or counter == END:
                if counter <= 10 and OVER_WRITE:
                    save_jsonl(outs, output_path,mode='w', add_timestamp=False, verbose=False)
                else:
                    save_jsonl(outs, output_path,mode='a', add_timestamp=False, verbose=False)
                outs = []
    except Exception as e:
        print(f'<|{str(e)}|>')
        pool.terminate()  # 立即终止所有子进程
        print('[Restarting...]')
        os.execl(sys.executable, sys.executable, *sys.argv)


    save_jsonl(all, output_path, add_timestamp=True)
    print(f"Saved to {output_path}")
    save_jsonl([], output_path, add_timestamp=False)




if __name__ == '__main__':
    Fire(main)
