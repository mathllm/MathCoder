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
    global api, prompt
    # sys = f'<|system|><|text|>{system}<|endofblock|><|endofmessage|>'
    sys = ''
    user = f'<s><|system|><|text|><|endofblock|><|endofmessage|><|user|><|text|>{query}<|endofblock|><|endofmessage|>'
    prompt = prompt + sys + user + '<|assistant|>'

    messages = [
        {'role': 'user', 'content': query}
    ]

    jupyter = JupyterNotebookKernel()
    jupyter.start_monitoring()

    parameters=dict(
        do_sample=False,
        max_new_tokens=1024,
        stop_sequences=['<|endofmessage|>', '<|endofblock|>', '</s>'], 
        truncate=3072,
        details=True, 
        decoder_input_details=True
    )
    code = ''
    for _ in range(16):
        result = api.get_result(prompt, parameters=parameters)

        if result is None:
            messages.append({'role': 'exceed_max_length/return_first_code', 'content': code})
            print('WARNING: exceed_max_length/return_first_c!')
            break
        result = truncate_decimals(result,role=result.split('|>')[0]+'|>').replace('<|im_start|>', '').replace('<|im_end|>', '')

        prompt += result
        print(result)
        

        
        if result.startswith('<|code|>') or ('# ' in result and ' = ' in result):
            code = result.replace('<|code|>', '').replace('<|text|>', '').replace('<|endofblock|>', '')
            messages.append({'role': 'code', 'content': code})
            
            execution = f"<|execution|>{jupyter.run_code(code)}<|endofblock|>"

            # print(f'\n<|code|>\n{code}<|endofblock|>\n{execution}\n')
            print(f'\n{execution}\n')
            prompt += execution
            messages.append({'role': 'execution', 'content': execution})
        elif not result.endswith('<|endofmessage|>') and not result.endswith('</s>'):
            messages.append({'role': 'text', 'content': result.replace('<|text|>', '').replace('<|endofblock|>', '')})
        else:
            break
    
    jupyter.shutdown()
    return prompt



def main(ip='', port='8000', csvon=False, memory_on=False):
    global api, prompt
    prompt = ''
    
    api = API(ip=ip, port=port) 
    if csvon:
        csv = 'To solve the problem using code interpreter step by step, even in every sub-step. And following your answer, please "verify" it using code interpreter by yourself.\n'
    else:
        csv = ""


    print("************************** INFO **************************")
    print('CSV:', f"'{csv}'")
    print('menory:', 'on (on/off)' if memory_on else 'off (on/off)' )
    print("************************ END INFO ************************")

    
    
    while True:
        if not memory_on:
            prompt = ''
        query = input('[Q]: ')
        if query == '<|exit|>':
            break
        if query == '<|clear|>':
            prompt = ''
            continue

        print('[A]: ', end='', flush=True)

        answer = code_generation(csv+query)
        answer = answer.split('<|assistant|>')[1]
        answer = answer.replace('<|text|>', '\n').replace('<|endofblock|>', '\n').replace('<|endofmessage|>', '\n')
        answer = answer.replace('<|code|>', '\n<|code|>\n').replace('<|execution|>', '<|execution|>\n')

        print('')
        # print('\n[A]: ', answer)

    


if __name__ == '__main__':
    Fire(main)
