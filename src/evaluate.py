import fire
import re
from tqdm import tqdm
import time
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal



def eval(input_path, num=-1, save_wrong=False):
    if ',' in input_path:
        eval_multifile(input_path, num)
        return
    wrong = []
    correct = []
    clen, wlen = {}, {}
    cerror, werror = {}, {}
    answers = load_jsonl(input_path)
    if num > 0:
        answers = answers[:num]
    count_correct = 0
    gt_sol = ''
    for line in tqdm(answers, desc='eval'):
        if 'ground_truth' in line.keys():
            gt_answer = str(line['ground_truth']['answer'])
            if gt_answer in 'ABCD' and 'MMLU' in input_path:
                gt_sol = line['ground_truth']['solution']
        elif 'extra' in line.keys():
            gt_answer = str(line['extra']['answer'])
        else:
            gt_answer = line['answer']
            

        # model_answer = find_math_answer(line['completion'])
        model_answer = line['completion']

        for choice in ['A', 'B', 'C', 'D']:
            if 'oxed{'+f'{choice} ' in model_answer or  'oxed{'+f'{choice}. ' in model_answer or  'oxed{'+f'{choice}' + '}' in model_answer or f'Answer: {choice} ' in model_answer:
                model_answer = choice
        ans_len = len(line['debug_result'])-3
        e_num = ''.join([content['content'] for  content in line['debug_result']]).count('Traceback (most recent call last)')
        # gt_answer = gt_answer.split('{')[0]
        # model_answer = model_answer.split('{')[0]
        gt_answer = find_math_answer(gt_answer)
        model_answer = find_math_answer(model_answer)
        if 'GSM8K' in input_path:
            model_answer = re.sub(r'[a-zA-Z]', '', model_answer)
        if is_equal(gt_answer, model_answer):
            # print(gt_answer, model_answer, gt_sol)
            correct.append(line)
            count_correct += 1
            if ans_len not in clen.keys():
                clen[ans_len] = 0
            clen[ans_len] += 1
            if e_num not in cerror.keys():
                cerror[e_num] = 0
            cerror[e_num] += 1
        else:
            # print(gt_answer, model_answer, gt_sol)
            line['model_answer'] = model_answer
            wrong.append(line)
            if ans_len not in wlen.keys():
                wlen[ans_len] = 0
            wlen[ans_len] += 1
            if e_num not in werror.keys():
                werror[e_num] = 0
            werror[e_num] += 1
    acc = count_correct / len(answers)
    print(f'\nCorrect error num distribution:', sorted(cerror.items(), key=lambda x: x[0]))
    print(f'Wrong error num distribution  :', sorted(werror.items(), key=lambda x: x[0]))
    print(f'\nCorrect len distribution:', sorted(clen.items(), key=lambda x: x[0]))
    print(f'Wrong len distribution  :', sorted(wlen.items(), key=lambda x: x[0]))
    print(f'\nError Freq of correct: {sum([k*v for k, v in cerror.items()])/max(1,sum(cerror.values()))}')
    print(f'Error Freq of wrong  : {sum([k*v for k, v in werror.items()])/max(1,sum(werror.values()))}')
    print(f'\nAvg. lenth of correct: {sum([k*v for k, v in clen.items()])/max(1,sum(clen.values()))}')
    print(f'Avg. lenth of wrong  : {sum([k*v for k, v in wlen.items()])/max(1,sum(wlen.values()))}')
    print(f'\ncount_correct: {count_correct}')
    print(f'Accuracy: {round(acc*100, 2)}%\n\n')
    if save_wrong:
        wrong = sorted(wrong, key=lambda x: len(x['debug_result']))
        save_jsonl(input_path.replace('.jsonl', '-wrong.jsonl'), wrong, t_stamp=False)
        save_jsonl(input_path.replace('.jsonl', '-correct.jsonl'), correct, t_stamp=False)



def math_level_subject_acc(input_path):
    subject_level = load_jsonl('./data/MATH_subject_level.jsonl')[0]
    answers = load_jsonl(input_path)
    all_correct = 0
    print(' '*33+'Level1\tLevel2\tLevel3\tLevel4\tLevel5\tOverall', end='')
    for subject in ['algebra', 'prealgebra', 'number_theory', 'counting_and_probability', 'precalculus', 'intermediate_algebra', 'geometry']:
    # for subject in ['geometry']:
        print('\n%-32s: '%subject, end='')
        lc, la = 0, 0
        for level in range(1,6):
            count_correct = 0
            count_all = 0
            for line in answers:
                if f'/{subject}/' in line['id'] and (subject_level[line['id']] == level or 0):
                    count_all += 1
                    if 'ground_truth' in line.keys():
                        gt_answer = str(line['ground_truth']['answer'])
                    else:
                        gt_answer = line['answer']
                    model_answer = line['completion']
                    if is_equal(gt_answer, model_answer):
                        count_correct += 1
            all_correct += count_correct
            print(f'{round(count_correct/count_all, 2)}', end='\t')
            lc += count_correct
            la += count_all
        print(f'{round(lc/la, 2)}', end='\t')
    assert(len(answers) == 5000)
    print(f'\noverall: {all_correct/len(answers)}')
            


if __name__ == '__main__':
    # math_level_subject_acc('./outs/MathCoder-L-70b/MATH/MATH_test_result-20230920-0005.jsonl')
    fire.Fire(eval)