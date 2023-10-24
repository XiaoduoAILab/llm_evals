import pandas as pd


def get_prompt(task_data):
    result = ''
    result += '问题：' + task_data['question'] + '\n'
    result += '选项：\n'
    result += '(A) ' + task_data['A'] + '\n'
    result += '(B) ' + task_data['B'] + '\n'
    result += '(C) ' + task_data['C'] + '\n'
    result += '(D) ' + task_data['D'] + '\n'
    return result


def get_task_prompt(task, num_shot=2):
    filename = 'data/dev/{}_dev.csv'.format(task)
    df = pd.read_csv(filename)
    # print(df.to_string())

    # num_rows = len(df.index)
    # print('num_rows: ' + str(num_rows))
    task_prompt = ''
    for i in range(num_shot):
        task_prompt += get_prompt(dict(question=df['question'].values[i],
                                       A=df['A'].values[i],
                                       B=df['B'].values[i],
                                       C=df['C'].values[i],
                                       D=df['D'].values[i],
                                       answer=df['answer'].values[i]))
        task_prompt += '答案：(' + df['answer'].values[i] + ')\n\n'
    return task_prompt


def get_task_data(task):
    filename = 'data/val/{}_val.csv'.format(task)
    df = pd.read_csv(filename)

    num_rows = len(df.index)
    data = []
    for i in range(num_rows):
        data.append(dict(question=df['question'].values[i],
                         A=df['A'].values[i],
                         B=df['B'].values[i],
                         C=df['C'].values[i],
                         D=df['D'].values[i],
                         answer=df['answer'].values[i]))

    return data


if __name__ == "__main__":
    task = 'legal_professional'
    prompt = get_task_prompt(task)
    # print(prompt)

    task_data = get_task_data(task)
    # print(task_data)

    prompt += get_prompt(task_data[0])
    print(prompt)
