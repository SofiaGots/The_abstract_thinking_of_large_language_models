from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yaml
import hashlib

load_dotenv()

PROXY_URL = "http://www.gots.ru:9041"
FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

BASE_URL = f"{PROXY_URL}/v1"
API_KEY = f"{FOLDER_ID}@{YANDEX_API_KEY}"

PROMPT_QUESTION = ''
PROMPT_ANSWER = 'Ниже приведены вопрос, правильный ответ и полученный ответ. Если полученный ответ корректный, напиши "True", иначе "False". Пиши только эти два слова, больше ничего не пиши.\n\n'

FILE_NAME = 'tests.yaml'

def init_model(model, temperature=0.4):
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        # model="yandexgpt/latest",
        # model="llama/latest",
        model=model,
    )

def get_data_files(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.yaml')]

def read_yaml_file_1(yaml_file):
    with open(yaml_file, 'r') as f:
        yield yaml.full_load(f)

def work_with_question(qa_dict, question):
    llm = init_model('llama/latest')
    answer = llm.invoke(question)
    check_question = f"{PROMPT_ANSWER} Вопрос: {question}.\n Правильный ответ: {qa_dict['answer']}.\n Полученный ответ: {answer.content}"
    check_answer = llm.invoke(check_question)

    return {'question': qa_dict['question'], 'answer': answer.content, 'right answer': check_answer.content}


def save_yaml_file(filename, data):
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def read_yaml_file(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r') as file:
        return yaml.safe_load(file) or {}

def make_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def main():
    data = read_yaml_file(FILE_NAME)
    for yaml_file in get_data_files('./data'):
        for qa_list in read_yaml_file_1(yaml_file):
            for qa_dict in qa_list:
                question = f"{PROMPT_QUESTION} {qa_dict['question']}"
                hsh = make_hash(question)
                if hsh not in data:
                    data[hsh] = work_with_question(qa_dict, question)
                    save_yaml_file(FILE_NAME, data)
                
                print(
                    f"Вопрос: {qa_dict['question']}\n",
                    f"Правильный ответ: {qa_dict['answer']}\n",
                    f"Ответ модели: {data[hsh]['answer']}\n",
                    f"Результат проверки: {data[hsh]['right answer']}\n\n"
                        )
        

if __name__ == '__main__':
    main()