from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

PROXY_URL = "http://www.gots.ru:9041"
FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

BASE_URL = f"{PROXY_URL}/v1"
API_KEY = f"{FOLDER_ID}@{YANDEX_API_KEY}"

PROMPT_QUESTION = ''
PROMPT_ANSWER = 'Ниже приведены вопрос, правильный ответ и полученный ответ. Если полученный ответ корректный, напиши "True", иначе "False". Пиши только эти два слова, больше ничего не пиши.\n\n' 

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

def read_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        yield yaml.full_load(f)

def ask_question():
    pass


def main():
    llm = init_model('yandexgpt/latest')
    for yaml_file in get_data_files('./data'):
        for qa_list in read_yaml_file(yaml_file):
            for qa_dict in qa_list:
                question = f"{PROMPT_QUESTION} {qa_dict['question']}"
                answer = llm.invoke(question)
                check_question = f"{PROMPT_ANSWER} Вопрос: {question}.\n Правильный ответ: {qa_dict['answer']}.\n Полученный ответ: {answer.content}"
                check_answer = llm.invoke(check_question)
                print(
                    f"Вопрос: {qa_dict['question']}\n",
                    f"Правильный ответ: {qa_dict['answer']}\n",
                    f"Ответ модели: {answer.content}\n",
                    f"Результат проверки: {check_answer.content}\n\n"
                      )
        

if __name__ == '__main__':
    main()