from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yaml
import hashlib
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()

PROXY_URL = "http://www.gots.ru:9041"
FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

BASE_URL = f"{PROXY_URL}/v1"
API_KEY = f"{FOLDER_ID}@{YANDEX_API_KEY}"

PROMPT_QUESTION = ''
PROMPT_ANSWER = 'Ниже приведены вопрос, правильный ответ и полученный ответ. Если полученный ответ корректный, напиши "True", иначе "False". Пиши только эти два слова, больше ничего не пиши.\n\n'

models = ["yandexgpt/latest", "llama/latest"]

def init_model(model, temperature=0.4):
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        model=model,
    )

def get_data_files(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.yaml')]

def read_yaml_file(yaml_file):
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file) or {}
    return {}

def save_yaml_file(filename, data):
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def make_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def load_processed_hashes(model):
    return set(read_yaml_file(f'processed_hashes_{model.replace("/", "_")}.yaml'))

def save_processed_hashes(model, hashes):
    save_yaml_file(f'processed_hashes_{model.replace("/", "_")}.yaml', list(hashes))

def load_results_for_model(model):
    return read_yaml_file(f'results_{model.replace("/", "_")}.yaml')

def save_results_for_model(model, results):
    save_yaml_file(f'results_{model.replace("/", "_")}.yaml', results)

def load_stats_for_model(model):
    return read_yaml_file(f'stats_{model.replace("/", "_")}.yaml')

def save_stats_for_model(model, stats):
    save_yaml_file(f'stats_{model.replace("/", "_")}.yaml', stats)

def work_with_question(qa_dict, question, model):
    llm = init_model(model)
    answer = llm.invoke(question)
    check_question = f"{PROMPT_ANSWER} Вопрос: {question}.\n Правильный ответ: {qa_dict['answer']}.\n Полученный ответ: {answer.content}"
    check_answer = llm.invoke(check_question)

    return {
        'question': qa_dict['question'],
        'answer': answer.content,
        'right answer': check_answer.content,
        'model': model
    }

def plot_results(model_stats):
    fig, axes = plt.subplots(1, len(models), figsize=(15, 6))
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        results = model_stats.get(model, {})
        counter = Counter(results)
        labels = list(counter.keys())
        sizes = list(counter.values())

        axes[idx].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
        axes[idx].set_title(f"Распределение ответов {model}")
        axes[idx].axis('equal')

    plt.tight_layout()
    plt.show()

def main():
    model_stats = {model: load_stats_for_model(model) for model in models}

    for model in models:
        processed_hashes = load_processed_hashes(model)
        results_for_model = load_results_for_model(model)

        for yaml_file in get_data_files('./data'):
            for qa_list in read_yaml_file(yaml_file):
                for qa_dict in qa_list:
                    question = f"{PROMPT_QUESTION} {qa_dict['question']}"
                    hsh = make_hash(question)

                    if hsh in processed_hashes:
                        continue

                    result = work_with_question(qa_dict, question, model)
                    results_for_model.append(result)
                    model_stats[model].append(result['right answer'])

                    processed_hashes.add(hsh)

                    print(f"Модель: {result['model']}\nВопрос: {qa_dict['question']}\nПравильный ответ: {qa_dict['answer']}\nОтвет модели: {result['answer']}\nРезультат проверки: {result['right answer']}\n\n")

        save_results_for_model(model, results_for_model)
        save_processed_hashes(model, processed_hashes)
        save_stats_for_model(model, model_stats[model])

    plot_results(model_stats)

if __name__ == '__main__':
    main()
