# pip install langcain langchain-openai

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

PROXY_URL = "http://www.gots.ru:9041"
FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

base_url = f"{PROXY_URL}/v1"
api_key = f"{FOLDER_ID}@{YANDEX_API_KEY}"

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    temperature=0.9,
    # model="yandexgpt/latest",
    model="llama/latest",
)

question = "Сколько братьев и сколько сестер в семье, если каждая дочь имеет столько сестер, сколько у нее братьев, а сын имеет в два раза меньше братьев, чем сестер?"
right_answer = "В семье 3 брата и 4 сестры."

question = "У Александра есть сын Виктор. У Виктора родился сын Пётр. Кем Александр приходится Петру?"
right_answer = "Александр приходится дедушкой Петру."

question = "Сколько ног у лошади?"
right_answer = "У лошади 4 ноги."

question = "Сколько нужно букв чтобы написать слово молоко?"
right_answer = "Чтобы напистаь слово молоко, нужно 4 буквы."

answer = llm.invoke(question)
print(f'Ответ модели: {answer.content}')

question1 = (
    f'Я задала LLM-модели следующую задачу: {question}.\n'
    f'Модель дала мне следующий ответ: {answer.content}\n'
    f'Правильный ответ:{right_answer}.\n'
    f'Оцени, правильный ли ответ дала мне модель. Отвечай только "Да" или "Нет".'
)

answer_from_llm = llm.invoke(question1)
print(f'Результат теста: {answer_from_llm.content}')