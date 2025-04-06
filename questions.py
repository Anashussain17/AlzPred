
import random

ALZ_QUESTIONS = [
    {"id": 1, "text": "Do you frequently forget recent conversations or events?"},
    {"id": 2, "text": "Do you often misplace items like keys or glasses?"},
    {"id": 3, "text": "Have you noticed a decline in your ability to manage finances?"},
    {"id": 4, "text": "Do you experience difficulty in planning or solving problems?"},
    {"id": 5, "text": "Do you find it challenging to follow a conversation?"},
    {"id": 6, "text": "Have you noticed changes in your mood or personality?"},
    {"id": 7, "text": "Do you have difficulty remembering names of familiar people?"},
    {"id": 8, "text": "Do you often feel confused or disoriented?"},
    {"id": 9, "text": "Have you experienced significant memory lapses recently?"},
    {"id": 10, "text": "Do you struggle with routine tasks that used to be easy?"},
    {"id": 11, "text": "Do you have difficulty understanding visual images and spatial relationships?"},
    {"id": 12, "text": "Are you finding it hard to learn new information?"},
    {"id": 13, "text": "Do you frequently lose track of time?"},
    {"id": 14, "text": "Have you noticed a decrease in your ability to perform daily activities?"},
    {"id": 15, "text": "Do you sometimes forget important dates or events?"},
    {"id": 16, "text": "Are you finding it hard to concentrate on tasks?"},
    {"id": 17, "text": "Do you feel frustrated by your memory lapses?"},
    {"id": 18, "text": "Have you been told that you are more forgetful than usual?"},
    {"id": 19, "text": "Do you sometimes struggle to find the right words when speaking?"},
    {"id": 20, "text": "Have you experienced a noticeable decline in your problem-solving abilities?"},
    {"id": 21, "text": "Do you have difficulty recalling recent appointments?"},
    {"id": 22, "text": "Are you finding it challenging to adapt to changes in your routine?"},
    {"id": 23, "text": "Do you often feel overwhelmed by simple tasks?"},
    {"id": 24, "text": "Have you been experiencing more frequent episodes of confusion?"},
    {"id": 25, "text": "Do you struggle with decision-making?"},
    {"id": 26, "text": "Have you noticed a decline in your overall cognitive abilities?"},
    {"id": 27, "text": "Do you often forget to complete tasks you started?"},
    {"id": 28, "text": "Are you having trouble following instructions?"},
    {"id": 29, "text": "Do you feel that your memory is not as sharp as it used to be?"},
    {"id": 30, "text": "Do you find yourself relying more on reminders and notes than before?"}
]
THRESHOLD = 5  
def get_random_questions():
    return random.sample(ALZ_QUESTIONS, 10)