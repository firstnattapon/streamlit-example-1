import telebot

TOKEN = '7248188209:AAFhpA9doHJpESvb2BLkIz6sHTmUeUjOG6E'
chat_id = 7355273754
bot = telebot.TeleBot(TOKEN)

def send_message(chat_id, text):
    bot.send_message(chat_id, text)


text = 'This is a test message from Python.'
send_message(chat_id, text)