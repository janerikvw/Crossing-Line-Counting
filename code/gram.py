from telegram.ext import Updater
from telegram.ext import CommandHandler
import telegram
token = '1100809946:AAE5OLxa10ItfH39DpLyahLYzm9gfFbxGKg'
chat_id = 286859632

bot = telegram.Bot(token=token)

bot.send_message(chat_id=chat_id, text="Send me some info", disable_notification=False)

# updater = Updater(token=token, use_context=True)
# dispatcher = updater.dispatcher
#
# def start(update, context):
#     context.bot.send_message(chat_id=update.effective_chat.id, text="ID:")
#     context.bot.send_message(chat_id=update.effective_chat.id, text=update.effective_chat.id)
#
#
# start_handler = CommandHandler('start', start)
# dispatcher.add_handler(start_handler)
#
# def caps(update, context):
#     text_caps = ' '.join(context.args).upper()
#     context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
#
# caps_handler = CommandHandler('caps', caps)
# dispatcher.add_handler(caps_handler)
#
# updater.start_polling()

