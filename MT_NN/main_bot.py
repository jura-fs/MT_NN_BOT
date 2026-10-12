from os import getenv
import asyncio
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from MT_NN.Mushoku_Tensei_CNN_BOT.router import router

# загрузка ключа
load_dotenv()
TOKEN = getenv('BOT_TOKEN')


dp = Dispatcher()
dp.include_router(router)



# запуск
async  def main():
    bot = Bot(token=TOKEN)
    print('бот запущен')


    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
