from aiogram import Router, F, Bot
from aiogram.filters import Command
from aiogram.types import (Message,
                           CallbackQuery,
                           ReplyKeyboardMarkup,
                           KeyboardButton,
                           InlineKeyboardMarkup,
                           InlineKeyboardButton
                           )
from MT_NN.Mushoku_Tensei_CNN_BOT.model_for_bot import determinant

# прием разных данных фото, текст и тд
router = Router()

# все кнопки

def get_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='О боте')],
            [KeyboardButton(text='Старт'), KeyboardButton(text='Помощь')]
        ],
        resize_keyboard=True


    )
    return keyboard


# кнопка "Подробная информация по всей вселенной"

def get_inline_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text='Подробная информация по всей вселенной',
                url='https://mushokutensei.fandom.com/ru/wiki/'
            )]
        ]
    )


    return keyboard


# обработка фото

@router.message(F.photo)
async def photo(message: Message, bot: Bot):
    photo = message.photo[-1]
    file_id = photo.file_id

    file = await bot.get_file(file_id)

    photo_bytes = await bot.download_file(file.file_path)

    result =  determinant(photo_bytes)
    await message.answer(result)


@router.callback_query(lambda c: c.data == "info_more")
async def more_info(callback: CallbackQuery):
    await callback.message.answer('Вот более подробная информация')
    await callback.answer()

# функционал кнопки "START"

@router.message(Command("start"))
@router.message(F.text.lower() == 'старт') # <--- эта строка делает так
# что бы были кнопки по типу старт или инфа о боте и
#пользователю не надо писать /help тд
async def start(message: Message):
    await message.answer('Привет!\n я бот по реинкарнации безработного!\n /help для помощи')

# функционал кнопки "HELP"

@router.message(Command("help"))
@router.message(F.text.lower() == 'помощь')
async def help(message: Message):
    await message.answer('Команды!\n'
                         '/start - запуск бота\n'
                         '/help - список команд\n'
                         '/info - инфа о нас\n',
                         reply_markup=get_keyboard())

# функционал кнопки "О боте"

@router.message(Command("info"))
@router.message(F.text.lower() == 'О боте')
async def about(message: Message):
    await message.answer(f'Привет! {message.from_user.full_name}, я рад что ты пользуешься этим ботов ведь он поможет тебе узнать много нового о персонажах реинкарнации безработного'
                         f'\nниже официальный сайт вики реинкарнации безработного',reply_markup=get_inline_keyboard())


# если ввод это не команда или фото

@router.message()
async def other(message: Message):
    await message.answer('Команды!\n'
                         '/start - запуск бота\n'
                         '/help - список команд\n'
                         '/about - инфа о нас\n')


