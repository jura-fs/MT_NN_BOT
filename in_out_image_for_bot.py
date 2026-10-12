'''from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import (Message,
                           CallbackQuery,
                           ReplyKeyboardMarkup,
                           KeyboardButton,
                           InlineKeyboardMarkup,
                           InlineKeyboardButton
                           )



router = Router()


@router.message(F.photo)
async def photo(message: Message):
    photo = message.photo[-1]
    file_id = photo.file_id

    await message.answer_photo(file_id, caption='Вот ваше фото')'''