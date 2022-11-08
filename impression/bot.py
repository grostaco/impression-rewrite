import discord
from discord.ext import commands

from classify import model, util


class Impression(commands.Bot):
    def __init__(self, model_path: str, encoder_path: str, prefix=';'):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix=prefix, intents=intents)
        self.model = model.load_model(model_path)
        self.encoder = util.load_encoder(encoder_path)

        @commands.command(name="predict")
        async def predict(ctx: commands.context.Context, *args):
            async with ctx.typing():
                text = ' '.join(args)
                preds = util.make_prediction(self.model, self.encoder, text)
            await ctx.send('\n'.join([f'Predicted `{author}` with confidence `{confidence * 100:.2f}%`' for (author, confidence) in preds]))

        self.add_command(predict)

    async def on_ready(self):
        print('Bot is connected')
