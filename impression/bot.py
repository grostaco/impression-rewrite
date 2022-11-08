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
            text = ' '.join(args)

            embed = discord.Embed()
            embed.color = 0xAF69EE
            embed.title = "Top predictions"
            embed.description = f'Top predictions for query `{text}`\nTrained from authors: {", ".join(f"`{author}`" for author in self.encoder.categories_[0])}'

            async with ctx.typing():
                preds = util.make_prediction(self.model, self.encoder, text)

            preds = tuple(preds)
            authors = '\n'.join([f'`{author}`' for (author, _) in preds])
            confidences = '\n'.join(
                [f'{confidence:.2f}%' for (_, confidence) in preds])

            embed.add_field(name="Name", value=authors)
            embed.add_field(name="Confidence", value=confidences)
            await ctx.send(embed=embed)

        self.add_command(predict)

    async def on_ready(self):
        print('Bot is connected')
