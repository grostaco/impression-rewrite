from impression.bot import Impression
from dotenv import load_dotenv
import os

load_dotenv()

impression = Impression('assets/text5_clf', 'assets/encoder.pickle')
impression.run(os.getenv('DISCORD_TOKEN'))
