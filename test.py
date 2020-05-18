import os
from model import SlotIntentDetectorModel

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


model = SlotIntentDetectorModel()
print('Input intent (AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent)')
while True:
    inp = input('> ')
    intent_cl = model.nlu(inp)
    print('<', intent_cl)