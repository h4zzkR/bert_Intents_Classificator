import os
from control import main

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


model = main(ckp_path='model/intents_cls_e2_bs32', test=False)
print('Input intent (AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent')
while True:
    inp = input('> ')
    intent_cl = model.classify(inp)
    print('>', intent_cl)