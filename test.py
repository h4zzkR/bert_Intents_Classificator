import os
from control import main

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


model = main()
print('Input intent (AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent)')
while True:
    inp = input('> ')
    intent_cl = model.nlu(inp)
    print('<', intent_cl)