import time
import livebot

while True:
    try:
        livebot.run()
        time.sleep(60)
    except:
        print('Livebot did not run.')
    else:
        print("Livebot run success.")
