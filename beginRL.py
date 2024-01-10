import time
import rlbot

while True:
    try:
        rlbot.run()
        time.sleep(60)
    except:
        print('rlbot did not run.')
    else:
        print("rlbot run success.")
