import time
import simplebot

while True:
    try:
        simplebot.run()
        time.sleep(60)
    except:
        print('Simplebot did not run.')
    else:
        print("Simplebot run success.")
