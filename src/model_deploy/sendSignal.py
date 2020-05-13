import numpy as np
import serial
import time

waitTime = 0.1

song1 =np.array(
[
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261
]
)

noteLength1 =np.array(
[
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2
]
)

song2 =np.array(
[
  261, 294, 330, 330, 330, 440, 392,
  261, 349, 349, 294, 330, 330, 330,
  294, 330, 440, 330, 294, 392, 392,
  294, 392, 330, 261, 294, 294, 261,
  330, 349, 294, 261, 330, 392, 440,
  261, 294, 392, 261, 330, 330, 440
]
)

noteLength2 =np.array(
[
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1
]
)

song3 =np.array(
[
  330, 261, 261, 294, 294, 392, 392, 
  392, 294, 294, 330, 330, 261, 261,
  261, 330, 330, 294, 294, 392, 392, 
  392, 294, 294, 330, 330, 261, 261,
  330, 261, 261, 294, 294, 392, 392, 
  392, 294, 294, 330, 330, 261, 261
]
)

noteLength3 =np.array(
[
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 1, 1, 1
]
)

song1 = song1 /500

noteLength1 = noteLength1 /4

song2 = song2/500

noteLength2 = noteLength2 /4

song3 = song3 /500

noteLength3 = noteLength3 /4

#song4 = song4 /500

#noteLength4 = noteLength4 /4

# output formatter

a = 1

formatter = lambda x: "%.3f" % x

serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)

while a != 2:
    b = s.readline()
    print(b)
    c = (b)
    if b[0] == 49:
        print("Sending Song1 ...")
        for data in song1:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength1:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        #s.close()
        print("Song1 sended")
    if b[0] == 50:
        print("Sending Song2 ...")
        for data in song2:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength2:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        #s.close()
        print("Song2 sended")
    if b[0] == 51:
        print("Sending Song3 ...")
        for data in song3:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength3:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        print("Song3 sended")
