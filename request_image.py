import requests as r
import os
import sys

if __name__=="__main__":
    filename = sys.argv[1]
    print(filename)
    os.system("docker cp {0} $(docker ps -q):/root/".format(filename))
    resp = r.get('http://localhost:5000/detect_objects/', params={'filename': filename})

    with open('picture_out.png', 'wb') as f:
        f.write(resp.content)
