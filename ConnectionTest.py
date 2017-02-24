#!/usr/bin/python
import psycopg2
from shutil import copyfile
from shutil import rmtree
import os

def main():
    #load all records
    conn_string = "host='localhost' dbname='mydb' user='ollie' password='ollie'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT cc_image, x, y, w, h FROM moments WHERE page = 884 AND x > 290 AND y > 580")
    records = cursor.fetchall()
    cursor.close()

    #load all values
    values = []
    with open('/home/ollie/PycharmProjects/PrimeGenerator/test.txt', 'r') as f:
            for line in f:
                values.append(line.split('\n')[0].replace(" ", ""))
                if 'str' in line:
                    break

    primes = list(values)

    # load start of lines
    conn_string = "host='localhost' dbname='mydb' user='ollie' password='ollie'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT cc_image FROM moments WHERE page = 884 AND x < 325 AND y > 580 AND y < 5500 ORDER BY y ASC")
    startOfLines = cursor.fetchall()
    cursor.close()
    startOfLinesImg = []
    for c in startOfLines:
        startOfLinesImg.append(c[0])

    #fix broken orders
    a, b = startOfLinesImg.index('c04229_x0295_y2397.tif'), startOfLinesImg.index('c04266_x0272_y2398.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]
    a, b = startOfLinesImg.index('c04450_x0295_y2480.tif'), startOfLinesImg.index('c04468_x0271_y2481.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]
    a, b = startOfLinesImg.index('c06766_x0269_y3471.tif'), startOfLinesImg.index('c06760_x0294_y3470.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]
    a, b = startOfLinesImg.index('c07862_x0294_y3961.tif'), startOfLinesImg.index('c07885_x0270_y3962.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]
    a, b = startOfLinesImg.index('c08782_x0295_y4374.tif'), startOfLinesImg.index('c08820_x0272_y4375.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]
    a, b = startOfLinesImg.index('c11137_x0270_y5361.tif'), startOfLinesImg.index('c11117_x0294_y5360.tif')
    startOfLinesImg[b], startOfLinesImg[a] = startOfLinesImg[a], startOfLinesImg[b]

    indexes = range(9,188,2)
    for index in sorted(indexes, reverse=True):
        del startOfLinesImg[index]

    #remove 10 from 100 line
    startOfLinesImg.remove(startOfLinesImg[99])
    startOfLinesImg.remove(startOfLinesImg[99])

    #checking to see if line starts are accurate
    #directory = '/home/ollie/PycharmProjects/CharacterSorter/startOfLine'
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #else:
    #    rmtree(directory)
    #    os.makedirs(directory)

    #i = 0;
    #for image in startOfLinesImg:
    #    copyfile('/home/ollie/PycharmProjects/CharacterSorter/page-0884/' + image,
    #                 '/home/ollie/PycharmProjects/CharacterSorter/startOfLine/' +image)
    #    i+=1

    #create folders
    for i in ['0','1','2','3','4','5','6','7','8','9','all']:
        directory = '/home/ollie/PycharmProjects/CharacterSorter/' + i
        if not os.path.exists(directory):
           os.makedirs(directory)
        else:
           rmtree(directory)
           os.makedirs(directory)

    for num in range(0,len(values)):
        #search records for line start
        lineStart = startOfLinesImg[num]
        for record in records:
            if (lineStart == record[0]):
                xValue = record[1]
                yValue = record[2]
                height = record[4]

        characters = []
        for record in records:
            if record[2] < yValue + height and record[2] > yValue - height and record[1] > xValue:
                characters.append(record)

        characters.sort(key=lambda tup: tup[1])

        counter = 0
        while(counter < len(values[num])):
            number = values[num][counter]
            image = characters[counter][0]
            copyfile('/home/ollie/PycharmProjects/CharacterSorter/page-0884/' + image,
                 '/home/ollie/PycharmProjects/CharacterSorter/' + number + '/' + image)

            counter += 1

if __name__ == "__main__":
    main()