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
    cursor.execute("SELECT cc_image, x, y, w, h FROM moments WHERE page = 885 AND x > 265 AND y > 550")
    records = cursor.fetchall()
    cursor.close()



    #load all values
    values = []
    with open('/home/ollie/work/fyp/test.txt', 'r') as f:
            for line in f:
                values.append(line.split('\n')[0].replace(" ", ""))
                if 'str' in line:
                    break

    print(len(values[0]))


    # load start of lines
    conn_string = "host='localhost' dbname='mydb' user='ollie' password='ollie'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT cc_image FROM moments WHERE page = 885 AND x < 300 AND x > 265 AND y > 550 AND y < 5500 ORDER BY y ASC")
    startOfLines = cursor.fetchall()
    cursor.close()
    startOfLinesImg = []
    for c in startOfLines:
        startOfLinesImg.append(c[0])


    #checking to see if line starts are accurate
    directory = '/home/ollie/work/fyp/characterSorter/startOfLine'
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        rmtree(directory)
        os.makedirs(directory)

    i = 0;
    for image in startOfLinesImg:
        copyfile('/home/ollie/PycharmProjects/AandS-mono600_ccs/page-0885/' + image,
                     '/home/ollie/work/fyp/characterSorter/startOfLine/' +image)
        i+=1

    #create folders
    for i in ['0','1','2','3','4','5','6','7','8','9','all']:
        directory = '/home/ollie/work/fyp/characterSorter/' + i
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

        if len(characters) == 125:
            counter = 0
            while(counter < len(values[num])):
                number = values[num][counter]
                image = characters[counter][0]
                copyfile('/home/ollie/PycharmProjects/AandS-mono600_ccs/page-0885/' + image,'/home/ollie/work/fyp/characterSorter/' + number + '/' + image)

                counter += 1

if __name__ == "__main__":
    main()
