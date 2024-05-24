from fileinput import filename
import json 

size = 128
town_size = 9
file_name = "empty-{}-{}-{}/towns.json".format(size, size, town_size)
with open(file_name, 'r') as town_file:
     fcc_data = json.load(town_file)
     for key in fcc_data:
        file_name = "maps{}-{}/town_".format(size, town_size)+key+".map"
        with open(file_name, 'w') as f:
            f.write("type octile\n")
            f.write("height " + str(len(fcc_data[key]['map'])) + "\n")
            f.write("width " + str(len(fcc_data[key]['map'][0]))+ "\n")
            f.write("map\n")
            for line in fcc_data[key]['map']:
                line_list = list(line)
                for i in range(0,len(line_list)):
                    if line_list[i]=='L':
                        line_list[i] = '@'
                line = "".join(line_list)
                f.write(line+"\n")
        file_name = "maps{}-{}/town_info_".format(size, town_size)+key+".map"
        with open(file_name,'w') as f:
            f.write(str(fcc_data[key]['town id'])+"\n")
            f.write("height " + str(len(fcc_data[key]['map'])) + "\n")
            f.write("width " + str(len(fcc_data[key]['map'][0]))+ "\n")
            f.write(str(fcc_data[key]['origin'][0]) + " " + str(fcc_data[key]['origin'][1])+"\n")
            for line in fcc_data[key]['map']:
                f.write(line+"\n")
            


# read the json file